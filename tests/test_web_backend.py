"""Browser-backend tests: arbitrary uploads, pairing, training, inference, API.

The synthetic frames are shaped like a HIKMICRO export but use a smaller
sensor, which is the point: nothing in the backend may assume 256 x 192.
The server test runs the real HTTP stack on an ephemeral port with a temporary
board library, so it also covers multipart parsing and job polling.
"""

from __future__ import annotations

import json
import threading
import urllib.request
import uuid
from pathlib import Path

import numpy as np
import pytest

from tdp.io import HMParseError, parse_hm_csv
from tdp.web import registry, training
from tdp.web.inference import predict_frame
from tdp.web.server import (
    build_server,
    display_board_mask,
    encode_field,
    parse_multipart,
)
from tdp.web.sessions import (
    build_session,
    powered_prefix,
    register_window,
    state_pairs,
)

ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------
# synthetic HIKMICRO frames
# --------------------------------------------------------------------------
def _csv(frame: np.ndarray, name: str) -> bytes:
    height, width = frame.shape
    lines = ["坐标Y\\X," + ",".join(str(index) for index in range(width))]
    for row in range(height):
        lines.append(str(row) + "," + ",".join(f"{value:.2f}" for value in frame[row]))
    return ("\n".join(lines) + "\n").encode("utf-8")


def _synthetic_sessions(count: int = 10):
    """A board that heats, plus a low-emissivity lid the bare camera reads cold."""
    bare: list[tuple[str, bytes]] = []
    painted: list[tuple[str, bytes]] = []
    for index in range(count):
        level = 5.0 + 25.0 * (1.0 - np.exp(-index / 3.0))
        bare_frame = np.full((200, 160), 25.0, dtype=np.float32)
        painted_frame = bare_frame.copy()
        bare_frame[35:165, 25:135] = 25.0 + level
        painted_frame[35:165, 25:135] = 25.0 + level
        true_lid = 25.0 + level + 8.0
        bare_frame[75:105, 50:85] = 25.0 + 0.5 * level + 2.0
        painted_frame[75:105, 50:85] = true_lid
        name = f"HM2026010100{index:04d}.csv"
        bare.append((name, _csv(bare_frame, name)))
        painted.append((name, _csv(painted_frame, name)))
    return bare, painted


# --------------------------------------------------------------------------
# parsing and preprocessing
# --------------------------------------------------------------------------
def test_upload_parser_accepts_a_non_standard_sensor_grid():
    frame = np.full((64, 48), 31.5, dtype=np.float32)
    data = _csv(frame, "upload.csv")

    parsed = parse_hm_csv(data, "upload.csv", expect_shape=None)
    assert parsed.temps.shape == (64, 48)
    assert np.allclose(parsed.temps, 31.5)

    with pytest.raises(HMParseError):
        parse_hm_csv(data, "upload.csv", expect_shape=(256, 192))


def test_session_locates_and_rectifies_an_arbitrary_board():
    bare, _ = _synthetic_sessions(4)
    session, failures, _warnings = build_session(bare, board_size_mm=(85.0, 56.0))
    assert not failures
    assert session.grid_hw == (157, 103)
    assert session.theta_k.shape == (4, 157, 103)
    assert np.isfinite(session.theta_k).all()
    assert session.theta_k.max() > 20.0

    # Unknown physical outline: keep the measured aspect and report no forced
    # residual.  This is what the browser sends when the size field is empty.
    free, _, _ = build_session(bare, board_size_mm=None)
    assert free.window["aspect_residual"] is None
    assert free.grid_hw[0] > 32 and free.grid_hw[1] > 32


def test_pairing_uses_the_low_emissivity_mask_and_the_state_order():
    bare, painted = _synthetic_sessions(10)
    bare_session, _, _ = build_session(bare, board_size_mm=(85.0, 56.0))
    painted_session, _, _ = build_session(painted, board_size_mm=(85.0, 56.0))
    pairs = state_pairs(bare_session, painted_session)
    assert pairs["x_theta"].shape == pairs["y_theta"].shape
    assert pairs["diagnostics"]["n_pairs"] == 10
    unknown = pairs["unknown_mask"]
    assert unknown.mean() > 0.0
    # The lid is much colder on the bare camera than on the painted reference.
    assert pairs["diagnostics"]["masked_bias_k"] > 3.0


def test_registration_finds_a_shifted_board_instead_of_trusting_the_detector():
    """A painted sheet can be 23 % larger and 20 px away from the PCB beneath it.

    The Orange Pi tape run is the measured example.  This synthetic version
    moves the same component pattern by (12, -9) px and asks the structural
    matcher to find it from a deliberately wrong starting window.
    """
    from tdp.data.crop import estimate_ambient
    from tdp.web.sessions import UploadedFrame

    def board_frame(shift_r: int, shift_c: int) -> np.ndarray:
        frame = np.full((200, 160), 25.0, dtype=np.float32)
        frame[35 + shift_r:165 + shift_r, 25 + shift_c:135 + shift_c] = 45.0
        frame[75 + shift_r:105 + shift_r, 50 + shift_c:85 + shift_c] = 53.0
        for row, col in ((50, 40), (50, 110), (140, 40), (140, 110), (95, 30)):
            frame[row + shift_r:row + shift_r + 6, col + shift_c:col + shift_c + 6] = 40.0
        return frame

    def encoded(frame: np.ndarray, name: str) -> tuple[str, bytes]:
        return name, _csv(frame, name)

    reference, failures, _ = build_session(
        [encoded(board_frame(0, 0), "reference.csv")], board_size_mm=None
    )
    assert not failures
    source_frame = board_frame(12, -9)
    source = UploadedFrame(
        name="shifted.csv", digest="", timestamp=None,
        temps=source_frame, ambient_c=estimate_ambient(source_frame),
    )
    registered = register_window(
        reference.theta_k[reference.reference_index],
        reference.window,
        source,
        reference.grid_hw,
        initial_window=reference.window,
    )
    true_center = np.asarray(reference.window["center_rc"]) + np.asarray([12.0, -9.0])
    assert registered["score"] > 0.8
    assert np.linalg.norm(np.asarray(registered["center_rc"]) - true_center) < 3.0


def _mean_sessions(level: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    theta = np.broadcast_to(level[:, None, None], (len(level), 8, 8)).astype(np.float64)
    return theta, np.ones((8, 8), dtype=bool)


def test_powered_cut_finds_a_real_load_stop():
    n_frames = 80
    times = np.arange(n_frames, dtype=np.float64) * 6.0
    level = 5.0 + 25.0 * (1.0 - np.exp(-np.arange(n_frames) / 8.0))
    stop = 40
    level[stop:] = level[stop] * np.exp(-np.arange(n_frames - stop) * 0.03)
    theta, valid = _mean_sessions(level)
    indices, report = powered_prefix(theta, valid, times)
    assert report["cut"]
    assert abs(int(report["drop_frame"]) - (stop - 1)) <= 2
    assert len(indices) <= stop + 2


def test_powered_cut_ignores_a_recording_gap_with_no_load_stop():
    n_frames = 60
    times = np.arange(n_frames, dtype=np.float64) * 6.0
    times[30:] += 180.0  # the camera was paused for three minutes
    level = np.concatenate([
        5.0 + 25.0 * (1.0 - np.exp(-np.arange(30) / 8.0)),
        np.full(30, 2.0),
    ])
    theta, valid = _mean_sessions(level)
    indices, report = powered_prefix(theta, valid, times)
    assert not report["cut"]
    assert len(indices) == n_frames


def test_display_mask_removes_a_small_border_background_only():
    raw = np.full((20, 20), 20.0)
    raw[:6, :6] = 1.0  # a background corner touching the border
    board = np.ones((20, 20), dtype=bool)
    display = display_board_mask(raw, board)
    assert not display[:3, :3].any()
    assert display[10:, 10:].all()
    assert display.sum() > 0.9 * board.sum()

    # A genuinely cold board (a low region over 10 % of the crop) is kept.
    all_cold = np.full((20, 20), 1.0)
    assert display_board_mask(all_cold, board).all()


def test_encode_field_fills_a_masked_corner_for_display_but_keeps_the_mask():
    field = np.arange(16, dtype=np.float64).reshape(4, 4)
    board = np.ones((4, 4), dtype=bool)
    board[0, 0] = False
    encoded = encode_field(field, board, 20.0)
    display = np.asarray(encoded["display"]).reshape(4, 4)
    values = np.asarray(encoded["values"]).reshape(4, 4)
    assert display[0, 0] != -32768
    assert values[0, 0] == 0
    packed = np.unpackbits(
        np.frombuffer(__import__("base64").b64decode(encoded["board_mask"]), dtype=np.uint8),
        count=16,
    ).reshape(4, 4)
    assert not packed[0, 0]


def test_web_canvas_draws_the_board_in_source_orientation():
    """The recurring black top corners started with a 180-degree canvas flip.

    The canonical crop already has row 0 at the top.  Mirrored axes put the
    data bottom (the port edge on case10) at the display top.  This static
    check is deliberate: there is no browser test runner here, and the flip
    must not return silently.
    """
    source = (ROOT / "web/app.js").read_text(encoding="utf-8")
    assert "const x = width - 1 - col" not in source
    assert "const y = height - 1 - row" not in source
    assert "const x = col;" in source
    assert "const y = row;" in source
    assert "const row = view.height - 1 - y" not in source
    assert "for (let row = view.height - 1; row >= 0; row--)" not in source


# --------------------------------------------------------------------------
# multipart
# --------------------------------------------------------------------------
def test_multipart_parser_keeps_fields_and_files_apart():
    boundary = "----testboundary"
    body = (
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"name\"\r\n\r\n"
        "Board\r\n"
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"bare\"; filename=\"a.csv\"\r\n"
        "Content-Type: text/csv\r\n\r\n"
        "1,2,3\r\n"
        f"--{boundary}--\r\n"
    ).encode()
    form = parse_multipart(body, f"multipart/form-data; boundary={boundary}")
    assert form.text("name") == "Board"
    files = form.files("bare")
    assert files == [("a.csv", b"1,2,3")]


def test_board_ids_cannot_escape_the_library(tmp_path, monkeypatch):
    monkeypatch.setattr(registry, "BOARD_ROOT", tmp_path)
    for unsafe in ("../secret", "a/b", "..", ""):
        with pytest.raises(KeyError):
            registry.get_board(unsafe)


# --------------------------------------------------------------------------
# training and inference on synthetic data
# --------------------------------------------------------------------------
def test_training_lifts_the_low_emissivity_area_and_infers(tmp_path, monkeypatch):
    monkeypatch.setattr(registry, "BOARD_ROOT", tmp_path)
    monkeypatch.setattr(training, "BOARD_ROOT", tmp_path)
    bare, painted = _synthetic_sessions(10)
    board = training.train_board(
        name="Synthetic Lab Board",
        bare_files=bare,
        painted_files=painted,
        board_size_mm=(85.0, 56.0),
        epochs=30,
        report=lambda *args, **kwargs: None,
        device="cpu",
    )
    assert board.available
    assert board.grid_hw == (157, 103)
    assert board.metrics["final"]["masked_mae_k"] < board.metrics["pairing"]["masked_mae_k"] - 2.0

    result = predict_frame(board, bare[-1][0], bare[-1][1], device="cpu")
    unknown = result["unknown_mask"]
    shift = float(np.mean(result["predicted"][unknown] - result["raw"][unknown]))
    assert shift > 2.0
    assert result["signature_score"] is not None and result["signature_score"] > 0.8


# --------------------------------------------------------------------------
# HTTP API
# --------------------------------------------------------------------------
def _multipart(fields: dict[str, str], files: dict[str, list[tuple[str, bytes]]]):
    boundary = "----tdp" + uuid.uuid4().hex
    chunks: list[bytes] = []
    for key, value in fields.items():
        chunks.append(
            f'--{boundary}\r\nContent-Disposition: form-data; name="{key}"\r\n\r\n{value}\r\n'.encode()
        )
    for key, items in files.items():
        for name, data in items:
            chunks.append(
                f'--{boundary}\r\nContent-Disposition: form-data; name="{key}"; '
                f'filename="{name}"\r\nContent-Type: text/csv\r\n\r\n'.encode()
            )
            chunks.append(data)
            chunks.append(b"\r\n")
    chunks.append(f"--{boundary}--\r\n".encode())
    return boundary, b"".join(chunks)


def _request(url: str, data: bytes | None = None, content_type: str | None = None, method: str | None = None):
    request = urllib.request.Request(url, data=data, method=method)
    if content_type:
        request.add_header("Content-Type", content_type)
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.status, json.loads(response.read().decode("utf-8"))


def test_http_api_adds_a_board_trains_it_and_infers(tmp_path, monkeypatch):
    monkeypatch.setattr(registry, "BOARD_ROOT", tmp_path)
    monkeypatch.setattr(training, "BOARD_ROOT", tmp_path)
    server = build_server("127.0.0.1", 0, "cpu")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        bare, painted = _synthetic_sessions(6)
        boundary, body = _multipart(
            {"name": "API Synthetic", "epochs": "2", "board_size_mm": "85x56"},
            {"bare": bare, "painted": painted},
        )
        status, payload = _request(
            f"{base}/api/boards", body, f"multipart/form-data; boundary={boundary}"
        )
        assert status == 202
        job_id = payload["job"]["id"]

        job = payload["job"]
        for _ in range(120):
            _status, job = _request(f"{base}/api/jobs/{job_id}")
            if job["status"] in {"done", "error"}:
                break
            threading.Event().wait(0.25)
        assert job["status"] == "done", job
        assert job["board"]["grid_hw"] == [157, 103]
        board_id = job["board"]["id"]

        _status, listing = _request(f"{base}/api/boards")
        assert [board["id"] for board in listing["boards"]] == [board_id]

        boundary, body = _multipart({"board_id": board_id}, {"file": [("frame.csv", bare[0][1])]})
        _status, payload = _request(
            f"{base}/api/infer", body, f"multipart/form-data; boundary={boundary}"
        )
        frame = payload["frame"]
        assert frame["width"] * frame["height"] == len(frame["raw"]["values"])
        assert frame["predicted"]["values"] != frame["raw"]["values"]

        status, _payload = _request(f"{base}/api/boards/{board_id}", method="DELETE")
        assert status == 200
        _status, listing = _request(f"{base}/api/boards")
        assert listing["boards"] == []
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


# --------------------------------------------------------------------------
# regression against the saved case10 prediction
# --------------------------------------------------------------------------
def test_preset_inference_reproduces_the_saved_case10_prediction(tmp_path, monkeypatch):
    source = ROOT / "outputs/thermal/unet_case10/train_both/model_best.pt"
    masks = ROOT / "outputs/thermal/unet_case10/paired_masks.npz"
    saved = ROOT / "outputs/thermal/unet_case10/presentation/predicted_maps_case10.npz"
    sample = ROOT / "web/samples/raspberry-pi-4b/HM20260726165220.csv"
    if not all(path.exists() for path in (source, masks, saved, sample)):
        pytest.skip("saved case10 model, masks, predictions or sample frame are absent")
    monkeypatch.setattr(registry, "BOARD_ROOT", tmp_path)
    events = registry.ensure_presets()
    assert any(event["status"].startswith("created") for event in events)
    board = registry.get_board("raspberry-pi-4b")
    result = predict_frame(board, sample.name, sample.read_bytes(), device="cpu")
    with np.load(saved) as archive:
        target = archive["predicted_theta_k"][0]
        board_mask = archive["board_mask"]
    error = np.abs(result["predicted"] - target)[board_mask].mean()
    assert error < 0.01
