"""A small standard-library HTTP server and JSON API for the thermal studio.

The frontend used to be a static playback: its "train" button replayed a saved
Orange Pi animation and its inference step compared SHA-256 hashes against five
bundled frames.  This server replaces both with real work:

    GET    /api/health
    GET    /api/boards
    GET    /api/boards/<id>
    GET    /api/boards/<id>/preview
    DELETE /api/boards/<id>
    POST   /api/boards              multipart: name, bare[], painted[], epochs
    GET    /api/jobs/<id>
    POST   /api/infer               multipart: board_id, file (CSV or JPEG)

Training runs in a background thread and reports progress through a job record,
so the browser can draw the real loss curve while the model fits.  Inference is
synchronous: one frame is cheap and the user is waiting for it.

Only the local machine is served by default.  There is no authentication; do
not bind this to a public interface.
"""

from __future__ import annotations

import argparse
import json
import math
import mimetypes
import re
import sys
import threading
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter, label

from tdp.web import registry
from tdp.web.inference import predict_frame
from tdp.web.training import stage_labels, train_board

REPO_ROOT = Path(__file__).resolve().parents[2]
WEB_ROOT = REPO_ROOT / "web"
MAX_UPLOAD_BYTES = 1_500_000_000
MAX_JOB_LOG = 400


# --------------------------------------------------------------------------
# multipart/form-data
# --------------------------------------------------------------------------
@dataclass
class FormPart:
    name: str
    filename: str | None
    content: bytes

    @property
    def text(self) -> str:
        return self.content.decode("utf-8", errors="replace")


class Form:
    def __init__(self, parts: list[FormPart]) -> None:
        self.parts = parts

    def text(self, name: str, default: str = "") -> str:
        for part in self.parts:
            if part.name == name and part.filename is None:
                return part.text.strip()
        return default

    def files(self, *names: str) -> list[tuple[str, bytes]]:
        wanted = set(names)
        return [
            (part.filename or f"{part.name}.csv", part.content)
            for part in self.parts
            if part.name in wanted and part.filename is not None
        ]

    def file(self, name: str) -> tuple[str, bytes] | None:
        for part in self.parts:
            if part.name == name and part.filename is not None:
                return part.filename, part.content
        return None


def parse_multipart(body: bytes, content_type: str) -> Form:
    """Minimal multipart parser, independent of the removed ``cgi`` module."""
    match = re.search(r'boundary="?([^";]+)"?', content_type)
    if not match:
        raise ValueError("multipart body has no boundary")
    boundary = match.group(1).encode("ascii")
    delimiter = b"--" + boundary
    parts: list[FormPart] = []
    for chunk in body.split(delimiter):
        if not chunk or chunk.startswith(b"--"):
            continue
        if chunk.startswith(b"\r\n"):
            chunk = chunk[2:]
        if chunk.endswith(b"\r\n"):
            chunk = chunk[:-2]
        if not chunk:
            continue
        head, separator, content = chunk.partition(b"\r\n\r\n")
        if not separator:
            continue
        headers: dict[str, str] = {}
        for line in head.split(b"\r\n"):
            key, _, value = line.partition(b":")
            if value:
                headers[key.strip().lower().decode("ascii", "replace")] = value.strip().decode(
                    "utf-8", "replace"
                )
        disposition = headers.get("content-disposition", "")
        name_match = re.search(r'name="([^"]*)"', disposition)
        file_match = re.search(r'filename="([^"]*)"', disposition)
        if not name_match:
            continue
        parts.append(
            FormPart(
                name=name_match.group(1),
                filename=None if file_match is None else unquote(file_match.group(1)),
                content=content,
            )
        )
    return Form(parts)


# --------------------------------------------------------------------------
# jobs
# --------------------------------------------------------------------------
def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def json_safe(value: Any) -> Any:
    """Replace non-finite floats before they reach ``JSON.parse``.

    Python's encoder emits bare ``NaN``/``Infinity``, which is not valid JSON
    and makes a browser reject an otherwise successful response.  A board with
    no low-emissivity pixels can produce an empty-mask mean, so this is a real
    case, not a hypothetical one.
    """
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    return value


@dataclass
class Job:
    id: str
    name: str
    status: str = "queued"
    stage: str = "queued"
    detail: str = "等待开始"
    progress: float = 0.0
    logs: list[dict] = field(default_factory=list)
    history: list[dict] = field(default_factory=list)
    board: dict | None = None
    error: str | None = None
    created_at: str = field(default_factory=_now)

    def update(self, stage: str, message: str, progress: float, metrics: dict | None) -> None:
        self.stage = stage
        self.detail = message
        self.progress = float(progress)
        self.logs.append({"at": _now(), "stage": stage, "message": message})
        if len(self.logs) > MAX_JOB_LOG:
            self.logs = self.logs[-MAX_JOB_LOG:]
        if metrics is not None:
            record = {
                key: (float(value) if isinstance(value, (int, float, np.floating)) else value)
                for key, value in metrics.items()
            }
            record["epoch"] = int(metrics.get("epoch", 0))
            self.history.append(record)

    def snapshot(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "status": self.status,
            "stage": self.stage,
            "detail": self.detail,
            "progress": round(self.progress, 1),
            "logs": list(self.logs),
            "history": list(self.history),
            "board": self.board,
            "error": self.error,
            "created_at": self.created_at,
            "stages": stage_labels(),
        }


class JobManager:
    """One training at a time; each job owns a snapshot the browser can poll."""

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tdp-train")

    def get(self, job_id: str) -> Job:
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            raise KeyError(job_id)
        return job

    def submit(self, name: str, task) -> Job:
        job = Job(id=uuid.uuid4().hex[:12], name=name)
        with self._lock:
            self._jobs[job.id] = job
        self._executor.submit(self._run, job, task)
        return job

    def _run(self, job: Job, task) -> None:
        job.status = "running"

        def report(stage: str, message: str, progress: float, metrics: dict | None = None) -> None:
            job.update(stage, message, progress, metrics)

        try:
            board = task(report)
            job.board = board.public_dict()
            job.status = "done"
            job.progress = 100.0
            job.stage = "done"
            job.detail = "板型已加入板型库"
        except Exception as exc:  # a failed upload must not kill the server
            job.status = "error"
            job.error = str(exc)
            job.detail = "训练失败"
            job.logs.append(
                {
                    "at": _now(),
                    "stage": "error",
                    "message": f"{type(exc).__name__}: {exc}",
                }
            )
            traceback.print_exc()


# --------------------------------------------------------------------------
# array encoding for the browser
# --------------------------------------------------------------------------
def _pack_mask(mask: np.ndarray) -> str:
    import base64

    return base64.b64encode(np.packbits(np.asarray(mask, dtype=bool).ravel())).decode("ascii")


def _tenths(field: np.ndarray) -> list[int]:
    values = np.asarray(field, dtype=np.float64)
    encoded = np.where(np.isfinite(values), np.rint(values * 10.0), -32768.0)
    return encoded.astype(np.int16).ravel().tolist()


# A display-only background test.  The board is powered and sits well above
# ambient in every session here; a near-ambient region connected to the image
# border is background, not a cool board pixel.  The fraction guard keeps a
# genuinely cold board from being painted away.
DISPLAY_BACKGROUND_K = 3.0
DISPLAY_BACKGROUND_MAX_FRACTION = 0.10
DISPLAY_BACKGROUND_MIN_PX = 12


def display_board_mask(raw: np.ndarray, board: np.ndarray) -> np.ndarray:
    """Pixels that actually carry board data, for presentation only.

    A rectangular crop can contain background: case10's USB/Ethernet ports
    protrude past the PCB at the bottom, and the Orange Pi board has a cut
    top-right corner.  The measurements must keep that background out, but the
    heatmap should not show a black hole there.  This mask removes only a
    near-ambient region that touches the image border and is small relative to
    the board, so it cannot swallow a cold but real board.
    """
    values = np.asarray(raw, dtype=np.float64)
    board = np.asarray(board, dtype=bool)
    valid = np.isfinite(values) & board
    if not valid.any():
        return valid
    low = valid & (values < DISPLAY_BACKGROUND_K)
    if int(low.sum()) < DISPLAY_BACKGROUND_MIN_PX:
        return valid
    labels, count = label(low, structure=np.ones((3, 3), dtype=bool))
    if count == 0:
        return valid
    border = np.zeros(low.shape, dtype=bool)
    border[0, :] = border[-1, :] = border[:, 0] = border[:, -1] = True
    border_labels = np.unique(labels[border & low])
    border_labels = border_labels[border_labels > 0]
    if border_labels.size == 0:
        return valid
    background = np.isin(labels, border_labels)
    if int(background.sum()) > DISPLAY_BACKGROUND_MAX_FRACTION * int(valid.sum()):
        return valid
    return valid & ~background


def _display_field(field: np.ndarray, board: np.ndarray) -> np.ndarray:
    """Fill missing pixels for display only, exactly as the demo assets did."""
    values = np.asarray(field, dtype=np.float64)
    valid = np.isfinite(values) & board
    if valid.all():
        return values
    if not valid.any():
        return np.zeros_like(values)
    _, indices = distance_transform_edt(~valid, return_indices=True)
    completed = values.copy()
    completed[~valid] = values[indices[0][~valid], indices[1][~valid]]
    smoothed = gaussian_filter(completed, sigma=2.0, mode="nearest")
    completed[~valid] = smoothed[~valid]
    return completed


def encode_field(field: np.ndarray, board: np.ndarray, ambient_c: float) -> dict:
    values = np.asarray(field, dtype=np.float64)
    display = _display_field(values, board)
    return {
        "ambient_c": float(ambient_c),
        "width": int(values.shape[1]),
        "height": int(values.shape[0]),
        "values": _tenths(values),
        "display": _tenths(display),
        "board_mask": _pack_mask(board),
    }


def encode_frame(result: dict) -> dict:
    board = np.asarray(result["board_mask"], dtype=bool)
    valid = np.asarray(result.get("valid_mask", np.isfinite(result["raw"])), dtype=bool)
    model_board = board & valid
    display_board = display_board_mask(result["raw"], model_board)
    return {
        "board_id": result["board_id"],
        "source_name": result["source_name"],
        "width": int(result["grid_hw"][1]),
        "height": int(result["grid_hw"][0]),
        "ambient_c": float(result["ambient_c"]),
        "window": result["window"],
        "signature_score": result["signature_score"],
        "warnings": result["warnings"],
        "model_note": result["model_note"],
        "raw": encode_field(result["raw"], display_board, result["ambient_c"]),
        "predicted": encode_field(result["predicted"], display_board, result["ambient_c"]),
        "board_mask": _pack_mask(model_board),
        "unknown_mask": _pack_mask(np.asarray(result["unknown_mask"], dtype=bool)),
        "valid_mask": _pack_mask(valid),
    }


def encode_preview(board) -> dict:
    """A library-card preview for a board, generated once and cached."""
    preview_path = board.preview_path
    if preview_path.exists():
        with np.load(preview_path) as data:
            preview = {key: np.array(data[key]) for key in data.files}
    elif board.sample_path:
        sample = registry.abs_path(board.sample_path)
        result = predict_frame(board, sample.name, sample.read_bytes())
        preview = {
            "raw": result["raw"],
            "predicted": result["predicted"],
            "ambient_c": np.float32(result["ambient_c"]),
            "board_mask": np.asarray(result["board_mask"], dtype=bool),
            "unknown_mask": np.asarray(result["unknown_mask"], dtype=bool),
            "signature_score": np.float32(result["signature_score"] or 0.0),
        }
        np.savez_compressed(preview_path, **preview)
        board.preview = preview
    else:
        raise FileNotFoundError("board has no preview and no sample frame")
    board_mask = np.asarray(preview["board_mask"], dtype=bool)
    display_board = display_board_mask(preview["raw"], board_mask)
    return {
        "ambient_c": float(preview["ambient_c"]),
        "width": int(board.grid_hw[1]),
        "height": int(board.grid_hw[0]),
        "raw": encode_field(preview["raw"], display_board, float(preview["ambient_c"])),
        "predicted": encode_field(
            preview["predicted"], display_board, float(preview["ambient_c"])
        ),
        "board_mask": _pack_mask(board_mask),
    }


# --------------------------------------------------------------------------
# HTTP handler
# --------------------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    server_version = "ThermalStudio/1.0"
    protocol_version = "HTTP/1.1"
    jobs: JobManager
    device: str = "auto"

    # -- helpers ---------------------------------------------------------
    def _json(self, payload: Any, status: int = 200) -> None:
        body = json.dumps(json_safe(payload), ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _error(self, message: str, status: int = 400) -> None:
        self._json({"error": message}, status)

    def _read_body(self) -> bytes:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            raise ValueError("empty request body")
        if length > MAX_UPLOAD_BYTES:
            raise ValueError("upload is larger than the 1.5 GB limit")
        body = bytearray()
        remaining = length
        while remaining > 0:
            chunk = self.rfile.read(min(remaining, 1024 * 1024))
            if not chunk:
                break
            body.extend(chunk)
            remaining -= len(chunk)
        if remaining != 0:
            raise ValueError("request body ended before Content-Length")
        return bytes(body)

    def _form(self) -> Form:
        return parse_multipart(self._read_body(), self.headers.get("Content-Type", ""))

    def log_message(self, fmt: str, *args: Any) -> None:  # quieter access log
        if not self.path.startswith("/api/"):
            return
        sys.stderr.write(f"{self.address_string()} - {fmt % args}\n")

    # -- routing ---------------------------------------------------------
    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        if path.startswith("/api/"):
            self._api_get(path)
            return
        self._static(path)

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        if path == "/api/boards":
            self._create_board()
        elif path == "/api/infer":
            self._infer()
        else:
            self._error("unknown endpoint", 404)

    def do_DELETE(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        match = re.fullmatch(r"/api/boards/([^/]+)", path)
        if not match:
            self._error("unknown endpoint", 404)
            return
        try:
            registry.delete_board(match.group(1))
        except KeyError:
            self._error("board not found", 404)
        except PermissionError as exc:
            self._error(str(exc), 403)
        else:
            self._json({"deleted": match.group(1)})

    # -- API -------------------------------------------------------------
    def _api_get(self, path: str) -> None:
        try:
            if path == "/api/health":
                boards = registry.list_boards()
                self._json(
                    {
                        "ok": True,
                        "name": "thermal-studio",
                        "boards": len(boards),
                        "device": self.device,
                        "stages": stage_labels(),
                    }
                )
            elif path == "/api/boards":
                self._json(
                    {
                        "boards": [board.summary() for board in registry.list_boards()],
                        "stages": stage_labels(),
                    }
                )
            elif path.startswith("/api/boards/"):
                rest = path[len("/api/boards/"):]
                if rest.endswith("/preview"):
                    board = registry.get_board(rest[: -len("/preview")])
                    self._json(encode_preview(board))
                else:
                    board = registry.get_board(rest)
                    self._json(board.public_dict())
            elif path.startswith("/api/jobs/"):
                job = self.jobs.get(path[len("/api/jobs/"):])
                self._json(job.snapshot())
            else:
                self._error("unknown endpoint", 404)
        except KeyError:
            self._error("not found", 404)
        except FileNotFoundError as exc:
            self._error(str(exc), 404)
        except Exception as exc:
            traceback.print_exc()
            self._error(f"{type(exc).__name__}: {exc}", 500)

    def _board_size(self, form: Form) -> tuple[float, float] | None:
        raw = form.text("board_size_mm")
        if raw:
            parts = re.split(r"[xX×,\s]+", raw.strip())
            parts = [part for part in parts if part]
            if len(parts) != 2:
                raise ValueError("板体尺寸格式应为 长x短，例如 85x56")
            return (float(parts[0]), float(parts[1]))
        width, height = form.text("board_width_mm"), form.text("board_height_mm")
        if width and height:
            return (float(width), float(height))
        return None

    def _create_board(self) -> None:
        try:
            form = self._form()
            name = form.text("name")
            if not name:
                self._error("请填写板型名称")
                return
            bare = form.files("bare", "bare[]")
            painted = form.files("painted", "painted[]")
            if not bare or not painted:
                self._error("需要同时上传裸板与黑化板 CSV 或 JPEG")
                return
            epochs = int(form.text("epochs", "80") or 80)
            board_size = self._board_size(form)
        except (ValueError, KeyError) as exc:
            self._error(str(exc))
            return

        def task(report):
            return train_board(
                name=name,
                bare_files=bare,
                painted_files=painted,
                board_size_mm=board_size,
                epochs=epochs,
                report=report,
                device=self.device,
            )

        job = self.jobs.submit(name, task)
        self._json({"job": job.snapshot()}, 202)

    def _infer(self) -> None:
        try:
            form = self._form()
            board_id = form.text("board_id")
            upload = form.file("file")
            if not board_id or upload is None:
                self._error("需要 board_id 与一个 CSV 或 JPEG 文件")
                return
            board = registry.get_board(board_id)
            result = predict_frame(board, upload[0], upload[1])
            self._json({"frame": encode_frame(result)})
        except KeyError:
            self._error("board not found", 404)
        except (ValueError, FileNotFoundError) as exc:
            self._error(str(exc))
        except Exception as exc:
            traceback.print_exc()
            self._error(f"{type(exc).__name__}: {exc}", 500)

    # -- static ----------------------------------------------------------
    def _static(self, path: str) -> None:
        if path in ("", "/", "/web"):
            self.send_response(302)
            self.send_header("Location", "/web/")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        if not path.startswith("/web/"):
            self._error("not found", 404)
            return
        target = (REPO_ROOT / path.lstrip("/")).resolve()
        try:
            target.relative_to(WEB_ROOT.resolve())
        except ValueError:
            self._error("forbidden", 403)
            return
        if not target.exists():
            self._error("not found", 404)
            return
        if target.is_dir():
            target = target / "index.html"
            if not target.exists():
                self._error("not found", 404)
                return
        if not target.is_file():
            self._error("not found", 404)
            return
        content_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        if content_type.startswith("text/") or target.suffix in {".js", ".css", ".json", ".svg"}:
            content_type += "; charset=utf-8"
        data = target.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        if target.suffix in {".html", ".js", ".css"}:
            self.send_header("Cache-Control", "no-store")
        else:
            self.send_header("Cache-Control", "public, max-age=3600")
        self.end_headers()
        self.wfile.write(data)


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------
def build_server(host: str = "127.0.0.1", port: int = 8000, device: str = "auto") -> ThreadingHTTPServer:
    jobs = JobManager()
    handler = type("BoundHandler", (Handler,), {"jobs": jobs, "device": device})
    server = ThreadingHTTPServer((host, int(port)), handler)
    server.daemon_threads = True
    return server


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "mps", "cuda"],
        help="training device; inference stays on the CPU",
    )
    parser.add_argument(
        "--no-presets",
        action="store_true",
        help="skip registering the two built-in boards from outputs/thermal",
    )
    args = parser.parse_args(argv)

    if not args.no_presets:
        for event in registry.ensure_presets():
            if event["status"].startswith("missing"):
                print(f"[preset] {event['board']}: missing checkpoint {event['path']}", file=sys.stderr)
            else:
                print(f"[preset] {event['board']}: {event['status']}")

    server = build_server(args.host, args.port, args.device)
    host, port = server.server_address[:2]
    print(f"Thermal Studio backend on http://{host}:{port}/web/")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
