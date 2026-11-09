"""Radiometric JPEG reader: container layout, calibration, upload dispatch.

The synthetic container below reproduces the bytes this camera writes after the
JPEG image (HDRI header, 16-bit matrix, HIPT anchors).  The campaign itself is
gitignored, so the real-file check is conditional; when the archive is present
it compares a decoded JPEG against the CSV export of the same frame.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest

from tdp.io import HMParseError, is_hm_jpeg, parse_hm_csv, parse_hm_jpeg, parse_thermal_upload
from tdp.io.hm_jpeg import decode_temperatures, response

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_CASE = ROOT / "实验数据（黑胶带版）" / "case10_树莓派4B裸板_满负载"


# --------------------------------------------------------------------------
# synthetic container
# --------------------------------------------------------------------------
def _synthetic_container(
    raw: np.ndarray,
    t_min_c: float,
    t_max_c: float,
    ambient_c: float = 25.0,
    *,
    with_hipt: bool = True,
) -> bytes:
    height, width = raw.shape
    head = bytearray(44)
    head[0:4] = b"HDRI"
    head[4] = 40  # header length, counts itself and the fields
    head[5] = 16  # bit depth
    head[7:11] = int(width).to_bytes(4, "little")
    head[11:15] = int(width * height).to_bytes(4, "little")
    body = bytes(head) + np.asarray(raw, dtype="<u2").tobytes()
    if with_hipt:
        hipt = bytearray(168)
        struct.pack_into("<f", hipt, 32, 0.93)
        struct.pack_into("<f", hipt, 36, ambient_c)
        struct.pack_into("<f", hipt, 120, t_max_c)
        struct.pack_into("<f", hipt, 124, t_min_c)
        body += b"HIPT" + (168).to_bytes(4, "little") + bytes(hipt)
    body += b"RADIOMETRICIMAGE"
    return b"\xff\xd8\xff\xd9" + body


# --------------------------------------------------------------------------
# container and calibration
# --------------------------------------------------------------------------
def test_synthetic_block_round_trips_the_anchored_field():
    raw = np.tile(np.linspace(5000, 6000, 64), (48, 1)).astype(np.uint16)
    u_min = (float(raw.min()) - 5200.0) / 100.0
    u_max = (float(raw.max()) - 5200.0) / 100.0
    t_min = float(response(u_min))
    t_max = float(response(u_max))
    data = _synthetic_container(raw, t_min, t_max)

    assert is_hm_jpeg(data)
    frame = parse_hm_jpeg(data, "HM20260101120000.jpeg", expect_shape=None)

    expected = response((raw.astype(np.float64) - 5200.0) / 100.0)
    assert frame.temps.shape == (48, 64)
    assert np.allclose(frame.temps, expected, atol=1e-3)
    assert frame.meta["source"] == "hik-jpeg"
    assert frame.meta["stat_min_c"] == pytest.approx(t_min, rel=0, abs=1e-4)
    assert frame.timestamp is not None and frame.timestamp.year == 2026


def test_plain_jpeg_is_rejected_with_an_explanation():
    plain = b"\xff\xd8" + b"not a radiometric image" * 3 + b"\xff\xd9"
    assert not is_hm_jpeg(plain)
    with pytest.raises(HMParseError, match="HDRI"):
        parse_hm_jpeg(plain, "photo.jpeg")


def test_missing_hipt_box_is_rejected():
    raw = np.full((16, 12), 5200, dtype=np.uint16)
    data = _synthetic_container(raw, 20.0, 30.0, with_hipt=False)
    with pytest.raises(HMParseError, match="HIPT"):
        parse_hm_jpeg(data, "HM20260101120000.jpeg")


def test_narrow_frame_keeps_the_max_anchor_and_does_not_explode():
    # A painted board at room temperature spans about 1 K, and a 1-count frame
    # would demand an absurd gain.  The clamp must keep the maximum exact.
    raw = np.full((16, 16), 5347, dtype=np.uint16)
    raw[3, 4] = 5348
    temps = decode_temperatures(raw, 21.0, 22.0)
    assert np.isfinite(temps).all()
    assert float(temps.max()) == pytest.approx(22.0, abs=1e-3)
    assert float(temps.min()) > 21.0  # clamp compresses the span, never inverts it


def test_upload_dispatch_keeps_csv_and_accepts_jpeg():
    csv_rows = ["坐标Y\\X,0,1,2", "0,30.0,31.0,32.0", "1,33.0,34.0,35.0"]
    csv_bytes = ("\n".join(csv_rows) + "\n").encode("utf-8")
    parsed_csv = parse_thermal_upload(csv_bytes, "upload.csv", expect_shape=None)
    assert parsed_csv.temps.shape == (2, 3)
    assert np.allclose(parsed_csv.temps, [[30.0, 31.0, 32.0], [33.0, 34.0, 35.0]])

    raw = np.tile(np.linspace(5000, 6000, 8), (6, 1)).astype(np.uint16)
    jpeg_bytes = _synthetic_container(raw, 25.0, 50.0)
    parsed_jpeg = parse_thermal_upload(jpeg_bytes, "HM20260101120000.jpeg", expect_shape=None)
    assert parsed_jpeg.temps.shape == (6, 8)
    assert parsed_jpeg.encoding == "hik-jpeg"


def test_web_ingest_accepts_a_jpeg_upload():
    from tdp.web.sessions import load_uploads

    raw = np.tile(np.linspace(5000, 6000, 32), (24, 1)).astype(np.uint16)
    data = _synthetic_container(raw, 25.0, 50.0)
    frames, failures = load_uploads([("HM20260101120000.jpeg", data)])
    assert not failures
    assert len(frames) == 1
    assert frames[0].temps.shape == (24, 32)
    assert frames[0].timestamp is not None


# --------------------------------------------------------------------------
# campaign check (skipped when the gitignored archive is absent)
# --------------------------------------------------------------------------
@pytest.mark.skipif(
    not (CAMPAIGN_CASE / "photo").exists() or not (CAMPAIGN_CASE / "data").exists(),
    reason="campaign archive not present",
)
def test_campaign_jpeg_matches_its_csv_export():
    jpeg = sorted((CAMPAIGN_CASE / "photo").glob("HM*.jpeg"))[120]
    csv_path = CAMPAIGN_CASE / "data" / (jpeg.stem + ".csv")
    decoded = parse_hm_jpeg(jpeg.read_bytes(), jpeg.name, expect_shape=None)
    exported = parse_hm_csv(csv_path.read_bytes(), csv_path.name, expect_shape=None)
    error = decoded.temps - exported.temps
    assert error.shape == (256, 192)
    assert float(np.abs(error).mean()) < 0.25
    assert float(np.abs(error).max()) < 1.0
