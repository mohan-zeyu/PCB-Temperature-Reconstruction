"""Decoder for the radiometric HIKMICRO JPEGs in ``photo/HM*.jpeg``.

Why this module exists
----------------------
The campaign's ``photo/*.jpeg`` files are not only colour renderings.  After
the JPEG image each file carries a proprietary ``HDRI`` block with the camera's
full 16-bit measurement grid, and a ``HIPT`` box with the frame's own minimum
and maximum temperature.  The CSV exports the desktop pipeline uses were made
from exactly this block: the CSV metadata records the JPEG as 源热像文件.
A JPEG-only delivery is therefore still a full temperature recording, not a
picture of one.

Format (little endian, verified on the 2026-07/08 H21ProS+(AI) campaign)
-----------------------------------------------------------------------
``<jpeg>`` ``FF D9`` then::

    "HDRI"                     4 bytes
    header length              1 byte, counts itself + the fields (40)
    bit depth                  1 byte at +5 (16)
    width                      uint32 at +7
    sample count               uint32 at +11  (width * height)
    ...                        44-byte header in total
    samples                    count * uint16, row major, top row first

The raw samples are not degrees Celsius.  They are the camera's calibrated
counts after its internal ambient compensation: a fixed response shape plus a
per-frame gain and offset.  The ``HIPT`` box (``HIPT`` + uint32 length +
payload) supplies the two anchors that remove the per-frame ambiguity:

    payload + 32   a camera constant (0.93 in every file of the campaign)
    payload + 36   the frame's ambient/level value
    payload +120   frame maximum temperature, degrees Celsius
    payload +124   frame minimum temperature, degrees Celsius

``RADIOMETRICIMAGE`` closes the file.

Calibration
-----------
The response shape was fitted on 142 frames from the 13 campaign sessions that
also have CSV exports.  It is ``2.7*u + c2*u^2 + c3*u^3`` with
``u = (raw - 5200) / 100``.  Anchoring each frame at its own min/max pair
reproduces the CSV export with these errors (the CSV itself is rounded to
0.1 degC):

    case10, all 480 frames   mean 0.11 K   worst frame 0.54 K RMSE
    case17, all 293 frames   mean 0.09 K   worst frame 0.26 K RMSE

The two anchors make the decode independent of the session-level shift that
the camera applies; the ambient value is kept as metadata but is not required.
A narrow frame (a painted board at room temperature spans about 1 K) would
make the two-anchor gain unstable, so the gain is clamped to 0.9-1.1.  The
absolute level then still comes from the maximum anchor.
"""

from __future__ import annotations

import re
import struct
from datetime import datetime
from pathlib import Path

import numpy as np

from tdp.io.hm_csv import HMFrame, HMParseError

_KIND = "hik-jpeg"
_MAGIC = b"HDRI"
_EOI_MAGIC = b"\xff\xd9" + _MAGIC
_HEADER_BYTES = 44
_HIPT_TAG = b"HIPT"
_HIPT_MIN_BYTES = 128

# Response shape, fitted on the campaign (see module docstring).
_LINEAR = 2.7
_QUADRATIC = -0.02897985
_CUBIC = 0.00027534
_RAW_REFERENCE = 5200.0
_RAW_SCALE = 100.0
_GAIN_MIN = 0.9
_GAIN_MAX = 1.1

_TIMESTAMP_RE = re.compile(r"HM(\d{14})")


def is_hm_jpeg(data: bytes) -> bool:
    """True when ``data`` is a JPEG that carries the embedded HDRI block."""
    return data.startswith(b"\xff\xd8") and data.find(_EOI_MAGIC) >= 0


def response(u: np.ndarray | float) -> np.ndarray | float:
    """Camera response shape in raw-count units, before the per-frame gain."""
    return _LINEAR * u + _QUADRATIC * u * u + _CUBIC * u * u * u


def decode_temperatures(
    raw: np.ndarray, t_min_c: float, t_max_c: float
) -> np.ndarray:
    """Map raw counts to degrees Celsius with the frame's own min/max anchors.

    The anchors are the temperatures the camera itself reports for the coldest
    and hottest pixel, so they already contain the session-level shift.  Only
    the shape between them is shared between sessions.
    """
    if not np.isfinite(t_min_c) or not np.isfinite(t_max_c):
        raise HMParseError("JPEG metadata has no usable min/max temperature")
    if t_max_c <= t_min_c:
        raise HMParseError(
            f"JPEG metadata min/max out of order: {t_min_c:.2f} >= {t_max_c:.2f}"
        )
    u = (np.asarray(raw, dtype=np.float64) - _RAW_REFERENCE) / _RAW_SCALE
    u_min = float(u.min())
    u_max = float(u.max())
    r_min = float(response(u_min))
    r_max = float(response(u_max))
    span = r_max - r_min
    if span <= 1e-9:
        gain = 1.0
    else:
        gain = float((t_max_c - t_min_c) / span)
        gain = min(_GAIN_MAX, max(_GAIN_MIN, gain))
    offset = t_max_c - gain * r_max
    return (offset + gain * response(u)).astype(np.float32)


def _parse_hipt(payload: bytes, data_start: int, n_samples: int) -> tuple[float, float, float]:
    """Return ``(ambient_c, t_min_c, t_max_c)`` from the HIPT box."""
    tail = payload[data_start + 2 * n_samples :]
    index = tail.find(_HIPT_TAG)
    if index < 0:
        raise HMParseError("JPEG has no HIPT metadata box")
    length = int.from_bytes(tail[index + 4 : index + 8], "little")
    body = tail[index + 8 : index + 8 + length]
    if len(body) < _HIPT_MIN_BYTES:
        raise HMParseError(f"JPEG HIPT box is truncated: {len(body)} bytes")
    # +32 is a unitless camera constant, so read each wanted value by its own
    # offset rather than assuming the three floats are contiguous.
    ambient = struct.unpack_from("<f", body, 36)[0]
    t_max_c = struct.unpack_from("<f", body, 120)[0]
    t_min_c = struct.unpack_from("<f", body, 124)[0]
    for value, label in ((ambient, "ambient"), (t_min_c, "minimum"), (t_max_c, "maximum")):
        if not np.isfinite(value) or not -100.0 < value < 400.0:
            raise HMParseError(f"JPEG {label} temperature is implausible: {value}")
    return ambient, t_min_c, t_max_c


def parse_hm_jpeg(
    data: bytes,
    name: str,
    expect_shape: tuple[int, int] | None = None,
) -> HMFrame:
    """Parse one radiometric HIKMICRO JPEG already held in memory."""
    path = Path(name)
    marker = data.find(_EOI_MAGIC)
    if marker < 0:
        if data.startswith(b"\xff\xd8"):
            raise HMParseError(
                f"{path.name}: JPEG has no HDRI block. It is a colour rendering, "
                "not a radiometric frame."
            )
        raise HMParseError(f"{path.name}: not a JPEG")
    payload = data[marker + 2 :]
    if len(payload) < _HEADER_BYTES:
        raise HMParseError(f"{path.name}: radiometric block is too short")
    header_len = payload[4]
    data_start = 4 + header_len
    if data_start < 16 or data_start > len(payload):
        raise HMParseError(f"{path.name}: bad HDRI header length {header_len}")
    width = int.from_bytes(payload[7:11], "little")
    n_samples = int.from_bytes(payload[11:15], "little")
    if width < 2 or n_samples < 4:
        raise HMParseError(f"{path.name}: bad HDRI dimensions {width} x {n_samples}")
    if n_samples % width:
        raise HMParseError(f"{path.name}: sample count {n_samples} is not a multiple of {width}")
    height = n_samples // width
    end = data_start + 2 * n_samples
    if end > len(payload):
        raise HMParseError(
            f"{path.name}: radiometric matrix truncated ({len(payload) - data_start} of "
            f"{2 * n_samples} bytes)"
        )
    raw = np.frombuffer(payload[data_start:end], "<u2").astype(np.float64).reshape(height, width)

    ambient, t_min_c, t_max_c = _parse_hipt(payload, data_start, n_samples)
    temps = decode_temperatures(raw, t_min_c, t_max_c)

    if expect_shape is not None and temps.shape != expect_shape:
        raise HMParseError(f"{path.name}: matrix shape {temps.shape}, expected {expect_shape}")
    if temps.shape[0] < 2 or temps.shape[1] < 2:
        raise HMParseError(f"{path.name}: matrix shape {temps.shape} is not a usable image")

    meta: dict[str, float | str] = {
        "source": _KIND,
        "stat_min_c": float(t_min_c),
        "stat_max_c": float(t_max_c),
        "ambient_c": float(ambient),
        "raw_min": float(raw.min()),
        "raw_max": float(raw.max()),
    }
    match = _TIMESTAMP_RE.search(path.name)
    timestamp = datetime.strptime(match.group(1), "%Y%m%d%H%M%S") if match else None
    return HMFrame(
        temps=temps,
        meta=meta,
        timestamp=timestamp,
        path=path,
        encoding=_KIND,
    )
