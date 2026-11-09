"""Parser for HIKMICRO-style thermal CSV exports (HM<YYYYMMDDHHMMSS>.csv).

File layout (verified against 实验数据（黑胶带版）):
  - a metadata block (file path, units, 参数/统计 sections with label,value cells),
  - a header row  ``坐标Y\\X,0,1,...,191``,
  - 256 data rows ``<row_index>,v0,...,v191`` in °C.

Encoding varies *per file* within one session (some GBK/GB18030, some UTF-8-BOM),
so decoding tries a chain. Metadata stats (平均值/最小值/最大值) are used as a
per-file integrity self-check of the parsed matrix.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

_ENCODING_CHAIN = ("utf-8-sig", "gb18030")

# Metadata labels, tolerant to simplified/traditional variants.
_META_PATTERNS: dict[str, re.Pattern[str]] = {
    "emissivity": re.compile(r"[发發][射]率"),
    "reflected_temp_c": re.compile(r"反射[温溫]度"),
    "distance_m": re.compile(r"距[离離]"),
    "env_temp_c": re.compile(r"[环環]境[温溫]度"),
    "humidity": re.compile(r"[湿濕]度"),
    "stat_mean_c": re.compile(r"平均[值]?"),
    "stat_min_c": re.compile(r"最小[值]?"),
    "stat_max_c": re.compile(r"最大[值]?"),
}

_TIMESTAMP_RE = re.compile(r"HM(\d{14})")


class HMParseError(ValueError):
    """Raised when an HM CSV cannot be parsed or fails its integrity self-check."""


@dataclass
class HMFrame:
    temps: np.ndarray  # (H, W) float32 °C, [row, col], row 0 = top of frame
    meta: dict[str, float | str]
    timestamp: datetime | None
    path: Path
    encoding: str

    @property
    def shape(self) -> tuple[int, int]:
        return self.temps.shape  # type: ignore[return-value]


def _decode(raw: bytes, path: Path) -> tuple[str, str]:
    for enc in _ENCODING_CHAIN:
        try:
            return raw.decode(enc), enc
        except UnicodeDecodeError:
            continue
    # Metadata may be mojibake but the numeric matrix is ASCII — salvage it.
    return raw.decode("utf-8", errors="replace"), "utf-8/replace"


def _parse_metadata(lines: list[str]) -> dict[str, float | str]:
    meta: dict[str, float | str] = {}
    for line in lines:
        cells = [c.strip() for c in line.split(",")]
        for i, cell in enumerate(cells):
            if not cell:
                continue
            for key, pat in _META_PATTERNS.items():
                if key not in meta and pat.search(cell):
                    for value_cell in cells[i + 1 :]:
                        if value_cell:
                            try:
                                meta[key] = float(value_cell)
                            except ValueError:
                                meta[key] = value_cell
                            break
                    break
    return meta


def parse_hm_csv(
    data: bytes,
    name: str,
    expect_shape: tuple[int, int] | None = (256, 192),
    stat_tolerance_c: float = 0.2,
    self_check: bool = True,
) -> HMFrame:
    """Parse one HIKMICRO CSV already held in memory.

    The browser backend receives uploads as bytes and has no path yet, so the
    decoder and the integrity self-check must not depend on the filesystem.
    ``name`` is still needed: the clock lives in the ``HM<timestamp>`` filename.

    ``expect_shape=None`` reads the matrix shape from the coordinate header
    instead of asserting 256 x 192.  Uploads from another camera, or a frame
    exported at another resolution, then still run; the board window and the
    network grid are independent of the sensor grid.
    """
    path = Path(name)
    text, encoding = _decode(data, path)
    return parse_hm_text(
        text,
        path,
        encoding=encoding,
        expect_shape=expect_shape,
        stat_tolerance_c=stat_tolerance_c,
        self_check=self_check,
    )


def read_hm_csv(
    path: Path | str,
    expect_shape: tuple[int, int] | None = (256, 192),
    stat_tolerance_c: float = 0.2,
    self_check: bool = True,
) -> HMFrame:
    path = Path(path)
    return parse_hm_csv(
        path.read_bytes(),
        path.name,
        expect_shape=expect_shape,
        stat_tolerance_c=stat_tolerance_c,
        self_check=self_check,
    )


def parse_hm_text(
    text: str,
    path: Path,
    *,
    encoding: str = "utf-8",
    expect_shape: tuple[int, int] | None = (256, 192),
    stat_tolerance_c: float = 0.2,
    self_check: bool = True,
) -> HMFrame:
    """Parse decoded CSV text; shared by the path and bytes entry points."""
    lines = text.splitlines()

    n_cols = expect_shape[1] if expect_shape is not None else None
    header_idx: int | None = None
    for i, line in enumerate(lines):
        cells = line.split(",")
        if n_cols is None:
            if len(cells) >= 4 and [c.strip() for c in cells[1:4]] == ["0", "1", "2"]:
                while cells and not cells[-1].strip():
                    cells.pop()
                n_cols = len(cells) - 1
                header_idx = i
                break
        elif len(cells) >= n_cols and [c.strip() for c in cells[1:4]] == ["0", "1", "2"]:
            header_idx = i
            break
    if header_idx is None or n_cols is None:
        raise HMParseError(f"{path.name}: coordinate header row not found")

    rows: list[np.ndarray] = []
    row_labels: list[int] = []
    for line in lines[header_idx + 1 :]:
        cells = line.split(",")
        while cells and not cells[-1].strip():
            cells.pop()
        if len(cells) < n_cols:  # blank line / footer ends the matrix
            break
        try:
            row_labels.append(int(cells[0]))
            rows.append(np.asarray(cells[1 : n_cols + 1], dtype=np.float32))
        except ValueError as exc:
            raise HMParseError(f"{path.name}: bad data row after {len(rows)} rows: {exc}") from exc

    temps = np.stack(rows) if rows else np.empty((0, n_cols), np.float32)
    if expect_shape is not None:
        if temps.shape != expect_shape:
            raise HMParseError(f"{path.name}: matrix shape {temps.shape}, expected {expect_shape}")
        expected_rows = expect_shape[0]
    else:
        if temps.shape[0] < 2 or temps.shape[1] < 2:
            raise HMParseError(f"{path.name}: matrix shape {temps.shape} is not a usable image")
        expected_rows = int(temps.shape[0])
    if row_labels != list(range(expected_rows)):
        raise HMParseError(f"{path.name}: row indices not contiguous 0..{expected_rows - 1}")

    meta = _parse_metadata(lines[:header_idx])

    if self_check:
        checks = {
            "stat_mean_c": float(temps.mean()),
            "stat_min_c": float(temps.min()),
            "stat_max_c": float(temps.max()),
        }
        for key, parsed in checks.items():
            reported = meta.get(key)
            if isinstance(reported, float) and abs(parsed - reported) > stat_tolerance_c:
                raise HMParseError(
                    f"{path.name}: self-check failed on {key}: "
                    f"parsed {parsed:.2f} vs metadata {reported:.2f}"
                )

    m = _TIMESTAMP_RE.search(path.name)
    timestamp = datetime.strptime(m.group(1), "%Y%m%d%H%M%S") if m else None

    return HMFrame(temps=temps, meta=meta, timestamp=timestamp, path=path, encoding=encoding)


def read_case_dir(case_dir: Path | str, pattern: str = "data/HM*.csv", **kwargs) -> list[HMFrame]:
    """Read every HM CSV under a case directory, sorted by filename timestamp."""
    case_dir = Path(case_dir)
    files = sorted(case_dir.glob(pattern))
    if not files:
        files = sorted(case_dir.rglob("HM*.csv"))
    if not files:
        raise FileNotFoundError(f"no HM*.csv under {case_dir}")
    return [read_hm_csv(f, **kwargs) for f in files]
