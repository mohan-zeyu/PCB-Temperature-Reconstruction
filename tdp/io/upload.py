"""Choose the right reader for an uploaded thermal frame.

The browser uploads one temperature recording per file.  The campaign can
deliver it as a HIKMICRO CSV export or as the original radiometric JPEG, and
the two carry the same 256 x 192 measurement.  Detection is by content, not by
filename: a CSV that happens to be named ``.jpeg`` still parses as CSV, and an
original JPEG renamed ``.jpg`` still parses as a radiometric JPEG.

A plain JPEG (a screenshot, or an image saved by another tool) has no HDRI
block and raises :class:`tdp.io.hm_csv.HMParseError` with that explanation.
"""

from __future__ import annotations

from tdp.io.hm_csv import HMFrame, parse_hm_csv
from tdp.io.hm_jpeg import is_hm_jpeg, parse_hm_jpeg


def parse_thermal_upload(
    data: bytes,
    name: str,
    *,
    expect_shape: tuple[int, int] | None = None,
    stat_tolerance_c: float = 0.2,
    self_check: bool = True,
) -> HMFrame:
    """Parse one uploaded CSV or radiometric JPEG into the shared frame type.

    The CSV-only options are ignored for a JPEG: that reader does not need a
    statistical self-check because it anchors on the camera's own max/min pair.
    """
    if is_hm_jpeg(data):
        return parse_hm_jpeg(data, name, expect_shape=expect_shape)
    return parse_hm_csv(
        data,
        name,
        expect_shape=expect_shape,
        stat_tolerance_c=stat_tolerance_c,
        self_check=self_check,
    )
