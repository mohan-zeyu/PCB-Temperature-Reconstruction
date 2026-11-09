"""Launch the Thermal Studio backend.

    .venv/bin/python web/server.py --port 8000

The implementation lives in :mod:`tdp.web.server` so it can be imported by
tests and by other front ends.  This file only makes the command short.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tdp.web.server import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
