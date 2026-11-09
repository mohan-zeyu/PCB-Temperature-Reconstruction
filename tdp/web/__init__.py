"""Browser backend for the thermal reconstruction studio.

The static frontend can replay saved arrays, but it cannot turn an arbitrary
HIKMICRO CSV into a temperature field.  This package supplies the missing half:

* :mod:`tdp.web.sessions` -- board-agnostic ingestion.  Any 256 x 192 HIKMICRO
  CSV is parsed, the board is located in the raw frame, rectified onto one
  agreed grid, and described in kelvin above its own ambient.
* :mod:`tdp.web.training` -- pair a bare and a high-emissivity recording by
  thermal state, then train the residual correction net on the pair.
* :mod:`tdp.web.inference` -- apply a saved board operator to any new frame.
* :mod:`tdp.web.registry` -- the board library on disk.
* :mod:`tdp.web.server` -- a small standard-library HTTP server and JSON API.

Nothing here claims an absolute physical truth.  The high-emissivity recording
is the label, exactly as in the rest of the project, and the reported error is
a fit error against that label unless a genuinely held-out session is supplied.
"""
