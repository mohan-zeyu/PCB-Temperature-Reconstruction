"""Learning to undo low-emissivity surface artifacts on PCB infrared images.

Experimental package (branch: experiment/unet-painted-truth-reconstruction).

What it is for
--------------
A bare PCB's infrared field reads far too cold on low-emissivity metal (SoC
lids, connector shields): the same physical board painted matte black reads the
truth at the same load.  These modules build per-pixel supervised pairs

    input  = bare-board infrared temperature above ambient (theta), and
    target = painted-board temperature above ambient for the same physical
             point at the same thermal state,

then train a small residual U-Net to map input to target.  The output of the
trained operator is a *temperature field*, not an emissivity, conductivity, or
power: no absolute scale is ever claimed, in line with the repository units
rule.

Honesty rules carried into every module here
--------------------------------------------
* Pairs are only physically meaningful where the two sessions describe the same
  board state; trusted (non-footprint) pixels must agree in theta.  The label
  noise of the pairing itself is measured and reported, never assumed.
* The painted session is a calibration reference, not an independent test.
  Generalization beyond one board layout can only be probed with another
  bare/painted pair (case09 <-> case16) or a future third session.
* Frames are never split randomly.  This experiment trains on the whole archive
  (owner decision), so fit numbers are not independent generalization; the
  cross-load probe supplies that.
* Reported scores are masked-region and per-block numbers first; a good
  board-average must never hide a wrong lid.

This __init__ intentionally imports nothing, so the package loads with only
numpy + torch installed (training runs on the remote GPU box).
"""
