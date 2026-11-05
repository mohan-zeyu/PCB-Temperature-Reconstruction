# Thermal Studio frontend

Start from the repository root:

```bash
python3 -m http.server 8000
```

Open `http://localhost:8000/web/`.

The built-in Raspberry Pi 4B board uses saved case10 predictions. Five corresponding source HIKMICRO CSV frames are bundled in `samples/raspberry-pi-4b/`. The add-board flow replays the Orange Pi 5 Pro experiment, including 480 paired frames and the saved 300-epoch training history. Five matching Orange Pi source CSVs are bundled in `samples/orange-pi-5-pro/`.

The frontend reads and validates a selected 256 × 192 camera CSV. Only bundled frames whose file contents match a saved source frame can display a reconstruction; other valid CSVs can show the camera input, but have no model result in this frontend. The temperature maps and downloads are absolute °C using each frame's measured ambient temperature. Image and CSV corner gaps are filled from nearby valid board pixels. The displayed and exported maps are in portrait orientation.

After training playback, the named board is saved in browser local storage and appears in the picker and library. Use **板型库 → 恢复初始状态** to clear added boards, uploads, and training progress, then return to the starting Raspberry Pi 4B page. Training is presentation playback, not a new model fit; the saved Orange Pi model supplies its metrics and inference examples. The interface is written to accommodate a live backend later.

The JSON assets and bundled source CSVs are generated from the repository's saved `.npz` predictions, source recordings, and training metric files. To rebuild them:

```bash
.venv/bin/python web/generate_demo_assets.py
```
