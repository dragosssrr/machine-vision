# machine-vision: Assignment 2, marker pose tracking (classical vs. learned)

Every number in the report can be reproduced with the commands below, run from the repository root.
Seeds are fixed (`config.SEED = 0`; renderer seeds 0 / 2 / 1 for train / validation / test).

## Set-up
```
poetry install            # or: pip install opencv-python numpy matplotlib torch torchvision
```
Camera: calibrated once at 1280x720 (`calibration_output.npz` + `.json`). Every script opens the camera at that size
and stops if the driver delivers another resolution. Markers: `DICT_4X4_50`, F = 10, M = 22, L = 100 mm (`config.py`).

Data and trained models are **not** in git (too big). Download `data/` from our shared Google Drive folder
_(team: add the link)_ and put it at the repository root.

## Layout
Scripts sit flat at the top level on purpose: they all start with `from config import ...`, and Python finds
`config.py` only when the script is run from the folder that contains it. Run every command from the repository root.

```
machine-vision/
├── README.md                    how to reproduce every number
├── pyproject.toml, poetry.lock  libraries
├── config.py                    marker IDs, L, folders: the ONE source of constants
├── calibration_output.npz/.json camera calibration (made once by calibrate_camera.py)
│
│  ── T1–T4: classical pipeline ──────────────────────────
├── generate_markers.py          T1  printable markers -> markers/
├── calibrate_camera.py          T3  intrinsic calibration (step 1)
├── detect_live.py               T2  live IDs, fps, largest distance / tilt that still detects
├── track_live.py                T3+T4 live pose of M in F with filtered 3D / x,y,z / roll,pitch,yaw plot
│
│  ── T5: learned pipeline ───────────────────────────────
├── synth_render.py              renders the synthetic images
├── pose_net.py                  the network
├── pose_data.py                 feeds images to training
├── train_pose.py                training + λ sweep + fine-tuning
├── label_real.py                labels real training recordings with the classical pose
│
│  ── shared by T5 and T6 ────────────────────────────────
├── record.py                    records one recording folder
├── pipeline.py                  classical + learned: frame -> ^F T_M
├── run_pipeline.py              side-by-side video, T4 GIF
├── metrics.py                   error formulas
│
│  ── T6 ─────────────────────────────────────────────────
├── evaluate.py                  E1–E6, comparison table
│
├── machine_vision/              lecture starter code that we REUSE: mvlib.py, animate_relative_pose.py
├── lecture_reference/           lecture files we read but don't run
├── markers/                     T1 output: marker PNGs, markers_A4.pdf, markers.json (measured side)
├── data/                        NOT in git: inputs we recorded or generated
│   ├── calibration_images/      chessboard views of the calibration
│   ├── synth_train/ synth_val/ synth_test/
│   ├── recordings/<name>/       one folder per recording (train_01, e1_p1_y0, e2_d06_t30, ...)
│   └── models/                  posenet_synth.pt, posenet_real.pt (+ .json logs)
├── results/                     made by the code, never edited by hand (tables.md, results.json, fig_*.png,
│                                worst/, videos/, t2_*.json/csv)
├── report/                      report (≤ 12 pages) + figures/ copied from results/
├── video/                       final 2–3 min video + raw clips
├── docs/                        T5_T6_GUIDE.md
└── tests/                       make_fake_recordings.py (software check only)
```

* **data/ vs results/**: `data/` holds what we recorded or generated as *input*; `results/` holds what the code
  *computes* from it. `results/` can always be deleted and rebuilt by rerunning `evaluate.py` / `run_pipeline.py`.
* **Recording names** follow `experiment_what_number`, e.g. `e2_d06_t30` = E2, distance 0.6 m, tilt 30°.
  The real information (role, measured truth) is in each folder's `meta.json`.

| Track | Owner | Files |
|---|---|---|
| A classical | _(name)_ | `generate_markers.py`, `calibrate_camera.py`, `detect_live.py`, `track_live.py`, `markers/` |
| B network | _(name)_ | `synth_render.py`, `pose_net.py`, `pose_data.py`, `train_pose.py`, `label_real.py`, `data/synth_*`, `data/models/` |
| C evaluation | _(name)_ | `record.py`, `metrics.py`, `evaluate.py`, `run_pipeline.py`, `data/recordings/`, `results/`, `video/` |

## T1–T4 (classical pipeline)
```
# T1: markers. Print markers/markers_A4.pdf at 100 % ("actual size"), measure the black squares, then:
python generate_markers.py
python generate_markers.py --measured_mm <F> <M>     # writes markers/markers.json; update config.L if it differs
# T3 step 1: calibration (c = capture a chessboard view, q = calibrate); saves views to data/calibration_images/
python calibrate_camera.py
# T2: live detection: IDs + fps; move/tilt a marker until it drops out (s = log row, q = quit)
python detect_live.py --label "office light"          # -> results/t2_detection_log.csv, t2_detection_limits.json
# T3 + T4: live relative pose with filtered plots         -> results/videos/t4_sequence_export.mp4
python track_live.py
# T4 animation from a recording (3D frames + x,y,z + roll,pitch,yaw)
python run_pipeline.py --rec data/recordings/<rec> --method classical --gif results/<rec>
```

## T5: learned pose of M
```
# 1. synthetic data (exact labels, rendered with our K)
python synth_render.py --out data/synth_train --n 10000 --seed 0
python synth_render.py --out data/synth_val   --n 1000  --seed 2
python synth_render.py --out data/synth_test  --n 1000  --seed 1 --check     # E5 set + convention check
# 2. choose the loss weight on the synthetic validation set
python train_pose.py --synth data/synth_train --synth_val data/synth_val --epochs 3 --lam_sweep 0.5 1 2 5 10
# 3. train on synthetic data
python train_pose.py --synth data/synth_train --synth_val data/synth_val --epochs 20 --lam <LAM> --out data/models/posenet_synth.pt
# 4. real training recordings (role = train), labelled by the classical pipeline, then fine-tune
python record.py --name train_01 --role train --experiment train --seconds 60      # ... train_02 ... train_05
python label_real.py data/recordings/train_0*
python train_pose.py --init data/models/posenet_synth.pt --synth data/synth_train \
    --real_train data/recordings/train_01 data/recordings/train_02 data/recordings/train_03 data/recordings/train_04 \
    --real_val data/recordings/train_05 --epochs 5 --lr 1e-4 --lam <LAM> --out data/models/posenet_real.pt
# 5. the network inside the T3/T4 pipeline, on a test recording never used in training
python run_pipeline.py --rec data/recordings/<test_rec> --method both --model data/models/posenet_real.pt \
    --video results/videos/<test_rec>_side_by_side.mp4 --gif results/<test_rec>
```

## T6: comparison
```
# test recordings (role = test) with hand-measured truth, see docs/T5_T6_GUIDE.md section 4.2 for the shot list
python record.py --name e1_p1_y0   --role test --experiment E1 --frames 100 --t_mm 150 0 0 --rpy_deg 0 0 0
python record.py --name e2_d06_t30 --role test --experiment E2 --frames 100 --t_mm 150 0 0 --rpy_deg 30 0 0 \
                 --distance_m 0.6 --tilt_deg 30
python record.py --name e3_line_1  --role test --experiment E3 --seconds 20 --line_mm 0 120 0 1 0 0 --rpy_deg 0 0 0
python record.py --name e4_dim_1   --role test --experiment E4 --condition dim --seconds 20 --t_mm 100 50 0 --rpy_deg 0 0 0
# ...
python evaluate.py --model data/models/posenet_real.pt --synth_test data/synth_test
python evaluate.py --methods classical          # before the network exists
```
Output in `results/`: `tables.md` (comparison table + E1–E6 tables), `results.json`, `fig_errors_cdf.png`,
`fig_e2_tilt.png`, `fig_e3_paths.png`, `fig_e5_hist.png`, `worst/` (largest-error frames with axes drawn),
`cache/` (per-frame poses, reused until the model, calibration or L changes; `--no_cache` to force a rerun).

| Exp. | What | Grouped by |
|---|---|---|
| E1 | static poses: bias (accuracy) + jitter (precision) | recording |
| E2 | distance × tilt: output rate + error | `distance_m`, `tilt_deg` in meta.json |
| E3 | moving M: distance from the true line / circle | recording |
| E4 | conditions: output rate + error | `condition` in meta.json |
| E5 | synthetic test set: ^C T_M vs exact labels | method |
| E6 | run time per frame (ms, fps) | method |

## Data
* `data/recordings/<name>/`: `frames/*.jpg` (raw camera frames), `timestamps.csv`, `meta.json`
  (role train/test, experiment, condition, measured truth in the F frame: mm and ZYX roll-pitch-yaw in degrees).
* Split by recording: `train_pose.py` and `label_real.py` refuse every recording whose role is `test`;
  `evaluate.py` uses only recordings whose role is `test`.
* `data/synth_*`: rendered images + `labels.npz` (R, t of M in the camera frame, metres; K; image size).

## Software test (not results)
`tests/make_fake_recordings.py` renders recordings in the real format with exact truth, to check the evaluation
code end to end. Its numbers are never reported, and they go to their own folders, never `data/recordings/`:
```
python tests/make_fake_recordings.py --make_calib
MV_CALIB=tests/fake_calib.npz python tests/make_fake_recordings.py --out data/fake_recordings
MV_CALIB=tests/fake_calib.npz python evaluate.py --recordings data/fake_recordings --methods classical --out results/fake
```
Expected: classical E1 errors of about 1 mm / 0.1°.
