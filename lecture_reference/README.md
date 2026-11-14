# lecture_reference

Lecture files kept for reading only; nothing in this project imports or runs them.
They work on the lecture's synthetic scene (not our camera), and they import `mvlib` / `synth` from their own
folder, so they do not run from here. The lecture code we do reuse lives in `machine_vision/`
(`mvlib.py`, `animate_relative_pose.py`).

* `learned_pose.py`, `synth.py`, `example_pipeline.py`, `occlusion_prediction.py`: lecture examples (T5, T3, E7 ideas)
* `check_timing.py`, `check_layout.py`, `extract_listing.py`, `make_figures.py`, `dictionary_table.py`,
  `rotation_repr_figure.py`, `make_calibration.py`: tools the lecturer used to build the slides
