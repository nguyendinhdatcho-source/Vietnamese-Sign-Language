# Vendored upstream source

Repository: https://github.com/mpuu00001/Siformer
Commit: `979a14ed15ed0f20afd77d447ad23c4f4107a2c3`
Retrieved/read: 2026-10-20. License: MIT, see LICENSE.

The active tree retains only framework-independent normalization and
rectification helpers. The original model, attention, encoder and decoder
source is preserved in `backups/pre_tensorflow_only_20261020.zip` for
methodology audit; runtime inference uses the TensorFlow port in `vsl_tf/`.
Original training script was read remotely but is not called by this project.
No weights or dataset were downloaded.
Text encoding was written as UTF-8 by PowerShell; algorithm text is unchanged.
Three __init__.py packaging shims were added locally to avoid namespace collisions.
Landmark adaptation lives in `vsl/adapter.py`; the TensorFlow model lives in `vsl_tf/`.
The motion-table CSV dependencies for rectification are absent from the upstream
tree at this commit. This project does not fabricate them.
