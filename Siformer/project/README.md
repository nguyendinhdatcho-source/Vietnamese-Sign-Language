# Siformer VSL100 — TensorFlow

One RGB video contains one Vietnamese Sign Language word. The pipeline keeps the existing 100-class `train`/`valid`/`test` split and fixed [label mapping](configs/labels.json): RGB video → MediaPipe Holistic landmarks → normalization and temporal sampling → TensorFlow Siformer → training/validation → validation-only inference calibration → one final test evaluation.

This is a **MediaPipe 2D adaptation** of Siformer, not a claim of exact paper reproduction. The [method audit](docs/METHOD.md) and [migration report](docs/TENSORFLOW_MIGRATION_REPORT.md) describe the feature-isolated encoders, ProbSparse attention, Transformer decoder, 100-class head, numerical comparison and remaining differences. The TensorFlow implementation lives in `vsl_tf/`; `vsl/` retains only shared adapter, augmentation, metrics and config utilities. No active path requires PyTorch.

## Data and preprocessing

`data/train`, `data/valid`, `data/test` contain MP4 files under the original 100 class-name folders. `configs/labels.json` fixes class IDs 0–99. `prepare_data.py inventory` writes the manifest and verifies split counts/integrity; `prepare_data.py cache` decodes RGB, performs train-only clip-consistent crop/color jitter before MediaPipe extraction for configured offline variants, and saves 54-point XY landmarks and masks. Train-only temporal speed is applied during sampling. Validation/test use deterministic sampling. The weighted train sampler uses inverse class frequency; validation/test have no augmentation or weighted sampling. Keep the original split and label mapping.

Default paths and all hyperparameters are in [configs/train.json](configs/train.json): `data_root`, `manifest_dir`, `cache_dir`, `label_mapping`, `run_dir`, `device`, `offline_variants`, `train.batch_size`, `train.workers`, `train.amp`, and epochs. Local defaults target CPU; set `device` to `cuda` only in a TensorFlow GPU environment. No pretrained weights are bundled. Rectification is disabled unless verified tables are supplied. Existing PyTorch checkpoints cannot resume this TensorFlow model. Pre-consolidation TensorFlow checkpoints carry old source hashes and require their matching archived source tree for exact resume; no such completed local run was found.

## Install and run locally

Use Python 3.11 and a separate environment. [requirements.txt](requirements.txt) pins MediaPipe and bounds TensorFlow and preprocessing packages. Native Windows TensorFlow can run on CPU; use a compatible Linux GPU environment for GPU training. Commands run from the project root:

```bash
python -m pip install -r requirements.txt
python -m unittest tests.test_pipeline -v
python prepare_data.py inventory --config configs/train.json
python prepare_data.py cache --config configs/train.json
python visualize_augmentation.py --config configs/train.json --output artifacts/augmentation_preview_v1
python scripts/model_summary.py --config configs/train.json
python train.py --config configs/train.json --smoke --run-dir runs/smoke_v1
python train.py --config configs/train.json
```

The smoke run is diagnostic only. The full run writes TensorFlow checkpoint files and epoch metadata under `run_dir`, with `best.json` selected by validation Top-1 accuracy (validation loss breaks ties) and `last.json` for exact epoch-boundary resume. Config/label/source/cache identities are checked on load. To resume an interrupted full run in a **new** directory, keep method settings and data unchanged; only `train.epochs` may increase:

```bash
python train.py --config configs/train.json --resume runs/siformer_vsl100_v1/last.json --run-dir runs/siformer_vsl100_resume_v1
```

Choose the `best.json` in the final selected branch. Lock static inference, or specify `--adaptive` to select encoder patience using validation only, **before viewing test results**:

```bash
python calibrate.py --config configs/train.json --checkpoint runs/siformer_vsl100_v1/best.json --output runs/siformer_vsl100_v1/policy.json
python evaluate.py --policy runs/siformer_vsl100_v1/policy.json --output runs/siformer_vsl100_v1/test_v1
```

`run_dir` contains config, fixed labels, class counts, sampling weights, history, and best/last checkpoint pointers plus complete TensorFlow state. Evaluation writes `metrics.json`, 100-class classification report, confusion matrices, top confusions, `predictions.csv`, decode/landmark errors, model-only latency and model information. Failed test videos remain in the denominator. Real VSL accuracy is not yet available; local synthetic integration results are not dataset results.

## Kaggle GPU

Use [notebooks/kaggle.ipynb](notebooks/kaggle.ipynb) and [docs/KAGGLE.md](docs/KAGGLE.md) to train from the existing `landmark/` archive on Kaggle GPU. Add separate Kaggle Inputs for the source tree and complete landmark archive. The notebook verifies and converts the 16-frame archive to the Siformer 54-point cache, remaps labels by class name, and writes all cache, checkpoints and results under `/kaggle/working`. It supports smoke, full training, validation-selected checkpoints, optional final test and resume bundles. Kaggle GPU execution remains unverified locally.

The original source before consolidation is recoverable from `backups/pre_tensorflow_only_20261020.zip`, with per-file hashes verified in its embedded manifest. User RGB data, caches, runs and snapshots were left in place.
