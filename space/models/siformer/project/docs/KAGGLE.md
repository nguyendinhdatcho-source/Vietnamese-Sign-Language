# Train Siformer on Kaggle from the existing landmark archive

The local archive in `landmark/<recipe-hash>/` contains 3,866 videos (3,077 train, 385 validation, 404 test). Its 16 × 300 features differ from the original RGB extraction cache. The notebook imports them into the Siformer 54-point XY format, remaps class IDs using `configs/labels.json`, and trains with `configs/landmark_train.json` (16 frames, 100 epochs). It does not need the RGB videos or MediaPipe on Kaggle.

## Kaggle Inputs

Upload the two prepared ZIP files as Kaggle Inputs for [notebooks/kaggle.ipynb](../notebooks/kaggle.ipynb):

1. `dat301m_source_kaggle.zip`: source files from this project, including `train.py`, `vsl/`, `vsl_tf/`, `vendor/`, `scripts/`, `configs/`, `docs/dataset_counts.csv`, and `requirements.txt`. Include `configs/landmark_train.json` and `scripts/import_landmark_cache.py`. Exclude local `.venv*`, `data/`, `artifacts/`, `runs/`, and `backups/`.
2. `dat301m_landmark_kaggle.zip`: the complete `landmark/` directory with its recipe subdirectory, `index.json`, all `.npz` files, and receipt `.json` files. Keep the recipe directory name and files unchanged. Its raw NPZ checksums are verified during import.

Enable a Kaggle GPU accelerator. The notebook finds these ZIP files under `/kaggle/input` and extracts them into `/kaggle/working`. In the first cell, set `SOURCE_ROOT` and `LANDMARK_ROOT` only if automatic discovery finds multiple candidates. Use a new `SESSION_NAME`, leave `MODE='fresh'` and `RUN_TRAIN=True`, then run cells in order. The notebook copies source to `/kaggle/working`, imports the archive there, checks GPU and data integrity, runs one smoke batch, and starts full training. Checkpoints and `history.csv` are written under `/kaggle/working/<SESSION_NAME>/run/` after each epoch. Kaggle GPU execution has not been verified locally.

The notebook uses Kaggle's existing TensorFlow GPU runtime. Its dependency check requires TensorFlow `>=2.18,<2.22` and NumPy `<2` and leaves core GPU packages unchanged. MediaPipe is skipped for this cache-only path. If the runtime does not meet these bounds or TensorFlow cannot see the GPU, choose a compatible Kaggle image before training.

For another session, download the generated `<SESSION_NAME>_resume_bundle.zip`, extract it, publish the extracted folder as a Kaggle Dataset or Model, and add it as Input. Set `MODE='resume'`, `RESTORE_ROOT` to that folder, a new `SESSION_NAME`, and `RUN_RESUME=True`. Resume keeps the cache, mapping, and model settings; only `TOTAL_EPOCHS` may increase. `RUN_TEST` is off by default and should be enabled only after choosing the final checkpoint from validation.
