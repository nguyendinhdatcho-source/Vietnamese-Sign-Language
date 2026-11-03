"""TensorFlow-facing NumPy loader over the unchanged VSL NPZ cache and adapter.

Sampling, normalizing, masks, splits and metadata are shared with preprocessing.
"""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from vsl.adapter import SCHEMA, normalize
from vsl.augmentation import temporal_sample
from vsl.common import cache_signature, read_json, sha256


class LandmarkDataset:
    def __init__(self, cfg, split, allow_errors=False):
        if split not in ("train", "validation", "test"):
            raise ValueError(split)
        self.cfg, self.split, self.root = cfg, split, Path(cfg["cache_dir"])
        provenance = read_json(self.root / "provenance.json")
        if not provenance["complete"] or provenance["cache_signature"] != cache_signature(cfg):
            raise ValueError("Cache incomplete or preprocessing config changed")
        for field, filename in (("manifest_sha256", "videos.json"), ("labels_sha256", "labels.json")):
            if provenance[field] != sha256(Path(cfg["manifest_dir"]) / filename):
                raise ValueError("Manifest/mapping differs from cache")
        if read_json(cfg["label_mapping"]) != read_json(Path(cfg["manifest_dir"]) / "labels.json"):
            raise ValueError("Configured label mapping differs from frozen manifest mapping")
        all_rows = read_json(self.root / "index.json")
        self.rows = [r for r in all_rows if r["split"] == split]
        expected = [r for r in read_json(Path(cfg["manifest_dir"]) / "videos.json") if r["split"] == split]
        if not self.rows or sorted(r["video_id"] for r in self.rows if r["variant"] == 0) != sorted(r["video_id"] for r in expected):
            raise ValueError(f"{split}: missing or duplicate original videos")
        keys = [(r["video_id"], r["variant"]) for r in self.rows]
        if len(keys) != len(set(keys)):
            raise ValueError(f"{split}: duplicated cache variants")
        self.errors = [r for r in self.rows if r["status"] != "ok"]
        if self.errors and not allow_errors:
            raise RuntimeError(f"{split}: {len(self.errors)} cache errors; inspect errors.json")
        for r in self.rows:
            if r["schema"] != SCHEMA or not 0 <= int(r["label"]) < 100:
                raise ValueError(f"Schema/label error in {r['video_id']}")
            if split != "train" and (r["variant"] != 0 or r["augmentation"] is not None):
                raise ValueError("Augmented record in validation/test")
        self.labels = np.asarray([int(r["label"]) for r in self.rows], np.int64)

    def __len__(self):
        return len(self.rows)

    def get(self, index, speed=1.0):
        row = self.rows[index]
        try:
            if row["status"] != "ok":
                raise RuntimeError(row["error"])
            path = self.root / row["cache_path"]
            if "cache_sha256" in row and sha256(path) != row["cache_sha256"]:
                raise ValueError("Landmark cache checksum changed")
            with np.load(path, allow_pickle=False) as z:
                if str(z["schema"]) != SCHEMA:
                    raise ValueError("Cache schema mismatch")
                coords, mask = z["coords"], z["mask"]
            if coords.ndim != 3 or coords.shape[1:] != (54, 2) or mask.shape != coords.shape[:2]:
                raise ValueError("Invalid landmark/mask shape")
            if not mask.any() or not np.isfinite(coords).all():
                raise ValueError("Missing entire clip or nonfinite cache")
            if self.split != "train" and speed != 1:
                raise ValueError("Random speed forbidden for validation/test")
            coords, mask, indices = temporal_sample(coords, mask, self.cfg["frames"], speed)
            coords = normalize(coords, mask)
            return {"coords": coords, "mask": mask, "indices": indices,
                    "label": int(row["label"]), "video_id": row["video_id"], "path": row["path"]}
        except Exception as exc:
            raise RuntimeError(f"{self.split} video={row['video_id']} variant={row['variant']}: {exc}") from exc

    def batches(self, rng, batch_size=None, workers=None):
        batch_size = batch_size or self.cfg["train"]["batch_size"]
        workers = self.cfg["train"]["workers"] if workers is None else workers
        if self.split == "train":
            if self.cfg["train"]["weighted_sampler"]:
                counts = Counter(self.labels.tolist())
                weights = np.asarray([1 / counts[int(y)] for y in self.labels], np.float64)
                indices = rng.choice(len(self), size=len(self), replace=True, p=weights / weights.sum())
            else:
                indices = rng.permutation(len(self))
            speeds = rng.uniform(*self.cfg["augmentation"]["speed"], size=len(indices))
        else:
            indices = np.arange(len(self))
            speeds = np.ones(len(indices))
        executor = ThreadPoolExecutor(max_workers=workers) if workers else None
        try:
            for start in range(0, len(indices), batch_size):
                work = [(int(i), float(s)) for i, s in zip(indices[start:start+batch_size], speeds[start:start+batch_size])]
                def fetch(task):
                    return self.get(*task)
                samples = list(executor.map(fetch, work)) if executor else [fetch(task) for task in work]
                yield {"coords": np.stack([r["coords"] for r in samples]),
                       "mask": np.stack([r["mask"] for r in samples]),
                       "label": np.asarray([r["label"] for r in samples], np.int64),
                       "video_id": [r["video_id"] for r in samples],
                       "indices": np.stack([r["indices"] for r in samples])}
        finally:
            if executor:
                executor.shutdown(wait=True)


def sampling_rows(dataset):
    counts = Counter(dataset.labels.tolist())
    records = [{"video_id": r["video_id"], "variant": r["variant"], "label": int(y),
                "class_count": counts[int(y)], "weight": 1.0 / counts[int(y)]}
               for r, y in zip(dataset.rows, dataset.labels)]
    return records, [{"label": i, "count": counts[i]} for i in range(100)]
