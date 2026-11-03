"""Adapt the existing 75-joint landmark archive to the Siformer 54-joint cache."""
import json
import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from vsl.adapter import HAND_ORDER, POSE_ORDER, SCHEMA
from vsl.common import cache_signature, load_config, read_json, sha256, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/landmark_train.json")
    parser.add_argument("--source", default="landmark", help="Archive root containing one recipe directory")
    args = parser.parse_args()
    cfg = load_config(args.config)
    candidates = list(Path(args.source).glob("*/index.json"))
    if len(candidates) != 1:
        raise ValueError(f"Expected one landmark recipe index under {args.source}; found {len(candidates)}")
    source = candidates[0]
    archive = read_json(source)
    root = source.parent
    out = Path(cfg["cache_dir"])
    manifest = Path(cfg["manifest_dir"])
    if out.exists() or manifest.exists():
        raise FileExistsError("Import output already exists; keep the existing run/cache intact")
    labels = read_json(cfg["label_mapping"])
    rows, videos = [], []
    seen = set()
    for video_id, record in archive["records"].items():
        path = root / record["file"]
        if sha256(path) != record["sha256"]:
            raise ValueError(f"Archive checksum mismatch: {path}")
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(str(z["metadata"]))
            features = z["features"].reshape(-1, 75, 4)
            detected = z["detection_mask"]
            valid = z["valid_mask"] & ~z["padding_mask"]
        rel = meta["source_metadata"]["relative_path"]
        if Path(rel).stem != video_id or rel in seen or not rel.startswith(("train/", "valid/", "test/")):
            raise ValueError(f"Duplicate or inconsistent archive video: {video_id}")
        seen.add(rel)
        split = {"valid": "validation", "train": "train", "test": "test"}[meta["split"]]
        label = labels[Path(rel).parent.name]
        xy = features[:, :, :2]
        coords = np.zeros((len(xy), 54, 2), np.float32)
        mask = np.zeros((len(xy), 54), bool)
        for start, offset in ((0, 33), (21, 54)):
            indices = np.asarray(HAND_ORDER) + offset
            coords[:, start:start + 21] = xy[:, indices]
            mask[:, start:start + 21] = detected[:, indices]
        for j, pose in enumerate(POSE_ORDER):
            if pose is not None:
                coords[:, 42 + j] = xy[:, pose]
                mask[:, 42 + j] = detected[:, pose] & (features[:, pose, 3] >= 0.5)
        mask[:, 43] = mask[:, 48] & mask[:, 49]
        coords[:, 43] = (coords[:, 48] + coords[:, 49]) / 2
        coords[:, :, 1] = 1 - coords[:, :, 1]
        mask &= valid[:, None] & np.isfinite(coords).all(axis=-1)
        coords[~mask] = 0
        if not mask.any():
            raise ValueError(f"No usable joints: {video_id}")
        name = record["file"]
        out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out / name, coords=coords, mask=mask, schema=SCHEMA)
        videos.append({"video_id": rel, "path": rel, "split": split, "label": label,
                       "label_name": Path(rel).parent.name, "sha256": meta["source_metadata"]["file_sha256"],
                       "signer_id": None})
        rows.append({**videos[-1], "parent_video": rel, "variant": 0, "schema": SCHEMA,
                     "extractor": cfg["extractor"], "cache_signature": cache_signature(cfg),
                     "seed": cfg["seed"], "augmentation": None, "cache_path": name,
                     "status": "ok", "frames": len(coords), "cache_sha256": sha256(out / name)})
    counts = Counter(r["split"] for r in videos)
    if counts != {"train": 3077, "validation": 385, "test": 404}:
        raise ValueError(f"Unexpected archive split counts: {counts}")
    for split in counts:
        if {r["label"] for r in videos if r["split"] == split} != set(range(100)):
            raise ValueError(f"{split}: missing labels")
    manifest.mkdir(parents=True)
    write_json(manifest / "videos.json", videos)
    write_json(manifest / "labels.json", labels)
    write_json(out / "index.json", rows)
    write_json(out / "errors.json", [])
    write_json(out / "provenance.json", {"complete": True, "schema": SCHEMA,
        "cache_signature": cache_signature(cfg), "manifest_sha256": sha256(manifest / "videos.json"),
        "labels_sha256": sha256(manifest / "labels.json"), "source_index_sha256": sha256(source),
        "source_recipe_sha256": archive["recipe_sha256"], "conversion": "75x4 to 54x2, 16 frames"})
    print(f"Imported {len(rows)} records into {out}", flush=True)


if __name__ == "__main__":
    main()
