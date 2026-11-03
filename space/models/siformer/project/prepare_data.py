"""Inventory and cache entry points. Does nothing until main() is called."""
import argparse
import csv
import importlib.metadata
import time
from collections import Counter, defaultdict
from pathlib import Path

from vsl.common import load_config, write_json, read_json, write_csv, sha256, digest, cache_signature


def inventory(cfg):
    root, out = Path(cfg["data_root"]).resolve(), Path(cfg["manifest_dir"])
    if out.exists():
        raise FileExistsError(f"Refuse overwrite: {out}. Set a new manifest_dir.")
    labels = sorted(p.name for p in (root / "train").iterdir() if p.is_dir())
    if len(labels) != 100:
        raise ValueError(f"Expected 100 class directories, found {len(labels)}")
    mapping = read_json(cfg["label_mapping"])
    if set(mapping) != set(labels) or sorted(mapping.values()) != list(range(100)):
        raise ValueError("Configured fixed label mapping must match exactly the 100 train folders")
    rows, hashes = [], defaultdict(list)
    for split, folder in (("train", "train"), ("validation", "valid"), ("test", "test")):
        unknown = {p.name for p in (root / folder).iterdir() if p.is_dir()} - set(labels)
        if unknown:
            raise ValueError(f"Unknown {split} classes: {unknown}")
        for p in sorted((root / folder).rglob("*.mp4")):
            relative = p.relative_to(root).as_posix()
            row = {"video_id": relative, "path": relative, "split": split,
                   "label": mapping[p.parent.name], "label_name": p.parent.name,
                   "sha256": sha256(p), "signer_id": None}
            rows.append(row)
            hashes[row["sha256"]].append(relative)
    cross = []
    for h, group in hashes.items():
        if len({p.split("/")[0] for p in group}) > 1:
            cross.append({"sha256": h, "paths": group})
    out.mkdir(parents=True)
    write_json(out / "labels.json", mapping)
    write_json(out / "videos.json", rows)
    write_json(out / "audit.json", {
        "counts": dict(Counter(r["split"] for r in rows)),
        "exact_duplicate_groups": [v for v in hashes.values() if len(v) > 1],
        "cross_split_sha256": cross,
        "signer_overlap": "unknown: no signer metadata; filename is not a signer ID",
        "near_duplicate_check": "not performed; SHA256 only detects byte-identical files"})
    write_csv(out / "class_counts.csv", [
        {"label": label, "name": name, **{
            s: sum(r["split"] == s and r["label"] == label for r in rows)
            for s in ("train", "validation", "test")}}
        for name, label in mapping.items()])
    print(f"Manifest written without changing splits; cross-split hash groups: {len(cross)}")


def decode(path):
    import cv2
    import numpy as np
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    expected = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frames = []
    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()
    if not frames or (expected > 0 and len(frames) != expected):
        raise RuntimeError(f"Incomplete decode {path}: decoded={len(frames)}, header={expected}")
    return np.stack(frames).astype(np.float32) / 255, fps


def extract(clip, cfg):
    import mediapipe as mp
    import numpy as np
    from vsl.adapter import convert_result
    coordinates, masks = [], []
    # New tracker for each video/variant: no state leaked across clips or splits.
    with mp.solutions.holistic.Holistic(
        static_image_mode=False, model_complexity=cfg["model_complexity"],
        smooth_landmarks=True, enable_segmentation=False, refine_face_landmarks=False,
        min_detection_confidence=cfg["min_detection_confidence"],
        min_tracking_confidence=cfg["min_tracking_confidence"]
    ) as extractor:
        for frame in clip:
            result = extractor.process(np.rint(frame * 255).clip(0, 255).astype(np.uint8))
            xy, mask = convert_result(result, cfg["pose_visibility"])
            coordinates.append(xy)
            masks.append(mask)
    xy, mask = np.stack(coordinates), np.stack(masks)
    if not mask.any():
        raise RuntimeError("No landmark detected in entire clip")
    return xy, mask


def cache(cfg):
    import cv2
    import numpy as np
    from vsl.augmentation import draw_rgb_params, augment_rgb
    from vsl.adapter import SCHEMA, rectify
    out = Path(cfg["cache_dir"])
    if out.exists():
        raise FileExistsError(f"Refuse overwrite cache: {out}")
    installed = importlib.metadata.version("mediapipe")
    if installed != cfg["extractor"]["version"]:
        raise RuntimeError(f"Extractor version {installed} != config {cfg['extractor']['version']}")
    if cfg["extractor"]["name"] != "mediapipe_holistic":
        raise ValueError("Unsupported extractor")
    if cfg["offline_variants"] < 0:
        raise ValueError("offline_variants must be nonnegative")
    if cfg["rectification"]["enabled"]:
        for key in ("aa_table", "fe_table"):
            if not cfg["rectification"][key] or not Path(cfg["rectification"][key]).is_file():
                raise ValueError(f"Missing rectification {key}")
    rows = read_json(Path(cfg["manifest_dir"]) / "videos.json")
    out.mkdir(parents=True)
    records, errors = [], []
    for row in rows:
        variants = range(cfg["offline_variants"] + 1) if row["split"] == "train" else range(1)
        for variant in variants:
            record = {**row, "parent_video": row["video_id"], "variant": variant,
                      "schema": SCHEMA, "extractor": cfg["extractor"],
                      "cache_signature": cache_signature(cfg)}
            try:
                started = time.perf_counter()
                path = Path(cfg["data_root"]) / row["path"]
                if sha256(path) != row["sha256"]:
                    raise RuntimeError("Video changed since manifest creation")
                clip, fps = decode(path)
                seed = int(digest([cfg["seed"], row["video_id"], variant])[:16], 16)
                record["seed"], record["augmentation"] = seed, None
                if variant:
                    params = draw_rgb_params(np.random.default_rng(seed), cfg["augmentation"])
                    clip, record["augmentation"] = augment_rgb(clip, params, cfg["rgb_resize"])
                elif cfg["rgb_resize"]:
                    clip = np.stack([cv2.resize(f, tuple(cfg["rgb_resize"])) for f in clip])
                xy, mask = extract(clip, cfg["extractor"])
                xy = rectify(xy, mask, cfg["rectification"])
                cache_name = digest([row["video_id"], variant]) + ".npz"
                np.savez_compressed(out / cache_name, coords=xy, mask=mask, schema=SCHEMA)
                record.update(cache_path=cache_name, status="ok", frames=len(xy), fps=fps,
                              cache_sha256=sha256(out / cache_name),
                              missing_joint_fraction=float(1 - mask.mean()),
                              preparation_seconds=time.perf_counter() - started)
            except Exception as exc:
                record.update(status="error", error=f"{type(exc).__name__}: {exc}")
                errors.append(record.copy())
            records.append(record)
            # Durable progress for inspection; incomplete cache is never accepted.
            write_json(out / "index.partial.json", records)
    write_json(out / "index.json", records)
    write_json(out / "errors.json", errors)
    write_json(out / "provenance.json", {
        "config": cfg, "schema": SCHEMA, "cache_signature": cache_signature(cfg),
        "manifest_sha256": sha256(Path(cfg["manifest_dir"]) / "videos.json"),
        "labels_sha256": sha256(Path(cfg["manifest_dir"]) / "labels.json"),
        "rectification_table_hashes": {
            k: sha256(cfg["rectification"][k]) for k in ("aa_table", "fe_table")
            if cfg["rectification"]["enabled"]},
        "complete": True})
    print(f"Cache: {len(records)} records, {len(errors)} errors. Errors have not been discarded.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["inventory", "cache"])
    parser.add_argument("--config", default="configs/train.json")
    args = parser.parse_args()
    cfg = load_config(args.config)
    (inventory if args.action == "inventory" else cache)(cfg)


if __name__ == "__main__":
    main()
