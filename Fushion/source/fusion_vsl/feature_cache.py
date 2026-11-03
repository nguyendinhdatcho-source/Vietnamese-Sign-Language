"""M05 durable float32 caches; NumPy + stdlib + M00/M01 only.

NPZ + SHA256 JSON receipt = committed record. Index is derived/recoverable.
An OS lock enforces one writer per recipe and releases on process termination.
"""
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import tempfile

import numpy as np

from .config import atomic_write_text, sha256_file
from .manifest import load_manifest, load_label_mapping, validate_manifest

VERSION = "m05_npz_float32_receipt_v1"
ARRAYS = ("features", "frame_indices", "valid_mask", "padding_mask", "detection_mask")


class CacheError(ValueError):
    """Invalid/incompatible/incomplete cache; never silently repair committed data."""


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def fingerprint(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def require(condition, message):
    if not condition:
        raise CacheError("M05: " + message)


def is_sha(value):
    return isinstance(value, str) and re.fullmatch("[0-9a-f]{64}", value) is not None


def portable_path(root, relative):
    require(isinstance(relative, str) and bool(relative), "empty relative cache path")
    normalized = relative.replace("\\", "/")
    rel = PurePosixPath(normalized)
    require(not rel.is_absolute() and not PureWindowsPath(normalized).drive
            and ".." not in rel.parts and ":" not in normalized, "unsafe cache path")
    root = Path(root).resolve()
    path = (root / Path(rel)).resolve()
    require(path.is_relative_to(root), "cache path escapes root")
    return path


class CacheContext:
    """Bind to complete manifest/mapping; reuse M01 counts/classes/leak checks.

    Metadata audit does not claim to re-decode all videos. Workers check each
    selected source file against the previously verified locked manifest.
    """
    def __init__(self, rows, mapping, manifest_sha256, data_root, protected_roots,
                 expected_counts, classes=100):
        require(is_sha(manifest_sha256), "invalid manifest checksum")
        require(sorted(mapping.values()) == list(range(classes)), "invalid label mapping")
        self.audit = validate_manifest(rows, data_root, expected_counts, classes,
                                       check_files=False, label_mapping=mapping)
        self.rows = {r["video_id"]: deepcopy(r) for r in rows}
        self.mapping = deepcopy(mapping)
        self.binding = {"manifest_sha256": manifest_sha256,
                        "label_mapping_sha256": fingerprint(mapping)}
        self.protected_roots = [Path(p).resolve() for p in [data_root, *protected_roots]]

    def check_output(self, path):
        path = Path(path).resolve()
        require(not any(path.is_relative_to(p) for p in self.protected_roots),
                "output must be outside original dataset")

    def check_identity(self, record):
        vid = record.get("video_id")
        require(vid in self.rows, f"unknown video_id {vid!r}")
        row = self.rows[vid]
        for key in ("video_id", "split", "group_id"):
            require(record.get(key) == row[key], f"{vid}: {key} mismatch")
        require(type(record.get("label_id")) is int and record["label_id"] == int(row["label_id"]),
                f"{vid}: label_id mismatch")
        require(record.get("source_metadata") == row, f"{vid}: manifest source metadata mismatch")
        require(record.get("context") == self.binding, f"{vid}: manifest/label mapping mismatch")


def load_context(paths, cfg):
    manifest = Path(paths["manifest_csv"])
    preflight = json.loads((Path(paths["reports_dir"]) / "preflight.json").read_text(encoding="utf-8"))
    digest = sha256_file(manifest)
    require(preflight.get("status") == "PASS" and preflight.get("manifest_sha256") == digest,
            "locked manifest does not match PASS preflight")
    return CacheContext(load_manifest(manifest), load_label_mapping(paths["handoff_root"], cfg["num_classes"]),
                        digest, paths["data_root"], [paths["handoff_root"]], cfg["expected_counts"], cfg["num_classes"])


def make_recipe(modality, *, m02_version, m02_sha256, expected_provenance, settings, code_hashes):
    require(modality in ("rgb", "landmark"), "unknown modality")
    require(is_sha(m02_sha256) and bool(m02_version), "invalid M02 fingerprint")
    require(bool(expected_provenance) and bool(code_hashes)
            and all(is_sha(v) for v in code_hashes.values()), "missing recipe provenance/code hashes")
    return {"cache_version": VERSION, "modality": modality, "m02_version": m02_version,
            "m02_sha256": m02_sha256, "expected_provenance": deepcopy(expected_provenance),
            "settings": deepcopy(settings), "code_hashes": deepcopy(code_hashes), "dtype": "float32"}


def verify_cache_version(record, recipe, context):
    require(record.get("cache_version") == VERSION, "stale cache version")
    require(record.get("recipe_sha256") == fingerprint(recipe), "stale preprocessing/encoder recipe")
    context.check_identity(record)
    prov = record.get("provenance", {})
    require(isinstance(prov, dict) and bool(prov), "missing provenance")
    require(record.get("provenance_sha256") == fingerprint(prov), "provenance fingerprint mismatch")
    for key, value in recipe["expected_provenance"].items():
        require(prov.get(key) == value, f"encoder/model fingerprint mismatch: {key}")
    for key, value in (("m02_preprocessing_version", recipe["m02_version"]),
                       ("m02_preprocessing_sha256", recipe["m02_sha256"])):
        require(prov.get(key) == value, f"{key} mismatch")


def validate_feature_shapes(record, modality):
    shape = (16, 1280 if modality == "rgb" else 300)
    array = record.get("features")
    require(isinstance(array, np.ndarray) and array.shape == shape and array.dtype == np.float32,
            f"{modality} features must be float32 {shape}")
    require(np.isfinite(array).all(), "features contain NaN/Inf")
    for key, dtype in (("frame_indices", np.int64), ("valid_mask", np.bool_), ("padding_mask", np.bool_)):
        a = record.get(key)
        require(isinstance(a, np.ndarray) and a.shape == (16,) and a.dtype == dtype, f"invalid {key}")
    indices, valid, padding = (record[k] for k in ("frame_indices", "valid_mask", "padding_mask"))
    n = int(valid.sum())
    require(n > 0 and np.array_equal(valid, np.arange(16) < n)
            and np.array_equal(padding, ~valid), "invalid valid_mask/padding_mask")
    require((indices >= 0).all() and (np.diff(indices[:n]) > 0).all()
            and (indices[n:] == indices[n-1]).all(), "invalid frame_indices temporal order/padding")
    if modality == "landmark":
        mask = record.get("detection_mask")
        require(isinstance(mask, np.ndarray) and mask.shape == (16,75) and mask.dtype == np.bool_, "invalid detection_mask")
        points = array.reshape(16,75,4)
        require((points[~mask] == 0).all(), "missing landmarks must remain zero with false detection_mask")
        require(((points[:,:,3] >= 0) & (points[:,:,3] <= 1)).all(), "invalid landmark confidence")
        require(np.array_equal(array[n:], np.repeat(array[n-1:n], 16-n, axis=0))
                and np.array_equal(mask[n:], np.repeat(mask[n-1:n], 16-n, axis=0)), "landmark padding mismatch")


def validate_record(record, recipe, context):
    verify_cache_version(record, recipe, context)
    modality = recipe["modality"]
    require(record.get("modality") == modality, "wrong modality")
    validate_feature_shapes(record, modality)
    require(is_sha(record.get("input_frames_sha256")), "missing input-frame SHA256")
    q = record.get("quality_metadata", {})
    require(isinstance(q, dict) and q.get("preprocessing_sha256") == recipe["m02_sha256"]
            and q.get("preprocessing", {}).get("preprocess_version") == recipe["m02_version"],
            "missing/incompatible M02 quality/preprocessing metadata")
    require(hashlib.sha256(json.dumps(q["preprocessing"], sort_keys=True).encode()).hexdigest()
            == recipe["m02_sha256"], "M02 preprocessing metadata hash mismatch")
    require(q.get("sampled_original_indices") == record["frame_indices"].tolist()
            and q.get("padding_count") == int(record["padding_mask"].sum())
            and type(q.get("valid_frames")) is int and q["valid_frames"] >= int(record["valid_mask"].sum()),
            "M02 quality indices/counts mismatch")
    require(isinstance(record.get("quality_flags"), list), "missing quality flags")
    require(isinstance(record.get("timings"), dict) and bool(record["timings"]), "missing inference timings")
    run = record.get("cache_run", {})
    require(isinstance(run, dict) and all(run.get(k) for k in ("run_id", "created_utc", "python_version")),
            "missing processing session metadata")
    prov = record["provenance"]
    if modality == "rgb":
        require(prov.get("weights_verified") is True and prov.get("weights_identifier") == "imagenet"
                and all(is_sha(prov.get(k)) for k in ("weights_sha256", "model_weights_sha256"))
                and all(prov.get(k) for k in ("encoder_version", "rgb_preprocessing_version",
                                            "tensorflow_version", "keras_version")), "missing pretrained RGB provenance")
    else:
        require(all(prov.get(k) for k in ("module_version", "graph_version", "mediapipe_version"))
                and all(is_sha(prov.get(k)) for k in ("source_graph_sha256", "derived_graph_sha256", "options_sha256"))
                and bool(prov.get("model_sha256"))
                and all(is_sha(v) for v in prov["model_sha256"].values()), "missing landmark graph/model provenance")
        require(isinstance(record.get("landmark_quality"), dict)
                and isinstance(record["landmark_quality"].get("rates"), dict), "missing landmark detection rates")
    canonical({k:v for k,v in record.items() if k not in ARRAYS})


def adapt_features(result, clip, row, recipe, context, run_metadata):
    """Add boundary metadata without changing M02/M03/M04 result contracts."""
    require(result.get("video_id") == clip.get("video_id") == row["video_id"], "adapter video_id mismatch")
    for key in ("frame_indices", "valid_mask", "padding_mask"):
        require(np.array_equal(result[key], clip[key]), f"adapter {key} mismatch")
    require(result["quality_metadata"] == clip["quality_metadata"], "adapter lost M02 quality metadata")
    frames = clip["frames"]
    require(frames.shape == (16,224,224,3) and frames.dtype == np.uint8, "adapter needs original RGB uint8 frames")
    out = deepcopy(result)
    if "group_id" in out:
        require(out["group_id"] == row["group_id"], "adapter group_id mismatch")
    out.update(group_id=row["group_id"], context=deepcopy(context.binding), cache_version=VERSION,
               modality=recipe["modality"], recipe_sha256=fingerprint(recipe),
               input_frames_sha256=hashlib.sha256(frames.tobytes(order="C")).hexdigest(),
               provenance_sha256=fingerprint(result["provenance"]), cache_run=deepcopy(run_metadata))
    validate_record(out, recipe, context)
    return out


@contextmanager
def writer_lock(folder):
    """Nonblocking Windows/POSIX lock; no stale PID lock after a crash."""
    path = folder / ".writer.lock"
    with path.open("a+b") as stream:
        if path.stat().st_size == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise CacheError("M05: another writer owns this recipe") from exc
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


class FeatureCache:
    def __init__(self, root, recipe, context):
        context.check_output(root)
        require(recipe.get("cache_version") == VERSION and recipe.get("dtype") == "float32"
                and recipe.get("modality") in ("rgb", "landmark"), "unsupported recipe")
        self.context, self.recipe = context, deepcopy(recipe)
        self.root = Path(root).resolve()
        self.folder = self.root / recipe["modality"] / fingerprint(recipe)
        context.check_output(self.folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.recipe_path = self.folder / "recipe.json"
        with writer_lock(self.folder):
            if self.recipe_path.exists():
                require(json.loads(self.recipe_path.read_text(encoding="utf-8")) == recipe, "recipe changed/corrupt")
            else:
                atomic_write_text(self.recipe_path, canonical(recipe))

    def paths(self, video_id):
        require(video_id in self.context.rows, f"unknown video_id {video_id!r}")
        name = hashlib.sha256(video_id.encode("utf-8")).hexdigest()
        return self.folder / (name + ".npz"), self.folder / (name + ".json")

    def state(self, video_id):
        data, receipt = self.paths(video_id)
        if receipt.exists():
            self.load(video_id)
            return "complete"
        return "incomplete" if data.exists() else "missing"

    def load(self, video_id):
        data, receipt_path = self.paths(video_id)
        require(receipt_path.is_file(), f"{video_id}: missing/incomplete {self.recipe['modality']} cache")
        try:
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            require(receipt.get("video_id") == video_id and receipt.get("file") == data.name
                    and receipt.get("recipe_sha256") == fingerprint(self.recipe), "invalid commit receipt identity")
            require(data.is_file() and sha256_file(data) == receipt.get("sha256"), f"{video_id}: file checksum mismatch")
            out = self._read_npz(data)
            require(fingerprint({k:v for k,v in out.items() if k not in ARRAYS}) == receipt.get("metadata_sha256"),
                    "metadata checksum mismatch")
            validate_record(out, self.recipe, self.context)
            require(out["video_id"] == video_id, "payload video_id mismatch")
            return out
        except (OSError, KeyError, ValueError) as exc:
            if isinstance(exc, CacheError):
                raise
            raise CacheError(f"M05: {video_id}: corrupt cache: {exc}") from exc

    @staticmethod
    def _read_npz(path):
        with np.load(path, allow_pickle=False) as archive:
            require("metadata" in archive.files, "NPZ missing metadata")
            metadata = json.loads(str(archive["metadata"].item()))
            require(isinstance(metadata, dict) and not set(metadata).intersection(ARRAYS), "invalid NPZ metadata")
            expected = {"metadata", "features", "frame_indices", "valid_mask", "padding_mask"}
            if metadata.get("modality") == "landmark":
                expected.add("detection_mask")
            require(set(archive.files) == expected, "unexpected/missing NPZ arrays")
            for key in expected - {"metadata"}:
                metadata[key] = archive[key].copy()
                metadata[key].setflags(write=False)
            return metadata

    def save(self, record):
        validate_record(record, self.recipe, self.context)
        vid = record["video_id"]
        data, receipt_path = self.paths(vid)
        with writer_lock(self.folder):
            if receipt_path.exists():
                previous = self.load(vid)
                require(all(np.array_equal(previous[k], record[k]) for k in ARRAYS if k in record)
                        and previous["input_frames_sha256"] == record["input_frames_sha256"],
                        f"{vid}: refusing overwrite of different valid cache")
                ignore = {*ARRAYS, "cache_run", "timings"}
                require({k:v for k,v in previous.items() if k not in ignore}
                        == {k:v for k,v in record.items() if k not in ignore},
                        f"{vid}: refusing conflicting metadata for valid cache")
                return {"status": "reused", "receipt": receipt_path}
            metadata = {k:v for k,v in record.items() if k not in ARRAYS}
            arrays = {k:record[k] for k in ARRAYS if k in record}
            fd, name = tempfile.mkstemp(prefix=data.stem + ".", suffix=".tmp", dir=self.folder)
            temporary = Path(name)
            try:
                with os.fdopen(fd, "wb") as stream:
                    np.savez_compressed(stream, metadata=np.array(canonical(metadata)), **arrays)
                    stream.flush()
                    os.fsync(stream.fileno())
                validate_record(self._read_npz(temporary), self.recipe, self.context)
                digest = sha256_file(temporary)
                os.replace(temporary, data)
                require(sha256_file(data) == digest, "post-write checksum mismatch")
                atomic_write_text(receipt_path, canonical({"video_id":vid, "file":data.name,
                    "sha256":digest, "metadata_sha256":fingerprint(metadata), "recipe_sha256":fingerprint(self.recipe)}))
            finally:
                temporary.unlink(missing_ok=True)
        return {"status": "written", "receipt": receipt_path}

    def rebuild_index(self):
        with writer_lock(self.folder):
            entries = {}
            for path in sorted(self.folder.glob("*.json")):
                if path.name in ("recipe.json", "index.json"):
                    continue
                receipt = json.loads(path.read_text(encoding="utf-8"))
                vid = receipt.get("video_id")
                require(vid not in entries and self.paths(vid)[1] == path, "invalid/duplicate receipt filename")
                self.load(vid)
                entries[vid] = {"receipt":path.name, "receipt_sha256":sha256_file(path), **receipt}
            index = {"cache_version":VERSION, "recipe_sha256":fingerprint(self.recipe),
                     "context":self.context.binding, "records":entries,
                     "uncommitted":sorted(p.name for p in self.folder.glob("*.npz") if not p.with_suffix(".json").exists()),
                     "temporary":sorted(p.name for p in self.folder.glob("*.tmp"))}
            target = self.folder / "index.json"
            content = canonical(index)
            if not target.exists() or target.read_text(encoding="utf-8") != content:
                atomic_write_text(target, content)
            return index


def save_rgb_features(cache, record):
    require(cache.recipe["modality"] == "rgb", "need RGB cache")
    return cache.save(record)


def load_rgb_features(cache, video_id):
    require(cache.recipe["modality"] == "rgb", "need RGB cache")
    return cache.load(video_id)


def save_landmark_features(cache, record):
    require(cache.recipe["modality"] == "landmark", "need landmark cache")
    return cache.save(record)


def load_landmark_features(cache, video_id):
    require(cache.recipe["modality"] == "landmark", "need landmark cache")
    return cache.load(video_id)


def validate_feature_alignment(rgb, landmark, rgb_cache, landmark_cache):
    validate_record(rgb, rgb_cache.recipe, rgb_cache.context)
    validate_record(landmark, landmark_cache.recipe, landmark_cache.context)
    for key in ("video_id", "split", "label_id", "group_id", "context", "input_frames_sha256", "source_metadata"):
        require(rgb[key] == landmark[key], f"alignment {key} mismatch")
    for key in ("frame_indices", "valid_mask", "padding_mask"):
        require(np.array_equal(rgb[key], landmark[key]), f"alignment {key} mismatch")
    for key in ("m02_preprocessing_version", "m02_preprocessing_sha256"):
        require(rgb["provenance"][key] == landmark["provenance"][key], f"alignment {key} mismatch")
    return True


def join_features_by_video_id(rgb_records, landmark_records, rgb_cache, landmark_cache):
    def keyed(records):
        out = {}
        for record in records:
            vid = record["video_id"]
            require(vid not in out, f"duplicate video_id {vid}")
            out[vid] = record
        return out
    rgb, landmark = keyed(rgb_records), keyed(landmark_records)
    require(set(rgb) == set(landmark) and bool(rgb), "missing RGB or landmark modality")
    samples = []
    for vid in sorted(rgb):
        r, l = rgb[vid], landmark[vid]
        validate_feature_alignment(r, l, rgb_cache, landmark_cache)
        sample = {k:deepcopy(r[k]) for k in ("video_id", "label_id", "split", "group_id", "frame_indices", "valid_mask", "padding_mask")}
        sample.update(rgb_features=r["features"], landmark_features=l["features"], detection_mask=l["detection_mask"],
                      quality_metadata={"rgb":deepcopy(r["quality_metadata"]), "landmark":deepcopy(l["quality_metadata"]),
                          "rgb_flags":deepcopy(r["quality_flags"]), "landmark_flags":deepcopy(l["quality_flags"]),
                          "landmark_detection":deepcopy(l["landmark_quality"])},
                      provenance={"rgb":deepcopy(r["provenance"]), "landmark":deepcopy(l["provenance"])})
        samples.append(sample)
    return samples


def build_aligned_dataset(rgb_cache, landmark_cache, video_ids):
    ids = list(video_ids)
    require(len(ids) == len(set(ids)) and bool(ids), "empty/duplicate requested video_ids")
    return join_features_by_video_id([rgb_cache.load(v) for v in ids],
                                    [landmark_cache.load(v) for v in reversed(ids)], rgb_cache, landmark_cache)


def write_alignment_manifest(rgb_cache, landmark_cache, video_ids):
    """Persist validated references, without copying tensors or implementing M06."""
    from .cache_workflow import store_reference
    require(rgb_cache.root == landmark_cache.root, "aligned stores require a common portable root")
    samples = build_aligned_dataset(rgb_cache, landmark_cache, video_ids)
    records = {}
    for sample in samples:
        vid = sample["video_id"]
        records[vid] = {mod: {"receipt":cache.paths(vid)[1].relative_to(cache.root).as_posix(),
                             "sha256":sha256_file(cache.paths(vid)[1])}
                       for mod,cache in (("rgb",rgb_cache),("landmark",landmark_cache))}
    value = {"cache_version":VERSION, "context":rgb_cache.context.binding,
             "rgb":store_reference(rgb_cache), "landmark":store_reference(landmark_cache), "records":records}
    path = rgb_cache.root/"aligned"/(fingerprint(value)+".json")
    rgb_cache.context.check_output(path)
    content = canonical(value)
    if path.exists():
        require(path.read_text(encoding="utf-8") == content, "corrupt aligned manifest")
    else:
        atomic_write_text(path,content)
    return path


def load_feature_manifest(path, cache_root, context):
    """Reload an aligned dataset from another process/machine with remapped roots."""
    from .cache_workflow import open_store_reference
    path = Path(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    require(path.stem == fingerprint(value), "aligned manifest checksum mismatch")
    require(value.get("cache_version") == VERSION and value.get("context") == context.binding,
            "stale aligned manifest/mapping")
    stores = {mod:open_store_reference(cache_root,value[mod],context) for mod in ("rgb","landmark")}
    for vid, references in value["records"].items():
        for mod, cache in stores.items():
            receipt = portable_path(cache_root,references[mod]["receipt"])
            require(receipt == cache.paths(vid)[1] and sha256_file(receipt) == references[mod]["sha256"],
                    f"{vid}: aligned receipt checksum/path mismatch")
    return build_aligned_dataset(stores["rgb"],stores["landmark"],value["records"])
