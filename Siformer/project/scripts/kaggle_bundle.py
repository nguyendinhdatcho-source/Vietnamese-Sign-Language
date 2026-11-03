"""Archive exact TensorFlow resume state, source, manifests and landmark cache."""
import argparse
import json
import shutil
import sys
import zipfile
from pathlib import Path


def build_bundle(config_path, output):
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    from vsl.common import load_config, read_json, sha256
    from vsl_tf.engine import resolve_pointer, resolve_checkpoint
    cfg = load_config(config_path)
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    run = Path(cfg["run_dir"]).resolve()
    entries = {}
    for name in ("vsl", "vsl_tf", "vendor", "configs", "scripts", "tests", "docs", "notebooks"):
        for file in (root / name).rglob("*"):
            if file.is_file() and "__pycache__" not in file.parts and file.suffix != ".pyc":
                entries["project/" + file.relative_to(root).as_posix()] = file
    for file in root.glob("*"):
        if file.is_file() and (file.suffix in (".py", ".md", ".txt", ".ini") or file.name == ".gitignore"):
            entries["project/" + file.name] = file
    for prefix, directory in (("manifests", Path(cfg["manifest_dir"])), ("cache", Path(cfg["cache_dir"]))):
        for file in directory.rglob("*"):
            if file.is_file() and not file.name.endswith(".tmp"):
                entries[prefix + "/" + file.relative_to(directory).as_posix()] = file
    pointers = {}
    for name in ("best.json", "last.json"):
        metadata = resolve_pointer(run / name)
        state = read_json(metadata)
        resolve_checkpoint(metadata.parent / state["checkpoint"])
        entries["run/" + metadata.name] = metadata
        for filename in state["files"]:
            entries["run/" + filename] = metadata.parent / filename
        pointers["run/" + name] = {"checkpoint": metadata.name, "sha256": sha256(metadata)}
    for file in run.rglob("*"):
        if file.is_file() and file.name not in ("best.json", "last.json") and file.suffix != ".index" and ".data-" not in file.name and not (file.name.startswith("epoch_") and file.suffix == ".json"):
            entries["run/" + file.relative_to(run).as_posix()] = file
    entries["run/config.json"] = Path(config_path)
    entries["run/runtime_config.json"] = Path(config_path)
    environment = Path(config_path).parent / "environment_versions.json"
    if environment.is_file():
        entries["run/environment_versions.json"] = environment
    raw = sum(file.stat().st_size for file in entries.values())
    output.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(output.parent).free < raw + 256 * 1024**2:
        raise RuntimeError("Not enough free disk space for the archive")
    print(f"Archiving {len(entries)} files; raw {raw / 1024**3:.2f} GiB", flush=True)
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as z:
        for name, file in entries.items():
            z.write(file, name)
        for name, pointer in pointers.items():
            z.writestr(name, json.dumps(pointer, indent=2))
        z.writestr("bundle.json", json.dumps({"format": "vsl-kaggle-tensorflow-v1",
                                                  "project": "project", "run": "run",
                                                  "manifests": "manifests", "cache": "cache"}))
    print("Bundle ready:", output)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    build_bundle(a.config, a.output)


if __name__ == "__main__":
    main()
