"""Check Kaggle's existing TensorFlow GPU stack, then install only unmet extras.

The runtime interpreter is used directly. TensorFlow, Keras, NumPy and installed
CUDA libraries are pinned with pip constraints and never selected for upgrade.
"""

from __future__ import annotations

import argparse
import importlib.metadata as md
import json
import subprocess
import sys
from pathlib import Path

try:
    from packaging.requirements import Requirement
    from packaging.specifiers import SpecifierSet
    from packaging.utils import canonicalize_name
    from packaging.version import Version
except ModuleNotFoundError:
    # pip ships packaging even when it is not exposed as a separate distribution.
    from pip._vendor.packaging.requirements import Requirement
    from pip._vendor.packaging.specifiers import SpecifierSet
    from pip._vendor.packaging.utils import canonicalize_name
    from pip._vendor.packaging.version import Version


PROTECTED = {"tensorflow", "tensorflow-cpu", "tensorflow-intel", "keras", "numpy"}


def installed_versions():
    return {canonicalize_name(d.metadata["Name"]): d.version
            for d in md.distributions() if d.metadata.get("Name")}


def protected_versions(versions):
    def is_protected(name):
        return (name in PROTECTED or name.startswith("nvidia-") or
                any(token in name for token in ("cuda", "cudnn", "cublas", "nccl")) or
                name == "triton")
    return {name: version for name, version in versions.items() if is_protected(name)}


def check_runtime(requirements, cache_only=False):
    if sys.version_info < (3, 11):
        raise RuntimeError(f"Python {sys.version.split()[0]} is too old; Python >=3.11 is required")

    import tensorflow as tf
    import keras
    import numpy as np

    versions = installed_versions()
    required = {canonicalize_name(r.name): r for r in requirements}
    for name, runtime_version in (("tensorflow", tf.__version__),
                                  ("keras", keras.__version__),
                                  ("numpy", np.__version__)):
        installed = versions.get(name)
        if installed is None or Version(installed) != Version(runtime_version):
            raise RuntimeError(f"{name} runtime {runtime_version} differs from installed distribution {installed}; "
                               "choose a consistent Kaggle image before installing extras")
    compatibility_errors = []
    media_version = versions.get("mediapipe")
    if not cache_only and sys.version_info >= (3, 13) and (media_version is None or
                                         not required["mediapipe"].specifier.contains(media_version)):
        compatibility_errors.append(
            "Python 3.13 has no published wheel for pinned MediaPipe 0.10.21 Legacy Holistic; "
            "use a Python 3.11/3.12 image for this RGB-to-landmark notebook")
    for name in ("tensorflow", "numpy"):
        req = required[name]
        if not req.specifier.contains(versions[name]):
            compatibility_errors.append(f"Existing {name} {versions[name]} violates {req}; "
                                        "this helper will not replace TensorFlow or NumPy")
    if Version(keras.__version__).major != 3 or Version(tf.keras.__version__).major != 3:
        raise RuntimeError(f"TensorFlow {tf.__version__} / Keras {keras.__version__} is not a compatible Keras 3 runtime")
    for raw in md.requires("tensorflow") or []:
        dependency = Requirement(raw)
        name = canonicalize_name(dependency.name)
        if name in {"keras", "numpy"} and (dependency.marker is None or dependency.marker.evaluate({"extra": ""})):
            if not dependency.specifier.contains(versions[name]):
                raise RuntimeError(f"Existing {name} {versions[name]} violates TensorFlow dependency {dependency}")
    python_spec = md.metadata("tensorflow").get("Requires-Python")
    if python_spec and not SpecifierSet(python_spec).contains(".".join(map(str, sys.version_info[:3]))):
        raise RuntimeError(f"Python {sys.version.split()[0]} violates TensorFlow Requires-Python {python_spec}")
    if not tf.test.is_built_with_cuda() or not tf.config.list_physical_devices("GPU"):
        raise RuntimeError("Existing TensorFlow cannot see a CUDA GPU. Enable a compatible Kaggle GPU image; "
                           "this helper will not install or replace GPU libraries")

    if not cache_only and media_version and required["mediapipe"].specifier.contains(media_version):
        import mediapipe as mp
        if not hasattr(mp, "solutions"):
            raise RuntimeError(f"MediaPipe {media_version} lacks solutions.holistic required by prepare_data.py")
    if compatibility_errors:
        raise RuntimeError("Incompatible Kaggle runtime before installation: " + "; ".join(compatibility_errors))
    print("Compatible existing runtime:", {k: versions[k] for k in ("tensorflow", "keras", "numpy")},
          "Python:", sys.version.split()[0], "GPUs:", tf.config.list_physical_devices("GPU"),
          "MediaPipe:", media_version, flush=True)
    return versions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--requirements", default="requirements.txt")
    parser.add_argument("--wheelhouse")
    parser.add_argument("--report", default="environment_versions.json")
    parser.add_argument("--cache-only", action="store_true", help="Train from an existing landmark archive without MediaPipe")
    args = parser.parse_args()
    requirements = [Requirement(line.strip()) for line in Path(args.requirements).read_text(encoding="utf-8").splitlines()
                    if line.strip() and not line.lstrip().startswith("#")]
    if args.cache_only:
        requirements = [r for r in requirements if canonicalize_name(r.name) != "mediapipe"]
    versions_before = check_runtime(requirements, cache_only=args.cache_only)  # No package installation before this check.
    report = Path(args.report).expanduser().resolve()
    report.parent.mkdir(parents=True, exist_ok=True)
    pinned = protected_versions(versions_before)
    constraints = report.with_name("kaggle_protected_constraints.txt")
    constraints.write_text("".join(f"{name}=={version}\n" for name, version in sorted(pinned.items())), encoding="utf-8")

    extras = [str(req) for req in requirements
              if canonicalize_name(req.name) not in {"tensorflow", "keras", "numpy"} and
              (versions_before.get(canonicalize_name(req.name)) is None or
               not req.specifier.contains(versions_before[canonicalize_name(req.name)]))]
    print("Installing only unmet non-core requirements:", extras, flush=True)
    if extras:
        command = [sys.executable, "-m", "pip", "install", "--disable-pip-version-check",
                   "--constraint", str(constraints), *extras]
        if args.wheelhouse:
            wheelhouse = Path(args.wheelhouse).expanduser().resolve()
            if not wheelhouse.is_dir():
                raise FileNotFoundError(f"Wheelhouse is missing: {wheelhouse}")
            command += ["--no-index", "--find-links", str(wheelhouse)]
        subprocess.run(command, check=True)

    versions_after = installed_versions()
    if protected_versions(versions_after) != pinned:
        raise RuntimeError("A protected TensorFlow/Keras/NumPy/CUDA distribution changed unexpectedly")
    for req in requirements:
        current = versions_after.get(canonicalize_name(req.name))
        if current is None or not req.specifier.contains(current):
            raise RuntimeError(f"Unmet requirement after installation: {req}; installed {current}")
    subprocess.run([sys.executable, "-c",
                    "import tensorflow as tf, keras, numpy, cv2, pandas, sklearn; " +
                    ("" if args.cache_only else "import mediapipe; assert hasattr(mediapipe, 'solutions'); ") +
                    "assert tf.config.list_physical_devices('GPU'), 'TensorFlow GPU not visible'; "
                    "print('TensorFlow GPU and preprocessing imports OK')"], check=True)
    report.write_text(json.dumps({d.metadata["Name"]: d.version for d in md.distributions()
                                  if d.metadata.get("Name")}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("Dependency report:", report)


if __name__ == "__main__":
    main()
