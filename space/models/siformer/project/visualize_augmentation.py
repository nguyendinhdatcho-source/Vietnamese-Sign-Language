"""Produce real before/after examples on demand, never during import."""
import argparse
from pathlib import Path

from vsl.common import load_config, read_json, write_json


def visualize(cfg, output):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from prepare_data import decode
    from vsl.augmentation import augment_rgb, draw_rgb_params, sample_indices
    from vsl.adapter import HAND_ORDER, HAND_EDGES
    out = Path(output)
    if out.exists():
        raise FileExistsError(out)
    records = read_json(Path(cfg["cache_dir"]) / "index.json")
    by_key = {(r["video_id"], r["variant"]): r for r in records}
    selected = next((r for r in records if r["split"] == "train" and r["variant"] == 1
                     and r["status"] == "ok" and by_key[(r["video_id"], 0)]["status"] == "ok"), None)
    if selected is None:
        raise ValueError("Visualization needs one successful train original + RGB-augmented cache pair")
    original = by_key[(selected["video_id"], 0)]
    before, _ = decode(Path(cfg["data_root"]) / selected["path"])
    params = draw_rgb_params(np.random.default_rng(selected["seed"]), cfg["augmentation"])
    after, actual = augment_rgb(before, params, cfg["rgb_resize"])
    if actual != selected["augmentation"]:
        raise ValueError("Cannot reproduce cached RGB augmentation")
    out.mkdir(parents=True)
    positions = np.linspace(0, len(before)-1, 4).astype(int)
    fig, axs = plt.subplots(2, 4, figsize=(14, 7))
    for j, t in enumerate(positions):
        axs[0, j].imshow(before[t])
        axs[1, j].imshow(after[t])
        axs[0, j].set_title(f"Original frame {t}")
        axs[1, j].set_title(f"Augmented frame {t}")
        axs[0, j].axis("off")
        axs[1, j].axis("off")
    fig.tight_layout()
    fig.savefig(out / "rgb_before_after.png", dpi=150)
    plt.close(fig)
    arrays = []
    for r in (original, selected):
        with np.load(Path(cfg["cache_dir"]) / r["cache_path"]) as z:
            arrays.append((z["coords"], z["mask"]))
    inverse = np.argsort(HAND_ORDER)
    fig, axs = plt.subplots(2, 4, figsize=(14, 7))
    valid_points = np.concatenate([xy[m] for xy, m in arrays])
    lo, hi = np.minimum(valid_points.min(0), 0), np.maximum(valid_points.max(0), 1)
    body_edges = [(6, 7), (6, 8), (8, 10), (7, 9), (9, 11), (0, 1), (1, 6), (1, 7)]
    for row, (xy, mask) in enumerate(arrays):
        for col, t in enumerate(positions):
            ax = axs[row, col]
            ax.scatter(xy[t, mask[t], 0], xy[t, mask[t], 1], s=4)
            edges = [(start + int(inverse[a]), start + int(inverse[b]))
                     for start in (0, 21) for a, b in HAND_EDGES]
            edges += [(42+a, 42+b) for a, b in body_edges]
            for a, b in edges:
                if mask[t, a] and mask[t, b]:
                    ax.plot(xy[t, [a, b], 0], xy[t, [a, b], 1], linewidth=0.6)
            ax.set(xlim=(lo[0], hi[0]), ylim=(lo[1], hi[1]), aspect="equal",
                   title=f"{'Original' if row == 0 else 'Re-extracted'} frame {t}")
    fig.tight_layout()
    fig.savefig(out / "skeleton_before_after.png", dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots()
    for speed in (0.9, 1.0, 1.1):
        ax.plot(sample_indices(len(before), cfg["frames"], speed), label=f"speed={speed}")
    ax.set(xlabel="Output timestep", ylabel="Source frame index")
    ax.legend()
    fig.savefig(out / "temporal_indices.png", dpi=150)
    plt.close(fig)
    write_json(out / "example.json", selected)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/train.json")
    p.add_argument("--output", required=True)
    args = p.parse_args()
    visualize(load_config(args.config), args.output)


if __name__ == "__main__":
    main()
