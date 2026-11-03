"""Clip-consistent RGB transforms and speed-aware temporal indices."""
import numpy as np


def sample_indices(length, frames, speed=1.0):
    if length < 1 or frames < 2 or speed <= 0:
        raise ValueError("Invalid temporal sampling arguments")
    base = np.linspace(0, length - 1, frames, dtype=np.float64)
    # Speed is applied before integer rounding, never followed by uniform resampling.
    return np.floor(base * speed).astype(np.int64).clip(0, length - 1)


def temporal_sample(coords, mask, frames, speed=1.0):
    if len(coords) != len(mask):
        raise ValueError("Coordinates/mask time mismatch")
    indices = sample_indices(len(coords), frames, speed)
    return coords[indices], mask[indices], indices


def draw_rgb_params(rng, cfg):
    return {"scale": float(np.sqrt(rng.uniform(*cfg["crop_area"]))),
            "offset_x": float(rng.random()), "offset_y": float(rng.random()),
            **{k: float(rng.uniform(*cfg[k])) for k in ("brightness", "contrast", "saturation")}}


def crop_box(height, width, params):
    # Continuous box has exactly the source aspect ratio; pixel raster is rounded.
    scale = params["scale"]
    w, h = max(1, round(width * scale)), max(1, round(height * scale))
    x, y = round((width - w) * params["offset_x"]), round((height - h) * params["offset_y"])
    return x, y, w, h


def augment_rgb(clip, params, resize=None):
    """RGB float32 T,H,W,3 in [0,1]; one crop and contrast center for the clip."""
    import cv2
    if clip.ndim != 4 or clip.shape[-1] != 3 or not np.isfinite(clip).all():
        raise ValueError("Expected finite RGB clip T,H,W,3")
    if clip.min() < 0 or clip.max() > 1:
        raise ValueError("Apply RGB augmentation before normalization")
    x, y, w, h = crop_box(clip.shape[1], clip.shape[2], params)
    out = clip[:, y:y+h, x:x+w].copy()
    out *= params["brightness"]
    center = float(out.mean())  # shared scalar, not a separate mean per frame
    out = (out - center) * params["contrast"] + center
    gray = (out * np.array([0.299, 0.587, 0.114], dtype=np.float32)).sum(-1, keepdims=True)
    out = gray + params["saturation"] * (out - gray)
    out = out.clip(0, 1)
    size = tuple(resize) if resize else (clip.shape[2], clip.shape[1])
    out = np.stack([cv2.resize(f, size) for f in out]).astype(np.float32)
    return out, {**params, "crop_xywh": [x, y, w, h], "contrast_center": center,
                 "resize_wh": list(size)}
