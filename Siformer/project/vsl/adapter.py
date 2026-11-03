"""Semantic MediaPipe -> official Siformer ordering; no shape-only joint aliases."""
import numpy as np
from .common import vendor_imports

HAND_ORDER = [0, 8, 7, 6, 5, 12, 11, 10, 9, 16, 15, 14, 13, 20, 19, 18, 17, 4, 3, 2, 1]
BODY_NAMES = ["nose", "neck", "rightEye", "leftEye", "rightEar", "leftEar",
              "rightShoulder", "leftShoulder", "rightElbow", "leftElbow", "rightWrist", "leftWrist"]
POSE_ORDER = [0, None, 5, 2, 8, 7, 12, 11, 14, 13, 16, 15]
SCHEMA = "mp_holistic_xy_up_left21_right21_body12_v1"
HAND_EDGES = [(0, 1), (1, 2), (2, 3), (3, 4),
              (0, 5), (5, 6), (6, 7), (7, 8),
              (5, 9), (9, 10), (10, 11), (11, 12),
              (9, 13), (13, 14), (14, 15), (15, 16),
              (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)]


def convert_result(result, visibility=0.5):
    """Output 54 points in raw normalized image x, upward y; neck = shoulder midpoint."""
    coords = np.zeros((54, 2), np.float32)
    mask = np.zeros(54, bool)
    for start, hand in ((0, result.left_hand_landmarks), (21, result.right_hand_landmarks)):
        if hand:
            xy = np.array([(p.x, 1 - p.y) for p in hand.landmark], np.float32)
            coords[start:start+21] = xy[HAND_ORDER]
            mask[start:start+21] = np.isfinite(xy[HAND_ORDER]).all(-1)
    if result.pose_landmarks:
        pose = result.pose_landmarks.landmark
        for j, p in enumerate(POSE_ORDER):
            if p is not None and pose[p].visibility >= visibility:
                coords[42+j] = [pose[p].x, 1 - pose[p].y]
                mask[42+j] = True
        if mask[48] and mask[49]:
            coords[43] = (coords[48] + coords[49]) / 2
            mask[43] = True
    mask &= np.isfinite(coords).all(-1)
    coords[~mask] = 0
    return coords, mask


def normalize(coords, mask):
    """Reuse upstream body/hand formulas; missing joints explicitly restored to -0.5."""
    vendor_imports()
    from normalization.body_normalization import normalize_single_dict as body_norm
    from normalization.hand_normalization import normalize_single_dict as hand_norm, HAND_IDENTIFIERS
    coords = np.asarray(coords, np.float32).copy()
    mask = np.asarray(mask, bool)
    if coords.shape != (len(coords), 54, 2) or mask.shape != coords.shape[:-1]:
        raise ValueError(f"Schema/shape mismatch: {coords.shape}, {mask.shape}")
    if not np.isfinite(coords[mask]).all():
        raise ValueError("Nonfinite observed landmark")
    coords[~mask] = 0
    names = [n + "_0" for n in HAND_IDENTIFIERS] + [n + "_1" for n in HAND_IDENTIFIERS] + BODY_NAMES
    row = {n: coords[:, i].tolist() for i, n in enumerate(names)}
    row = hand_norm(body_norm(row))
    out = np.stack([np.asarray(row[n], np.float32) for n in names], axis=1) - 0.5
    out[~mask] = -0.5
    if not np.isfinite(out).all():
        raise ValueError("Nonfinite normalized landmark")
    return out


def rectify(coords, mask, cfg):
    """Optional official functions, only with externally verified motion tables."""
    if not cfg["enabled"]:
        return coords
    if not cfg["aa_table"] or not cfg["fe_table"]:
        raise ValueError("Official motion tables are absent upstream; provide verified AA/FE tables")
    import ast
    import pandas as pd
    vendor_imports()
    from rectifications.abduction_adduction_rectification import rectify_each_hand_abduction_and_addiction
    from rectifications.flexion_extension_rectification import (
        check_thumb_flexion_direction, rectify_thumb_flexion_and_extension,
        rectify_other_flexion_and_extension)
    aa, fe = pd.read_csv(cfg["aa_table"]), pd.read_csv(cfg["fe_table"])
    def columns(table):
        return [[ast.literal_eval(v) for v in table[k]] for k in
                ("ranges", "paired_joint_index", "associated_joint_index")]
    inverse = np.argsort(HAND_ORDER)
    out = coords.copy()
    for t in range(len(out)):
        for start, side in ((0, "left"), (21, "right")):
            # Never rectify partial/absent hands using zero placeholders.
            if not mask[t, start:start+21].all():
                continue
            hand = out[t, start:start+21][inverse].tolist()
            hand, _ = rectify_each_hand_abduction_and_addiction(hand, side, *columns(aa), cfg["alpha"])
            direction = check_thumb_flexion_direction(hand)
            hand, _ = rectify_thumb_flexion_and_extension(
                hand, direction, *columns(fe[fe.fingers == "thumb"]), cfg["alpha"])
            hand, _ = rectify_other_flexion_and_extension(
                hand, *columns(fe[fe.fingers != "thumb"]), cfg["alpha"])
            out[t, start:start+21] = np.array(hand)[HAND_ORDER]
    if not np.isfinite(out).all():
        raise ValueError("Rectification produced NaN/Inf")
    return out
