"""M02: read-only RGB decoding, conservative spatial QC and shared clip preparation."""
from pathlib import Path
from time import perf_counter
import hashlib
import json
import math

import cv2
import numpy as np

from .config import sha256_file
from .sampling import sample_frames

PREPROCESS_VERSION = 'm02_rgb_qc_uniform_v1'
DEFAULT_QC = {'flat_max_channel_std': 1.0, 'min_valid_frames': 1}


class VideoProcessingError(ValueError):
    """A video cannot satisfy the M02 contract; never silently skip it."""


def _qc_policy(policy=None):
    policy = {} if policy is None else policy
    if not isinstance(policy, dict) or set(policy) - set(DEFAULT_QC):
        raise ValueError('M02: unsupported QC policy entries')
    result = {**DEFAULT_QC, **policy}
    threshold = result['flat_max_channel_std']
    if (isinstance(threshold, bool) or not isinstance(threshold, (int, float))
            or not math.isfinite(threshold) or not 0 <= threshold <= 255):
        raise ValueError('M02: flat_max_channel_std must be finite in [0,255]')
    minimum = result['min_valid_frames']
    if type(minimum) is not int or minimum < 1:
        raise ValueError('M02: min_valid_frames must be a positive integer')
    return result


def _frame_error(frame):
    if not isinstance(frame, np.ndarray) or frame.size == 0:
        return 'empty_frame'
    if frame.ndim != 3 or frame.shape[-1] != 3 or min(frame.shape[:2]) < 1:
        return 'invalid_shape'
    if frame.dtype != np.uint8:
        return 'invalid_dtype'
    return None


def decode_video(path, *, strict_frame_count=True):
    """Return RGB uint8 [N,H,W,3] + zero-based decode-order metadata.

    Sequential reads, no seeking or implicit frame removal. A known metadata
    frame-count mismatch fails by default (possible truncation/codec issue).
    OpenCV cannot distinguish every EOF from decoder failure; unknown counts
    are recorded as a limitation. prepare_clip also checks manifest counts.
    """
    path = Path(path)
    if not path.is_file():
        raise VideoProcessingError(f'M02 decode: video does not exist: {path}')
    if type(strict_frame_count) is not bool:
        raise ValueError('M02: strict_frame_count must be boolean')
    started = perf_counter()
    cap = None
    frames = []
    try:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise VideoProcessingError(f'M02 decode: cannot open video: {path}')
        fps_value = cap.get(cv2.CAP_PROP_FPS)
        count_value = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        fps = float(fps_value) if math.isfinite(fps_value) and fps_value > 0 else None
        reported = int(round(count_value)) if math.isfinite(count_value) and count_value > 0 else None
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            problem = _frame_error(frame)
            if problem:
                raise VideoProcessingError(f'M02 decode frame={len(frames)}: {problem}: {path}')
            if frames and frame.shape != frames[0].shape:
                raise VideoProcessingError(f'M02 decode frame={len(frames)}: shape changed: {path}')
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if not frames:
            raise VideoProcessingError(f'M02 decode: no decodable frames: {path}')
        if strict_frame_count and reported is not None and len(frames) != reported:
            raise VideoProcessingError(
                f'M02 decode: frame count mismatch, reported={reported}, decoded={len(frames)}: {path}')
    except cv2.error as error:
        raise VideoProcessingError(f'M02 decode: OpenCV error in {path}: {error}') from error
    finally:
        if cap is not None:
            cap.release()
    warnings = []
    if reported is None:
        warnings.append('Frame count unavailable; EOF cannot be distinguished from every decoder error.')
    elif reported != len(frames):
        warnings.append('Metadata frame count mismatch explicitly allowed by configuration.')
    return np.stack(frames), {
        'decoded_frames': len(frames), 'reported_frames': reported, 'fps': fps,
        'original_frame_indices': list(range(len(frames))), 'original_shape': list(frames[0].shape),
        'color_order': 'RGB', 'dtype': 'uint8', 'opencv_version': cv2.__version__,
        'strict_frame_count': strict_frame_count, 'warnings': warnings,
        'decode_seconds': perf_counter() - started}


def decode_rgb_video(path, **kwargs):
    """Compatibility name from the original M02 blueprint."""
    return decode_video(path, **kwargs)


def check_frame_quality(frame, policy=None):
    """Spatial near-flat test only; no motion/blur/skin threshold.

    Each channel's spatial std <= 1 on the [0,255] scale is the default.
    A static but textured hand/person remains valid even if repeated in time.
    """
    policy = _qc_policy(policy)
    problem = _frame_error(frame)
    if problem:
        return {'valid': False, 'reason': problem, 'channel_std': None, 'max_channel_std': None}
    std = cv2.meanStdDev(frame)[1].ravel()
    is_flat = float(std.max()) <= policy['flat_max_channel_std']
    return {'valid': not is_flat, 'reason': 'near_flat_spatial' if is_flat else None,
            'channel_std': std.tolist(), 'max_channel_std': float(std.max())}


def check_video_quality(frames, policy=None):
    """Return per-frame QC in original decode order, including rejected indices."""
    policy = _qc_policy(policy)
    checks = [{'frame_index': i, **check_frame_quality(frame, policy)}
              for i, frame in enumerate(frames)]
    valid = [r['frame_index'] for r in checks if r['valid']]
    invalid = [r['frame_index'] for r in checks if not r['valid']]
    return {'status': 'PASS' if len(valid) >= policy['min_valid_frames'] else 'FAIL',
            'policy': policy, 'total_frames': len(checks), 'valid_frames': len(valid),
            'invalid_frames': len(invalid), 'valid_frame_indices': valid,
            'invalid_frame_indices': invalid, 'valid_frame_ratio': len(valid)/len(checks) if checks else 0.0,
            'frame_checks': checks}


def remove_invalid_frames(frames, policy=None):
    """Filter in memory only; return clean RGB frames, original indices and QC."""
    quality = check_video_quality(frames, policy)
    if quality['status'] != 'PASS':
        raise VideoProcessingError(
            f"M02 QC: insufficient valid frames ({quality['valid_frames']}/{quality['total_frames']}); "
            f"minimum={quality['policy']['min_valid_frames']}; "
            f"invalid_indices={quality['invalid_frame_indices']}")
    indices = np.asarray(quality['valid_frame_indices'], dtype=np.int64)
    kept = [frames[i] for i in indices]
    if any(frame.shape != kept[0].shape for frame in kept):
        raise VideoProcessingError('M02 QC: valid frames have inconsistent shapes')
    return np.stack(kept), indices, quality


def resize_frames(frames, image_size=224):
    """Direct bilinear resize, preserving RGB uint8; no model normalization."""
    if type(image_size) is not int or image_size < 1:
        raise ValueError('M02: image_size must be a positive integer')
    if len(frames) == 0:
        raise VideoProcessingError('M02 resize: empty frame sequence')
    result = []
    for i, frame in enumerate(frames):
        problem = _frame_error(frame)
        if problem:
            raise VideoProcessingError(f'M02 resize frame={i}: {problem}')
        resized = (frame.copy() if frame.shape[:2] == (image_size, image_size) else
                   cv2.resize(frame, (image_size, image_size), interpolation=cv2.INTER_LINEAR))
        result.append(resized)
    return np.ascontiguousarray(np.stack(result), dtype=np.uint8)


def prepare_clip(path, config, video_id=None, manifest_row=None):
    """One shared deterministic clip for downstream RGB/landmark consumers.

    Output arrays are read-only. valid_mask=True marks real selected frames;
    False means repeat-last padding. Neither branch should independently resample.
    """
    started = perf_counter()
    if config.get('frames') != 16 or type(config.get('frames')) is not int:
        raise ValueError('M02: contract requires frames=16')
    if config.get('image_size') != 224 or type(config.get('image_size')) is not int:
        raise ValueError('M02: contract requires image_size=224')
    options = config.get('video_processing', {})
    if not isinstance(options, dict) or set(options) - {'qc', 'strict_frame_count'}:
        raise ValueError('M02: unsupported video_processing configuration')
    policy = _qc_policy(options.get('qc'))
    strict = options.get('strict_frame_count', True)
    if type(strict) is not bool:
        raise ValueError('M02: strict_frame_count must be boolean')
    manifest_row = {} if manifest_row is None else manifest_row
    vid = video_id if video_id is not None else manifest_row.get('video_id', Path(path).stem)
    if not isinstance(vid, str) or not vid.strip():
        raise ValueError('M02: video_id must be a nonempty string')
    if 'video_id' in manifest_row and manifest_row['video_id'] != vid:
        raise ValueError('M02: supplied video_id disagrees with manifest')
    try:
        if 'file_sha256' in manifest_row and sha256_file(path) != manifest_row['file_sha256']:
            raise VideoProcessingError('M02: file checksum disagrees with manifest')
        decoded, decode_meta = decode_video(path, strict_frame_count=strict)
        if 'frames' in manifest_row and len(decoded) != int(manifest_row['frames']):
            raise VideoProcessingError('M02: decoded frame count disagrees with manifest')
        qc_started = perf_counter()
        cleaned, original_indices, quality = remove_invalid_frames(decoded, policy)
        qc_seconds = perf_counter() - qc_started
        sampling_started = perf_counter()
        sampled, indices, valid_mask = sample_frames(cleaned, original_indices, config['frames'])
        output = resize_frames(sampled, config['image_size'])
        sampling_resize_seconds = perf_counter() - sampling_started
    except (VideoProcessingError, OSError, ValueError) as error:
        raise VideoProcessingError(f'M02 video_id={vid!r}, path={path}: {error}') from error
    resolved_options = {'frames': 16, 'image_size': 224, 'qc': policy,
                        'strict_frame_count': strict, 'padding': 'repeat_last',
                        'sampling': 'linspace_floor_on_qc_valid_frames', 'resize': 'bilinear_direct',
                        'preprocess_version': PREPROCESS_VERSION}
    quality.update(
        decode=decode_meta, preprocessing=resolved_options,
        preprocessing_sha256=hashlib.sha256(json.dumps(resolved_options, sort_keys=True).encode()).hexdigest(),
        sampled_original_indices=indices.tolist(), padding_count=int((~valid_mask).sum()),
        timings={'decode_seconds': decode_meta['decode_seconds'], 'qc_seconds': qc_seconds,
                 'sampling_resize_seconds': sampling_resize_seconds,
                 'total_seconds': perf_counter() - started})
    if decode_meta['original_shape'][:2] != [224, 224]:
        quality['decode']['warnings'].append('Direct resize may change the original aspect ratio.')
    padding_mask = ~valid_mask
    for array in [output, indices, valid_mask, padding_mask]:
        array.setflags(write=False)
    return {'frames': output, 'frame_indices': indices, 'valid_mask': valid_mask,
            'padding_mask': padding_mask, 'video_id': vid, 'quality_metadata': quality}
