"""M02: deterministic indices; used by both RGB and landmark branches."""
import numpy as np


def sample_indices(total, target_frames=16):
    if not isinstance(total, (int, np.integer)) or total <= 0:
        raise ValueError(f'M02: positive num_frames required; received {total!r}')
    if not isinstance(target_frames, (int, np.integer)) or target_frames <= 0:
        raise ValueError(f'M02: positive target_frames required; got {target_frames!r}')
    if total >= target_frames:
        index = np.linspace(0, total-1, target_frames).astype(np.int64)
    else:
        index = np.concatenate([np.arange(total, dtype=np.int64),
                                np.full(target_frames-total, total-1, dtype=np.int64)])
    assert index.shape == (target_frames,)
    return index


def assert_same_frame_indices(a, b):
    if not np.array_equal(a, b):
        raise ValueError('M05: RGB and MediaPipe use different frame indices!')


def sample_frames(frames, original_indices=None, target_frames=16):
    """Uniform sampling on QC-valid frames, with original indices and real-frame mask.

    Below target length use every valid frame once, then repeat the last frame.
    valid_mask=True means an actual selection; False means an added padding slot.
    """
    frames = np.asarray(frames)
    if frames.ndim != 4 or frames.shape[-1] != 3 or frames.dtype != np.uint8:
        raise ValueError('M02: sample_frames requires RGB uint8 [N,H,W,3]')
    if frames.shape[1] == 0 or frames.shape[2] == 0:
        raise ValueError('M02: empty frame dimensions')
    if isinstance(target_frames, (bool, np.bool_)):
        raise ValueError('M02: target_frames must be a positive integer')
    count = len(frames)
    positions = sample_indices(count, target_frames)
    original = (np.arange(count, dtype=np.int64) if original_indices is None else
                np.asarray(original_indices))
    if (original.shape != (count,) or not np.issubdtype(original.dtype, np.integer)
            or np.any(original < 0) or np.any(original[1:] <= original[:-1])):
        raise ValueError('M02: original_indices must be strictly increasing nonnegative integers matching frames')
    selected = original[positions].astype(np.int64)
    valid_mask = np.arange(target_frames) < min(count, target_frames)
    return frames[positions].copy(), selected, valid_mask
