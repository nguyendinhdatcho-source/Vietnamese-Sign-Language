"""M01: Fail-fast manifest validation and split leak checks.

Expected CSV schema is documented in CONTRACTS.md. No raw data alterations.
"""
import csv
import hashlib
import io
import json
import pickle
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath, PureWindowsPath

from .config import atomic_write_text, sha256_file

REQUIRED = ('video_id', 'split', 'relative_path', 'label_id', 'gloss', 'group_id', 'content_hash')
VALID_SPLITS = ('train', 'valid', 'test')
VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}


def safe_video_path(root, relative):
    """Portable confinement check, including Windows drive/UNC paths on Linux."""
    if not isinstance(relative, str) or not relative.strip():
        raise ValueError('M01: need relative video path')
    relative = relative.replace('\\', '/')
    path = PurePosixPath(relative)
    if path.is_absolute() or PureWindowsPath(relative).drive or '..' in path.parts:
        raise ValueError('M01: path escapes dataset root; need relative video path')
    if path.suffix.lower() not in VIDEO_EXTENSIONS:
        raise ValueError('M01: relative_path must identify a video file')
    absolute = (Path(root).resolve() / Path(path)).resolve()
    if not absolute.is_relative_to(Path(root).resolve()):
        raise ValueError('M01: path escapes dataset root')
    return path.as_posix(), absolute


def load_manifest(csv_path):
    with open(csv_path, encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or not set(REQUIRED).issubset(reader.fieldnames):
            raise ValueError(f'M01: manifest must have columns {REQUIRED}')
        return list(reader)


def validate_manifest(rows, data_root, expected_counts, classes=100, check_files=True,
                      label_mapping=None):
    """Return report, raise ValueError with contextual errors on first failed invariant."""
    root = Path(data_root).resolve()
    totals = Counter()
    label_sets = defaultdict(set)
    ids, relpaths = set(), set()
    split_by_hash, split_by_group = {}, {}
    id_to_gloss, gloss_to_id = {}, {}
    label_by_hash, label_by_group = {}, {}
    histogram = {s: Counter() for s in VALID_SPLITS}
    for n, row in enumerate(rows, start=2):
        missing = [key for key in REQUIRED if key not in row or row[key] is None]
        if missing:
            raise ValueError(f'M01 row {n}: missing fields {missing}')
        split = row['split'].strip()
        if split not in VALID_SPLITS:
            raise ValueError(f'M01 row {n}: bad split {split!r}')
        totals[split] += 1
        vid = row['video_id'].strip()
        if not vid or vid in ids:
            raise ValueError(f'M01 row {n}: missing/duplicate video_id {vid!r}')
        ids.add(vid)
        try:
            label = int(row['label_id'])
        except (ValueError, TypeError):
            raise ValueError(f'M01 row {n}: label_id must be integer')
        if not 0 <= label < classes:
            raise ValueError(f'M01 row {n}: label_id out of range: {label}')
        gloss = row['gloss'].strip()
        if not gloss:
            raise ValueError(f'M01 row {n}: gloss is empty')
        previous_gloss = id_to_gloss.setdefault(label, gloss)
        previous_id = gloss_to_id.setdefault(gloss, label)
        if previous_gloss != gloss or previous_id != label:
            raise ValueError(f'M01 row {n}: inconsistent label_id ↔ gloss mapping')
        if label_mapping is not None and label_mapping.get(gloss) != label:
            raise ValueError(f'M01 row {n}: label does not match canonical mapping: {gloss!r}/{label}')
        label_sets[split].add(label)
        histogram[split][label] += 1
        relative, absolute = safe_video_path(root, row['relative_path'].strip())
        if PurePosixPath(relative).parts[0] != split:
            raise ValueError(f'M01 row {n}: path split disagrees with split field')
        if str(absolute) in relpaths:
            raise ValueError(f'M01 row {n}: duplicate relative_path')
        relpaths.add(str(absolute))
        if check_files and not absolute.is_file():
            raise ValueError(f'M01 row {n}: missing video file: {absolute}')
        for field, index in (('group_id', split_by_group),('content_hash', split_by_hash)):
            key = row[field].strip()
            if not key:
                raise ValueError(f'M01 row {n}: missing {field}; no leakage guarantee without it')
            old = index.setdefault(key, split)
            if old != split:
                raise ValueError(f'M01 row {n}: {field} leaked across splits {old}/{split}: {key}')
            label_index = label_by_hash if field == 'content_hash' else label_by_group
            if label_index.setdefault(key, label) != label:
                raise ValueError(f'M01 row {n}: conflicting labels for {field}: {key}')
    if dict(totals) != expected_counts:
        raise ValueError(f'M01: expected {expected_counts}, got {dict(totals)}')
    for split in VALID_SPLITS:
        if label_sets[split] != set(range(classes)):
            raise ValueError(f'M01: {split} missing class IDs: {sorted(set(range(classes))-label_sets[split])}')
    return {'status':'PASS' if check_files else 'INCOMPLETE', 'total':len(rows),'counts':dict(totals),
            'distinct_classes': {s:len(label_sets[s]) for s in VALID_SPLITS},
            'distinct_hashes': len(split_by_hash),'distinct_groups':len(split_by_group),
            'checks': {'counts': True, 'all_classes_per_split': True, 'unique_video_ids': True,
                       'unique_paths': True, 'hash_split_overlap': 0, 'group_split_overlap': 0,
                       'files_checked': len(rows) if check_files else 0,
                       'missing_files': 0 if check_files else None,
                       'canonical_label_mapping': label_mapping is not None},
            'per_class': [{'label_id': label, 'gloss': id_to_gloss[label],
                           **{s: histogram[s][label] for s in VALID_SPLITS}}
                          for label in range(classes)]}


def _read_csv(path, required):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        if (not reader.fieldnames or len(set(reader.fieldnames)) != len(reader.fieldnames)
                or not set(required).issubset(reader.fieldnames)):
            raise ValueError(f'M01: {Path(path).name} missing/duplicate columns; requires {required}')
        rows = list(reader)
    if any(None in row or any(row[k] is None for k in required) for row in rows):
        raise ValueError(f'M01: malformed CSV rows in {path}')
    return rows


class _MappingReader(pickle.Unpickler):
    def find_class(self, module, name):
        raise ValueError('M01: label_mapping.pkl must contain plain data only')


def load_label_mapping(data_root, classes=100):
    root = Path(data_root)
    rows = _read_csv(root / 'label_mapping.csv', ['label', 'label_id'])
    mapping = {}
    for row in rows:
        try:
            label_id = int(row['label_id'])
        except (ValueError, TypeError):
            raise ValueError('M01: mapping label_id must be integer')
        if not row['label'] or row['label'] in mapping:
            raise ValueError('M01: empty/duplicate label in mapping')
        mapping[row['label']] = label_id
    if len(mapping) != classes or sorted(mapping.values()) != list(range(classes)):
        raise ValueError('M01: mapping must contain exactly the original contiguous class IDs')
    with (root / 'label_mapping.pkl').open('rb') as stream:
        pickled = _MappingReader(stream).load()
    if (not isinstance(pickled, dict) or any(not isinstance(k, str) for k in pickled)
            or any(type(v) is not int for v in pickled.values())):
        raise ValueError('M01: CSV/pickle label mappings disagree')
    csv_nfc = {unicodedata.normalize('NFC', k): v for k, v in mapping.items()}
    pickle_nfc = {unicodedata.normalize('NFC', k): v for k, v in pickled.items()}
    if len(csv_nfc) != len(mapping) or len(pickle_nfc) != len(pickled):
        raise ValueError('M01: Unicode normalization collision in label mapping')
    if csv_nfc != pickle_nfc:
        raise ValueError('M01: CSV/pickle label mappings disagree')
    return mapping


def adapt_handoff_csv(data_root, expected_counts, classes=100):
    """Adapt the observed three-way handoff schema; never split or relabel.

    No explicit video_id exists in these CSVs: preserve the complete filename stem
    (including _1/_2). Collisions fail rather than generating new/random IDs.
    `hash` is the decoded RGB+shape SHA-256; file_sha256 is kept separately.
    """
    root = Path(data_root)
    mapping = load_label_mapping(root, classes)
    rows = []
    source_columns = ['path', 'relative_path', 'label', 'label_id', 'group_id', 'hash',
                      'split', 'source_relative_path', 'file_sha256', 'frames', 'flat_frames']
    inputs = [root / name for name in ['label_mapping.csv', 'label_mapping.pkl', 'split_summary.json']]
    with (root / 'split_summary.json').open(encoding='utf-8') as stream:
        summary = json.load(stream)
    if (summary.get('counts') != expected_counts
            or summary.get('eligible_videos') != sum(expected_counts.values())
            or summary.get('classes_per_split') != {s: classes for s in VALID_SPLITS}):
        raise ValueError('M01: split_summary metadata disagrees with locked counts/classes')
    for split in VALID_SPLITS:
        csv_path = root / f'{split}_split.csv'
        inputs.append(csv_path)
        source_rows = _read_csv(csv_path, source_columns)
        if len(source_rows) != expected_counts[split]:
            raise ValueError(f'M01: {csv_path.name} count does not match expected split')
        for n, original in enumerate(source_rows, 2):
            context = f'{csv_path.name} row {n}'
            if original['split'] != split:
                raise ValueError(f'M01 {context}: split field disagrees with CSV')
            relative, _ = safe_video_path(root, original['relative_path'])
            path, _ = safe_video_path(root, original['path'])
            parts = PurePosixPath(relative).parts
            if path != relative or len(parts) != 3 or parts[0] != split or parts[1] != original['label']:
                raise ValueError(f'M01 {context}: path/split/label folder mismatch')
            source_relative, _ = safe_video_path(root, original['source_relative_path'])
            if source_relative != (PurePosixPath('train') / parts[1] / parts[2]).as_posix():
                raise ValueError(f'M01 {context}: original video filename/label was changed')
            for field in ['hash', 'file_sha256']:
                if not re.fullmatch('[0-9a-f]{64}', original[field]):
                    raise ValueError(f'M01 {context}: missing/invalid {field}')
            if not original['group_id'].strip():
                raise ValueError(f'M01 {context}: missing group_id; cannot infer fake groups')
            try:
                label = int(original['label_id'])
                frames, flat = int(original['frames']), int(original['flat_frames'])
            except ValueError:
                raise ValueError(f'M01 {context}: noninteger label/frame metadata')
            if mapping.get(original['label']) != label:
                raise ValueError(f'M01 {context}: canonical label mapping mismatch')
            if not 0 <= flat < frames:
                raise ValueError(f'M01 {context}: invalid/100-percent-flat video in handoff')
            rows.append(dict(video_id=original.get('video_id', PurePosixPath(relative).stem),
                             split=split, relative_path=relative, label_id=original['label_id'],
                             gloss=original['label'], group_id=original['group_id'],
                             content_hash=original['hash'], source_relative_path=source_relative,
                             file_sha256=original['file_sha256'], frames=original['frames'],
                             flat_frames=original['flat_frames']))
    evidence = {'adapter': 'olympic_three_way_handoff_v1', 'id_policy': 'explicit video_id or full original filename stem',
                'hash_policy': 'SHA-256 of decoded RGB frames, each preceded by int32 little-endian shape',
                'mapping_policy': 'Compare CSV/pickle names in NFC; preserve CSV spelling and original integer IDs; reject collisions.',
                'source_summary': summary,
                'input_sha256': {str(p.resolve()): sha256_file(p) for p in inputs}}
    return rows, mapping, evidence


def verify_video_evidence(rows, data_root, progress=None):
    """M01 audit only: stream decoded pixels to verify supplied hashes and QC.

    This is NOT the M02 decoder/preprocessor: no sampling, resizing, tensors or
    changes to videos. Match the audited notebook's RGB/shape hash algorithm.
    """
    import cv2
    import numpy as np
    root = Path(data_root)
    cv2.setNumThreads(1)
    records = []
    for i, row in enumerate(rows, 1):
        _, path = safe_video_path(root, row['relative_path'])
        if sha256_file(path) != row['file_sha256']:
            raise ValueError(f"M01 video_id={row['video_id']}: file_sha256 mismatch: {path}")
        cap = cv2.VideoCapture(str(path))
        digest = hashlib.sha256()
        count = flat = 0
        first_shape = None
        try:
            while cap.isOpened():
                ok, bgr = cap.read()
                if not ok:
                    break
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                if first_shape is None:
                    first_shape = rgb.shape
                if rgb.shape != first_shape:
                    raise ValueError(f"M01 video_id={row['video_id']}: frame shape changed")
                digest.update(np.asarray(rgb.shape, dtype='<i4').tobytes())
                digest.update(rgb.tobytes())
                count += 1
                flat += int(cv2.meanStdDev(rgb)[1].max() <= 1.0)
        finally:
            cap.release()
        if count == 0 or flat == count:
            raise ValueError(f"M01 video_id={row['video_id']}: empty/100-percent-flat video")
        if digest.hexdigest() != row['content_hash']:
            raise ValueError(f"M01 video_id={row['video_id']}: decoded content_hash mismatch")
        if count != int(row['frames']) or flat != int(row['flat_frames']):
            raise ValueError(f"M01 video_id={row['video_id']}: decoded frame metadata mismatch")
        records.append({'video_id': row['video_id'], 'split': row['split'], 'frames': count,
                        'flat_frames': flat, 'file_sha256': row['file_sha256'],
                        'content_hash': digest.hexdigest(), 'status': 'PASS'})
        if progress and (i % 500 == 0 or i == len(rows)):
            progress(f'M01 verified file + decoded content: {i}/{len(rows)}')
    return records


def verify_inventory(rows, data_root):
    root = Path(data_root)
    actual = {p.relative_to(root).as_posix() for s in VALID_SPLITS for p in (root / s).rglob('*')
              if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS}
    expected = {r['relative_path'] for r in rows}
    if actual != expected:
        raise ValueError(f'M01: inventory mismatch: missing={sorted(expected-actual)[:10]}, '
                         f'unlisted={sorted(actual-expected)[:10]}')
    return len(actual)


def write_manifest(rows, path):
    columns = list(REQUIRED) + ['source_relative_path', 'file_sha256', 'frames', 'flat_frames']
    buffer = io.StringIO(newline='')
    writer = csv.DictWriter(buffer, fieldnames=columns)
    writer.writeheader()
    writer.writerows(rows)
    atomic_write_text(path, buffer.getvalue())


def write_manifest_report(report, output_dir, video_checks=()):
    output_dir = Path(output_dir)
    atomic_write_text(output_dir / 'preflight.json', json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    for filename, data, columns in [
        ('per_class_counts.csv', report.get('per_class', []), ['label_id', 'gloss', *VALID_SPLITS]),
        ('video_checks.csv', video_checks,
         ['video_id', 'split', 'frames', 'flat_frames', 'file_sha256', 'content_hash', 'status'])]:
        buffer = io.StringIO(newline='')
        writer = csv.DictWriter(buffer, fieldnames=columns)
        writer.writeheader()
        writer.writerows(data)
        atomic_write_text(output_dir / filename, buffer.getvalue())
    return output_dir / 'preflight.json'
