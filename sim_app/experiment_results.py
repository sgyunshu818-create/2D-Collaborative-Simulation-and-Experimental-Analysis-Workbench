"""Read existing image experiment evidence using only the standard library."""

from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path


class ResultLoadError(ValueError):
    pass


def discover_results(root):
    """Find result directories, excluding backups and image trees."""
    root = Path(root)
    found = []
    for parent in (root / 'artifacts' / 'marker_experiment',
                   root / 'artifacts' / 'marker_repeat5'):
        if (parent / 'predictions.csv').is_file():
            found.append(parent)
        if parent.is_dir():
            found.extend(p.parent for p in parent.glob('*/predictions.csv'))
    return sorted(set(found))


def _manifest_directory(directory, protocol):
    if (directory / 'manifest.csv').is_file():
        return directory
    expected_digest = protocol.get('manifest_sha256') if isinstance(protocol, dict) else None

    def matches(candidate):
        if not (candidate / 'manifest.csv').is_file():
            return False
        if isinstance(expected_digest, str):
            manifest = candidate / 'manifest.json'
            return (manifest.is_file() and
                    hashlib.sha256(manifest.read_bytes()).hexdigest() == expected_digest.lower())
        return True

    # Prefer the portable dataset beside the results, retaining its recorded identity.
    for parent in (directory, *directory.parents):
        candidate = parent / 'marker_experiment'
        if matches(candidate):
            return candidate
    if not isinstance(protocol, dict):
        return None
    for key in ('source_directory', 'source', 'source_dir'):
        value = protocol.get(key)
        if isinstance(value, str):
            candidate = Path(value)
            if not candidate.is_absolute():
                candidate = directory / candidate
            if matches(candidate):
                return candidate
    return None


def load_results(directory):
    """Return raw rows, exact confusion counts and sample paths without writes."""
    directory = Path(directory).resolve()
    try:
        with (directory / 'predictions.csv').open(encoding='utf-8-sig', newline='') as stream:
            rows = list(csv.DictReader(stream))
        if not rows:
            raise ResultLoadError('predictions.csv 没有预测记录。')
        required = {'image_id', 'original_id', 'condition', 'label', 'prediction', 'algorithm', 'elapsed_ns'}
        if not required.issubset(rows[0]):
            raise ResultLoadError('predictions.csv 缺少字段：' + ', '.join(sorted(required - rows[0].keys())))
        protocol_file = directory / 'protocol.json'
        protocol = json.loads(protocol_file.read_text(encoding='utf-8')) if protocol_file.is_file() else {}
        source = _manifest_directory(directory, protocol)
        manifest = {}
        if source:
            with (source / 'manifest.csv').open(encoding='utf-8-sig', newline='') as stream:
                manifest = {row['image_id']: row for row in csv.DictReader(stream)}
        groups = defaultdict(list)
        for line, row in enumerate(rows, 2):
            elapsed = int(row['elapsed_ns'])
            if elapsed < 0:
                raise ResultLoadError(f'预测第 {line} 行耗时为负。')
            row['elapsed_ms'] = elapsed / 1_000_000
            row['correct'] = int(row['label'] == row['prediction'])
            row['csv_line'] = line
            item = manifest.get(row['image_id'])
            row['image_path'] = None
            if item and source:
                path = (source / item['path']).resolve()
                if not path.is_relative_to(source.resolve()):
                    raise ResultLoadError('样本路径越出数据目录。')
                row['image_path'] = str(path)
            groups[row['condition'], row['algorithm']].append(row)
        metrics = []
        for (condition, algorithm), items in sorted(groups.items()):
            labels = sorted({r['label'] for r in items} | {r['prediction'] for r in items})
            confusion = {label: {prediction: 0 for prediction in labels} for label in labels}
            for row in items:
                confusion[row['label']][row['prediction']] += 1
            metrics.append({'condition': condition, 'algorithm': algorithm,
                'call_count': len(items), 'image_count': len({r['image_id'] for r in items}),
                'independent_originals': len({r['original_id'] for r in items}),
                'correct': sum(r['correct'] for r in items),
                'accuracy': sum(r['correct'] for r in items) / len(items),
                'mean_ms': sum(r['elapsed_ms'] for r in items) / len(items),
                'confusion': confusion})
        return {'directory': str(directory), 'predictions_path': str(directory / 'predictions.csv'),
                'protocol': protocol, 'source_directory': str(source) if source else None,
                'metrics': metrics, 'rows': rows,
                'failures': [r for r in rows if not r['correct']],
                'independent_originals': len({r['original_id'] for r in rows}),
                'image_count': len({r['image_id'] for r in rows}),
                'round_count': len({r.get('round', '1') for r in rows}),
                'timing_definition': '原始 elapsed_ns 的纯预测调用耗时；重复调用不增加独立样本。'}
    except (OSError, ValueError, KeyError, TypeError, RecursionError) as exc:
        if isinstance(exc, ResultLoadError):
            raise
        raise ResultLoadError(f'无法载入 {directory}：{exc}') from exc
