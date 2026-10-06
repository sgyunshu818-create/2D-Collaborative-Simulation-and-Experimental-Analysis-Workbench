"""Rebuildable disk history with small stat-keyed entries and independent notes."""

import copy
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from .replay import load_recording

_CACHE = {}


def _recording_path(path) -> Path:
    path = Path(path)
    return path / "run.json" if path.is_dir() else path


def _note_path(path) -> Path:
    path = _recording_path(path)
    return path.with_name(path.stem + ".note.json")


def load_run(path) -> dict:
    """Read and validate the original embedded scene, events and snapshots."""
    return load_recording(_recording_path(path))


def save_note(path, note) -> Path:
    """Atomically replace only the note sidecar; never rewrite recorded facts."""
    if not isinstance(note, str):
        raise ValueError("note must be text")
    path = _recording_path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    target = _note_path(path)
    payload = {"note": note, "updated_at": datetime.now(timezone.utc).isoformat(), "recording": path.name}
    descriptor, temporary = tempfile.mkstemp(prefix=target.name + ".", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return target


def _signature(path):
    try:
        stat = path.stat()
        return stat.st_mtime_ns, stat.st_size
    except OSError:
        return None


def _entry(path):
    stat = path.stat()
    created_at = datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat()
    row = {"path": str(path.resolve()), "name": path.parent.name, "created_at": created_at,
           "scene_name": "未能载入", "status": "ERROR", "state": "ERROR", "duration": None,
           "duration_s": None, "time": created_at, "run_id": "旧记录 / 未记录", "error": None,
           "note": "", "note_error": None, "time_source": "file_mtime", "time_label": "文件修改时间（非实际开始时间）"}
    try:
        payload = load_run(path)
        result = payload["result"]
        metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
        time_source = "file_mtime"
        for field in ("started_at", "created_at", "saved_at"):
            value = metadata.get(field)
            if value is None or value == "":
                continue
            if not isinstance(value, str):
                raise ValueError(f"metadata.{field}: must be an ISO 8601 timestamp string")
            try:
                recorded_time = datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError as error:
                raise ValueError(f"metadata.{field}: invalid ISO 8601 timestamp") from error
            if recorded_time.tzinfo is None:
                raise ValueError(f"metadata.{field}: timestamp must include a timezone")
            created_at = recorded_time.astimezone(timezone.utc).isoformat()
            time_source = field
            break
        run_id = metadata.get("run_id")
        if run_id is not None and not isinstance(run_id, str):
            raise ValueError("metadata.run_id: must be a string")
        row.update(scene_name=payload["scene"]["name"], status=result["state"], state=result["state"],
                   duration=result["time"], duration_s=result["time"], created_at=created_at, time=created_at,
                   run_id=run_id or "旧记录 / 未记录", time_source=time_source,
                   time_label={"started_at": "实际开始时间", "created_at": "运行模型创建时间", "saved_at": "保存时间",
                               "file_mtime": "文件修改时间（非实际开始时间）"}[time_source])
    except (OSError, ValueError, TypeError, KeyError, RecursionError) as error:
        row["error"] = str(error)
    return _with_note(row, path)


def _with_note(row, path):
    row["note"], row["note_error"] = "", None
    note_path = _note_path(path)
    if note_path.exists():
        try:
            payload = json.loads(note_path.read_text(encoding="utf-8-sig"))
            if not isinstance(payload, dict) or not isinstance(payload.get("note"), str):
                raise ValueError("备注文件缺少文本 note 字段")
            row["note"] = payload["note"]
        except (OSError, UnicodeError, ValueError, RecursionError) as error:
            row["note_error"] = str(error)
    return row


def list_runs(root) -> list[dict]:
    """Scan run.json paths; unchanged refreshes use cached lightweight entries."""
    root = Path(root)
    if root.is_file():
        paths = [root]
    elif root.is_dir():
        paths = sorted(root.rglob("run.json"))
    else:
        return []
    rows = []
    for path in paths:
        key = str(path.resolve())
        signature = (_signature(path), _signature(_note_path(path)))
        cached = _CACHE.get(key)
        if cached is not None and cached[0] == signature:
            rows.append(copy.deepcopy(cached[1]))
            continue
        if cached is not None and cached[0][0] == signature[0]:
            row = _with_note(copy.deepcopy(cached[1]), path)
            _CACHE[key] = signature, row
            rows.append(copy.deepcopy(row))
            continue
        try:
            row = _entry(path)
        except OSError as error:
            row = {"path": key, "name": path.parent.name, "created_at": "", "scene_name": "未能载入",
                   "status": "ERROR", "state": "ERROR", "duration": None, "duration_s": None,
                   "time": "", "run_id": "旧记录 / 未记录", "error": str(error), "note": "", "note_error": None,
                   "time_source": "unavailable", "time_label": "未记录"}
        _CACHE[key] = signature, row
        rows.append(copy.deepcopy(row))
    # Drop vanished entries under this root; other imported roots can stay cached.
    current = {str(path.resolve()) for path in paths}
    for key in list(_CACHE):
        try:
            under_root = Path(key).is_relative_to(root.resolve()) if root.is_dir() else key == str(root.resolve())
        except (ValueError, OSError):
            under_root = False
        if under_root and key not in current:
            del _CACHE[key]
    return sorted(rows, key=lambda row: (row["created_at"], row["path"]), reverse=True)
