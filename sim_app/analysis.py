"""Pure, shared analysis of recorded facts; no GUI or optional dependencies."""

import copy
import csv
import math
from pathlib import Path


DEFINITIONS = {
    "duration_s": {"unit": "秒", "definition": "末帧仿真时间；不计暂停时的现实等待时间。"},
    "unit_count": {"unit": "个", "definition": "嵌入场景中的单位数量。"},
    "completed_count": {"unit": "个", "definition": "有航点或返航任务、全部航点已到达且末状态 STOPPED；返航任务还须到达队伍返回点。"},
    "blocked_count": {"unit": "个", "definition": "末帧行为为 BLOCKED 的单位。"},
    "incomplete_count": {"unit": "个", "definition": "有任务但未完成且未受阻的单位；提前结束时 STOPPED 不自动算完成。"},
    "unassigned_count": {"unit": "个", "definition": "没有航点或返航任务且未受阻的单位，单列，不计任务完成。"},
    "distance_total": {"unit": "仿真单位", "definition": "末帧各单位 distance_travelled 之和；累计路径长度，不是首末点直线距离。"},
    "event_count": {"unit": "条", "definition": "记录 events 数量，含控制、导航和虚构规则事件。"},
    "snapshot_count": {"unit": "帧", "definition": "原始 snapshots 数量；同时间控制帧单独保留。"},
    "score": {"unit": "虚构积分", "definition": "各帧记录的 red/blue 积分；无规则的旧记录为不可用。"},
    "contact": {"unit": "条", "definition": "各帧所有单位保存的联系人总数，同一目标由不同单位记忆时分别计数；缺少联系人字段时为不可用。"},
    "recorded_speed": {"unit": "仿真单位/秒", "definition": "与前一个不同时间快照的累计距离差除以时间差，为记录区间平均速度；首个时间组未知。"},
    "comparison": {"unit": "right - left", "definition": "差值为右侧减左侧。曲线保留各自时间轴；aligned_series 仅对齐双方真实存在的同时间末控制帧，不插值或补齐。"},
}


def _number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        value = float(value)
    except (ValueError, OverflowError):
        return None
    return value if math.isfinite(value) and value >= 0 else None


def _rows(value):
    return value if isinstance(value, list) else []


def _finite_sum(values):
    if not values or any(value is None for value in values):
        return None
    try:
        total = math.fsum(values)
    except OverflowError:
        return None
    return total if math.isfinite(total) else None


def _sum_recorded(rows, field):
    values = {row.get("id", str(index)): _number(row.get(field))
              for index, row in enumerate(rows) if isinstance(row, dict)}
    total = _finite_sum(list(values.values()))
    return total, values


def _series(snapshots):
    series = {"time": [], "distance": [], "score": [], "contact": []}
    timeline = []
    for index, frame in enumerate(snapshots):
        if not isinstance(frame, dict) or _number(frame.get("time")) is None:
            continue
        base = {"snapshot_index": index, "time": frame["time"], "step": frame.get("step")}
        units = [row for row in _rows(frame.get("units")) if isinstance(row, dict)]
        distance, distances = _sum_recorded(units, "distance_travelled")
        scores = frame.get("scores")
        teams = ({team: _number(scores.get(team)) for team in ("red", "blue")}
                 if isinstance(scores, dict) else {})
        score = _finite_sum(list(teams.values()))
        contacts = {row.get("id", str(n)): len(row["contacts"]) if isinstance(row.get("contacts"), list) else None
                    for n, row in enumerate(units)}
        contact = sum(contacts.values()) if contacts and all(v is not None for v in contacts.values()) else None
        series["time"].append(dict(base))
        series["distance"].append({**base, "value": distance, "per_unit": distances})
        series["score"].append({**base, "value": score, "teams": teams})
        series["contact"].append({**base, "value": contact, "per_unit": contacts})
        timeline.append({**base, "distance": distance, "score": score, "scores": teams, "contact": contact,
                         "state": frame.get("state"), "event_count": frame.get("event_count")})
    return series, timeline


def _finish_reason(result, events):
    if result.get("finish_reason"):
        return result["finish_reason"]
    for event in reversed(events):
        if not isinstance(event, dict):
            continue
        details = event.get("details", {})
        if isinstance(details, dict) and details.get("finish_reason"):
            return details["finish_reason"]
        if event.get("kind") == "run_finished":
            return "missions_stopped"
    return {"READY": "not_started", "RUNNING": "interrupted", "PAUSED": "interrupted",
            "FINISHED": "finished_reason_unrecorded"}.get(result.get("state"), "unknown")


def summarize_run(payload) -> dict:
    """Derive metrics from terminal recorded facts, preserving unknown values."""
    if not isinstance(payload, dict):
        raise ValueError("recording payload must be a dictionary")
    scene = payload.get("scene") if isinstance(payload.get("scene"), dict) else {}
    snapshots = _rows(payload.get("snapshots"))
    events = _rows(payload.get("events"))
    result = payload.get("result")
    if not isinstance(result, dict):
        result = snapshots[-1] if snapshots and isinstance(snapshots[-1], dict) else {}
    terminal_units = [row for row in _rows(result.get("units")) if isinstance(row, dict)]
    by_id = {row.get("id"): row for row in terminal_units}
    issues = []
    if not scene:
        issues.append("缺少嵌入场景，任务归属不可确定。")
    if not snapshots:
        issues.append("缺少快照，时间曲线不可用。")
    elif payload.get("result") != snapshots[-1]:
        issues.append("result 与末快照不一致。")
    if _number(result.get("time")) is None:
        issues.append("末帧时间不可用。")
    per_unit = []
    specs = [row for row in _rows(scene.get("units")) if isinstance(row, dict)]
    for spec in specs:
        unit_id = spec.get("id")
        saved = by_id.get(unit_id, {})
        if not saved:
            issues.append(f"单位 {unit_id} 缺少末帧状态。")
        waypoints = _rows(spec.get("waypoints"))
        has_task = bool(waypoints or spec.get("return_home", False))
        reached = saved.get("waypoint_index")
        if isinstance(reached, bool) or not isinstance(reached, int) or not 0 <= reached <= len(waypoints):
            reached = None
            issues.append(f"单位 {unit_id} 的任务进度不可用。")
        blocked = saved.get("behavior") == "BLOCKED"
        at_home = True
        if spec.get("return_home", False):
            home = scene.get("return_points", {}).get(spec.get("team"), {})
            coordinates = [saved.get("x"), saved.get("y"), home.get("x"), home.get("y")]
            at_home = all(_number(v) is not None for v in coordinates) and math.hypot(
                coordinates[0] - coordinates[2], coordinates[1] - coordinates[3]) <= 1e-7
        completed = (has_task and reached == len(waypoints) and saved.get("behavior") == "STOPPED"
                     and at_home and not blocked)
        status = "blocked" if blocked else "completed" if completed else "incomplete" if has_task else "unassigned"
        per_unit.append({"id": unit_id, "team": spec.get("team"), "type": spec.get("type"),
                         "has_task": has_task, "status": status, "completed": completed, "blocked": blocked,
                         "behavior": saved.get("behavior"), "reason": saved.get("reason", ""),
                         "waypoints_total": len(waypoints), "waypoints_reached": reached,
                         "return_home": spec.get("return_home", False),
                         "distance_travelled": _number(saved.get("distance_travelled"))})
    if set(by_id) != {spec.get("id") for spec in specs}:
        issues.append("末帧单位集合与嵌入场景不一致。")
    distance, _ = _sum_recorded(terminal_units, "distance_travelled")
    if distance is None:
        issues.append("累计距离不可用。")
    series, timeline = _series(snapshots)
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    missing_metadata = [field for field in ("run_id", "app_version", "source_sha256", "config_sha256", "dependencies", "environment")
                        if not metadata.get(field)]
    if missing_metadata:
        issues.append("旧记录 / 未记录完整程序与环境元数据。")
    counts = {status: sum(row["status"] == status for row in per_unit)
              for status in ("completed", "blocked", "incomplete", "unassigned")}
    return {"state": result.get("state", "UNKNOWN"), "finish_reason": _finish_reason(result, events),
            "duration_s": _number(result.get("time")), "unit_count": len(specs),
            "completed_count": counts["completed"], "blocked_count": counts["blocked"],
            "incomplete_count": counts["incomplete"], "unassigned_count": counts["unassigned"],
            "distance_total": distance, "event_count": len(events), "snapshot_count": len(snapshots),
            "data_quality": {"status": "complete" if not issues else "partial", "issues": issues,
                             "missing_metadata": missing_metadata, "has_snapshots": bool(snapshots),
                             "run_complete": result.get("state") == "FINISHED"},
            "per_unit": per_unit, "series": series, "timeline": timeline,
            "definitions": copy.deepcopy(DEFINITIONS), "metadata": copy.deepcopy(metadata),
            "input_scene": copy.deepcopy(scene), "events": copy.deepcopy(events)}


def _input_differences(left, right, path="scene"):
    if isinstance(left, dict) and isinstance(right, dict):
        differences = []
        for key in sorted(set(left) | set(right)):
            where = f"{path}.{key}"
            if key not in left or key not in right:
                differences.append({"path": where, "left": copy.deepcopy(left.get(key)),
                                    "right": copy.deepcopy(right.get(key)), "kind": "added" if key not in left else "removed"})
            else:
                differences.extend(_input_differences(left[key], right[key], where))
        return differences
    if isinstance(left, list) and isinstance(right, list):
        if path in ("scene.units", "scene.obstacles") and all(isinstance(row, dict) and "id" in row for row in left + right):
            lhs = {row["id"]: row for row in left}
            rhs = {row["id"]: row for row in right}
            differences = []
            for unit_id in sorted(set(lhs) | set(rhs)):
                where = f"{path}[{unit_id}]"
                if unit_id not in lhs or unit_id not in rhs:
                    differences.append({"path": where, "left": copy.deepcopy(lhs.get(unit_id)),
                                        "right": copy.deepcopy(rhs.get(unit_id)), "kind": "added" if unit_id not in lhs else "removed"})
                else:
                    differences.extend(_input_differences(lhs[unit_id], rhs[unit_id], where))
            return differences
        differences = []
        for index in range(max(len(left), len(right))):
            if index >= len(left) or index >= len(right):
                differences.append({"path": f"{path}[{index}]", "left": copy.deepcopy(left[index]) if index < len(left) else None,
                                    "right": copy.deepcopy(right[index]) if index < len(right) else None,
                                    "kind": "added" if index >= len(left) else "removed"})
            else:
                differences.extend(_input_differences(left[index], right[index], f"{path}[{index}]"))
        return differences
    equal = left == right and (not isinstance(left, bool) and not isinstance(right, bool) or type(left) is type(right))
    return [] if equal else [{"path": path, "left": copy.deepcopy(left), "right": copy.deepcopy(right), "kind": "changed"}]


def compare_runs(left, right) -> dict:
    lhs, rhs = summarize_run(left), summarize_run(right)
    metrics = ("duration_s", "unit_count", "completed_count", "blocked_count", "incomplete_count", "unassigned_count",
               "distance_total", "event_count", "snapshot_count", "state", "finish_reason")
    differences = {metric: {"left": lhs[metric], "right": rhs[metric],
                           "delta": rhs[metric] - lhs[metric] if isinstance(lhs[metric], (int, float)) and isinstance(rhs[metric], (int, float)) else None}
                   for metric in metrics}
    left_times = {point["time"]: point for point in lhs["timeline"]}
    right_times = {point["time"]: point for point in rhs["timeline"]}
    aligned = []
    for time in sorted(set(left_times) & set(right_times)):
        lpoint, rpoint = left_times[time], right_times[time]
        delta = {metric: rpoint[metric] - lpoint[metric] if lpoint[metric] is not None and rpoint[metric] is not None else None
                 for metric in ("distance", "score", "contact")}
        aligned.append({"time": time, "left": dict(lpoint), "right": dict(rpoint), "delta": delta})
    return {"left": lhs, "right": rhs,
            "input_differences": _input_differences(lhs["input_scene"], rhs["input_scene"]),
            "metric_differences": differences, "series": {"left": lhs["series"], "right": rhs["series"]},
            "aligned_series": aligned, "definitions": copy.deepcopy(DEFINITIONS)}


def export_metrics(payload, path, *, source_path=None) -> Path:
    """Export exactly the summary/series metrics also used by the workbench."""
    summary = summarize_run(payload)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["section", "metric", "unit", "unit_id", "time_s", "snapshot_index",
                                                    "value", "definition", "run_id", "source_path", "config_sha256"])
        writer.writeheader()
        provenance = {"run_id": summary["metadata"].get("run_id", ""), "source_path": str(source_path or ""),
                      "config_sha256": summary["metadata"].get("config_sha256", "")}
        for metric in ("state", "finish_reason", "duration_s", "unit_count", "completed_count", "blocked_count", "incomplete_count",
                       "unassigned_count", "distance_total", "event_count", "snapshot_count"):
            definition = DEFINITIONS.get(metric, {})
            writer.writerow({**provenance, "section": "summary", "metric": metric, "value": summary[metric], **definition})
        for row in summary["per_unit"]:
            for metric in ("status", "waypoints_total", "waypoints_reached", "distance_travelled", "reason"):
                writer.writerow({**provenance, "section": "per_unit", "unit_id": row["id"], "metric": metric, "value": row[metric],
                                 "unit": "仿真单位" if metric == "distance_travelled" else ""})
        for metric in ("distance", "score", "contact"):
            definition = DEFINITIONS["distance_total" if metric == "distance" else metric]
            for point in summary["series"][metric]:
                writer.writerow({**provenance, "section": "series", "metric": metric, "time_s": point["time"],
                                 "snapshot_index": point["snapshot_index"], "value": point["value"], **definition})
    return path


def export_comparison(left, right, path, *, left_source_path=None, right_source_path=None) -> Path:
    comparison = compare_runs(left, right)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["section", "metric", "time_s", "left", "right", "delta", "unit", "definition",
                                                    "left_run_id", "right_run_id", "left_source_path", "right_source_path"])
        writer.writeheader()
        provenance = {"left_run_id": comparison["left"]["metadata"].get("run_id", ""),
                      "right_run_id": comparison["right"]["metadata"].get("run_id", ""),
                      "left_source_path": str(left_source_path or ""), "right_source_path": str(right_source_path or "")}
        for metric, values in comparison["metric_differences"].items():
            writer.writerow({**provenance, "section": "summary", "metric": metric, **values, **DEFINITIONS.get(metric, {})})
        for difference in comparison["input_differences"]:
            writer.writerow({**provenance, "section": "input", "metric": difference["path"],
                             "left": repr(difference["left"]), "right": repr(difference["right"]), "definition": difference["kind"]})
        for side in ("left", "right"):
            for metric in ("distance", "score", "contact"):
                definition = DEFINITIONS["distance_total" if metric == "distance" else metric]
                for point in comparison["series"][side][metric]:
                    writer.writerow({**provenance, "section": f"series_{side}", "metric": metric, "time_s": point["time"],
                                     side: point["value"], **definition})
        for point in comparison["aligned_series"]:
            for metric in ("distance", "score", "contact"):
                definition = DEFINITIONS["distance_total" if metric == "distance" else metric]
                writer.writerow({**provenance, "section": "aligned_series", "metric": metric, "time_s": point["time"],
                                 "left": point["left"][metric], "right": point["right"][metric], "delta": point["delta"][metric], **definition})
    return path
