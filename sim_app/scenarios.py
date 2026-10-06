"""The three current demonstrations; historical JSON scenes remain loadable."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SCENE = PROJECT_ROOT / "configs" / "basic_scene.json"
SCENE_CHOICES = (DEFAULT_SCENE, PROJECT_ROOT / "configs" / "obstacle_scene.json",
                 PROJECT_ROOT / "configs" / "sharing_scene.json")
SCENE_LABELS = {
    "stage4_basic": ("基础场景 · 完整流程与返航", "Basic | Complete flow and return home"),
    "stage4_obstacle": ("障碍场景 · 空地运动差异", "Obstacles | Ground and air navigation"),
    "stage4_sharing": ("共享场景 · 观测来源与消息", "Sharing | Observations and message sources"),
    "stage4_blocked": ("异常示例 · 不可达路径", "Diagnostic | Unreachable route"),
}


def stage_name(scene, has_missions: bool) -> str:
    if scene.name in SCENE_LABELS:
        return "Stage 4"
    return "Stage 3" if scene.rules else "Stage 2" if has_missions else "Stage 1"


def artifact_prefix(scene, has_missions: bool) -> str:
    return scene.name if scene.name in SCENE_LABELS else stage_name(scene, has_missions).lower().replace(" ", "")
