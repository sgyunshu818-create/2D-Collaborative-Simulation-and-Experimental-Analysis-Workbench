"""One Pygame event loop, or a deterministic display-free mission run."""

import os
import sys
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from time import monotonic
from uuid import uuid4

from .models import BehaviorState, RunState
from .dpi_support import enable_native_dpi
from .recording import export_run
from .scene import SceneConfigError, load_scene
from .scenarios import DEFAULT_SCENE, SCENE_CHOICES, artifact_prefix, stage_name
from .simulation import Simulation


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUNS_ROOT = PROJECT_ROOT / "artifacts" / "runs"


def new_run_directory(base: Path) -> Path:
    stamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S")
    return base / f"run_{stamp}_{uuid4().hex[:6]}"


class App:
    def __init__(self, simulation: Simulation, renderer,
                 scene_path: Path | None = None, output_dir: Path | None = None,
                 capture_dir: Path | None = None, full_record: bool = False):
        self.simulation = simulation
        self.full_record = bool(full_record)
        self.renderer = renderer
        self.scene_path = scene_path
        self.output_dir = output_dir or RUNS_ROOT
        self.saved_for_run = False
        self.finish_export_attempted = False
        self.last_run_path: Path | None = None
        self.capture_dir = new_run_directory(capture_dir) if capture_dir else None
        self.capture_frames = 0
        self._next_capture_time = 0.0
        self.active = True
        self.notice = renderer.label("准备就绪 · Enter 开始", "Ready | Enter to start")

    def _sync_presentation(self):
        sync = getattr(self.renderer, 'sync_motion', None)
        if sync is not None:
            sync(self.simulation)

    def scaled_frame(self, frame_dt: float) -> float:
        """Wall time handed to the simulation, scaled by the shared time scale.

        One control serves live runs and replay alike. Scaling wall time only
        changes how long the run takes to watch: every logic step still advances
        exactly one fixed step, so the saved record is unaffected.
        """
        return frame_dt * self.renderer.playback_speed

    def save_if_needed(self) -> bool:
        if getattr(self.simulation, "is_replay", False):
            return True
        if self.saved_for_run or (self.simulation.state == RunState.READY
                                  and self.simulation.step_count == 0):
            return True
        directory = new_run_directory(self.output_dir)
        try:
            export_run(self.simulation, directory, self.scene_path)
        except OSError as exc:
            print(f"RECORDING_ERROR: {exc}", file=sys.stderr)
            return False
        self.saved_for_run = True
        self.last_run_path = directory / "run.json"
        print(f"RUN_SAVED: {directory}")
        return True

    def dispatch(self, action: str) -> bool:
        """Apply one command; return whether the timing baseline must be discarded."""
        if action in ("zoom_in", "zoom_out", "view_reset", "trails", "labels", "event_filter", "speed", "terrain", "grid",
                      "world_view", "scene_view", "focus_scene", "map_maximize"):
            self.renderer.display_action(action, self.simulation)
            map_notice = getattr(self.renderer, 'map_notice', '')
            if map_notice:
                self.notice = map_notice
            if action == "speed":
                # Discard this frame's elapsed time, as a start/resume does, so the
                # new scale never applies retroactively to time already measured.
                self.notice = self.renderer.label(
                    f"时间倍速 {self.renderer.playback_speed:g}x，实时与回放共用。",
                    f"Time scale {self.renderer.playback_speed:g}x for live runs and replay.")
                return True
            return False
        if hasattr(self.renderer, 'map_notice'):
            self.renderer.map_notice = ''
        if action == "help":
            self.notice = self.renderer.label("滚轮锚定缩放；左拖平移。全球浏览真实底图；场景返回虚拟地图；定位需地理参考。展开/恢复侧栏。T 切换倍速；列表滚轮；回放点击时间轴/事件定位。",
                "Scroll to zoom at cursor; left drag to pan. Global opens the real map; Scene shows virtual coordinates. Locate needs a geographic reference. Expand/restore panels. T cycles time scale; list scrolls; replay timeline/events seek.")
            return False
        if action == "workbench":
            try:
                subprocess.Popen([sys.executable, "-m", "sim_app.workbench"], cwd=PROJECT_ROOT)
            except OSError as exc:
                self.notice = str(exc)
            return False
        if action == "open_result":
            directory = self.last_run_path.parent if self.last_run_path else self.output_dir
            try:
                directory.mkdir(parents=True, exist_ok=True)
                if hasattr(os, "startfile"):
                    os.startfile(directory)
                else:
                    subprocess.Popen(["open" if sys.platform == "darwin" else "xdg-open", str(directory)])
            except OSError as exc:
                self.notice = str(exc)
            return False
        if action == "exit":
            if not self.save_if_needed():
                if self.simulation.state == RunState.RUNNING:
                    self.simulation.pause()
                self.notice = self.renderer.label("记录保存失败，运行已保留；修复输出目录后重试。",
                                                  "Export failed. Run retained; fix output directory and retry.")
                return True
            self.active = False
            return False
        if action == "scene":
            if self.simulation.state == RunState.RUNNING:
                return False
            if not self.save_if_needed():
                self.notice = self.renderer.label("记录保存失败，请重试。", "Export failed. Retry before switching.")
                return False
            current = self.scene_path.resolve() if self.scene_path else DEFAULT_SCENE
            index = next((i for i, path in enumerate(SCENE_CHOICES) if path == current), -1)
            path = SCENE_CHOICES[(index + 1) % len(SCENE_CHOICES)]
            try:
                scene = load_scene(path)
            except SceneConfigError as exc:
                self.notice = str(exc)
                return False
            self.simulation = Simulation(scene, record_every_step=self.full_record)
            self.scene_path = path
            self.saved_for_run = self.finish_export_attempted = False
            self.renderer.selected_unit_id = None
            self.notice = self.renderer.label("场景已切换，点击开始。", "Scene selected. Click Start.")
            self._set_caption()
            self._sync_presentation()
            return True
        if action == "sharing":
            if getattr(self.simulation, "is_replay", False) or self.simulation.scene.rules is None:
                return False
            changed = self.simulation.set_sharing(not self.simulation.sharing_enabled)
            if changed:
                self.notice = self.renderer.label("消息共享已开启。" if self.simulation.sharing_enabled else "消息共享已关闭。",
                                                  "Sharing enabled." if self.simulation.sharing_enabled else "Sharing disabled.")
            return changed
        if action == "replay":
            if self.simulation.state == RunState.RUNNING or getattr(self.simulation, "is_replay", False):
                return False
            if not self.save_if_needed() or self.last_run_path is None:
                return False
            from .replay import ReplayError, load_replay
            try:
                playback = load_replay(self.last_run_path)
            except ReplayError as exc:
                self.notice = str(exc)
                return False
            self.simulation = playback
            self.notice = self.renderer.label("已载入本轮记录，点击开始回放。", "Recording loaded. Click Start to play.")
            self._set_caption()
            self._sync_presentation()
            return True
        if action in ("previous", "next"):
            if not getattr(self.simulation, "is_replay", False):
                return False
            changed = getattr(self.simulation, action)()
            self.notice = self.renderer.label("按保存的快照逐帧查看。", "Viewing recorded snapshots.")
            self._sync_presentation()
            return changed
        if action == "reset" and not self.save_if_needed():
            if self.simulation.state == RunState.RUNNING:
                self.simulation.pause()
            self.notice = self.renderer.label("记录保存失败，当前运行未重置；请修复输出目录后重试。",
                                              "Export failed. Run retained; fix output directory and retry.")
            return True
        changed = getattr(self.simulation, action)()
        if action == "reset":
            self.saved_for_run = False
            self.finish_export_attempted = False
        if changed or action == "reset":
            self._sync_presentation()
            start_message = ("已开始，单位沿预设路线自主导航。", "Started autonomous waypoint navigation.")
            if getattr(self.simulation, "is_replay", False):
                start_message = ("已开始按记录回放。", "Playing recorded snapshots.")
            elif not self.simulation.has_missions:
                start_message = ("仿真时钟已开始，单位保持静止。", "Clock started. Units stay still.")
            messages = {
                "start": start_message,
                "pause": ("已暂停，位置与时钟保持不变。", "Paused. Positions and clock are frozen."),
                "resume": ("已继续，不补算暂停期间的时间。", "Resumed. Paused wall time is ignored."),
                "reset": ("已重置，恢复初始位置、任务与时钟。", "Reset. Positions, mission and clock restored."),
            }
            self.notice = self.renderer.label(*messages[action])
            if getattr(self.simulation, "is_replay", False) and action == "reset":
                self.notice = self.renderer.label("已回到第一份快照。", "Returned to the first snapshot.")
            return True
        return False

    def _set_caption(self) -> None:
        import pygame
        mode = "Replay" if getattr(self.simulation, "is_replay", False) else stage_name(
            self.simulation.scene, self.simulation.has_missions)
        pygame.display.set_caption(f"二维协同仿真与实验分析 | {self.simulation.scene.name} | {'回放' if getattr(self.simulation, 'is_replay', False) else '实时'}")

    def capture_frame(self) -> None:
        """Optionally save this app's rendered frames at a nominal 10 FPS."""
        if self.capture_dir is None:
            return
        now = monotonic()
        if now < self._next_capture_time:
            return
        import json
        import pygame
        try:
            self.capture_dir.mkdir(parents=True, exist_ok=True)
            pygame.image.save(self.renderer.surface, str(self.capture_dir / f"frame_{self.capture_frames:06d}.png"))
            with (self.capture_dir / "frames.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"frame": self.capture_frames, "wall_time": now,
                                         "step": self.simulation.step_count, "time": self.simulation.sim_time,
                                         "mode": "replay" if getattr(self.simulation, "is_replay", False) else "live",
                                         "state": self.simulation.state.value}) + "\n")
        except OSError as exc:
            print(f"CAPTURE_ERROR: {exc}", file=sys.stderr)
            self.capture_dir = None
            self.notice = self.renderer.label("画面采集失败并停止，仿真继续；运行记录另行保存。",
                                              "Capture stopped after an error; simulation and run recording continue.")
            return
        self.capture_frames += 1
        self._next_capture_time = now + 0.1

    def handle_event(self, event) -> bool:
        import pygame

        if self.renderer.ui_event(event, self.simulation):
            return False
        if event.type == pygame.QUIT:
            return self.dispatch("exit")
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            action = self.renderer.action_at(event.pos, self.simulation.state)
            if not action:
                action = self.renderer.extra_action_at(event.pos, self.simulation)
            if action:
                return self.dispatch(action)
            if getattr(self.simulation, "is_replay", False):
                index = self.renderer.seek_index_at(event.pos, self.simulation)
                if index is not None:
                    self.simulation.seek(index)
                    self._sync_presentation()
                    self.notice = self.renderer.label("时间轴已定位，回放暂停。", "Timeline selected. Playback paused.")
                    return True
                for rect, event_index in self.renderer.event_rows:
                    if rect.collidepoint(event.pos):
                        self.simulation.seek_event(event_index)
                        self._sync_presentation()
                        self.notice = self.renderer.label("已定位所选事件的记录快照。", "Selected event snapshot.")
                        return True
            self.renderer.select_at(event.pos, self.simulation)
        elif event.type == pygame.KEYDOWN:
            action = {pygame.K_RETURN: "start", pygame.K_KP_ENTER: "start",
                      pygame.K_r: "reset", pygame.K_ESCAPE: "exit",
                      pygame.K_s: "sharing", pygame.K_c: "scene", pygame.K_l: "replay",
                      pygame.K_g: "world_view", pygame.K_v: "scene_view",
                      pygame.K_f: "focus_scene", pygame.K_m: "map_maximize",
                      pygame.K_t: "speed",
                      pygame.K_LEFT: "previous", pygame.K_RIGHT: "next"}.get(event.key)
            if event.key == pygame.K_h:
                action = "help"
            if event.key == pygame.K_SPACE:
                action = "pause" if self.simulation.state == RunState.RUNNING else "resume"
            if action:
                return self.dispatch(action)
        return False

    def run(self) -> int:
        import pygame

        clock = pygame.time.Clock()
        while self.active:
            frame_dt = clock.tick(60) / 1000.0
            timing_changed = False
            for event in pygame.event.get():
                # Never short-circuit event handling: every queued command is processed.
                changed = self.handle_event(event)
                timing_changed = changed or timing_changed
            if not self.active:
                break
            if timing_changed:
                frame_dt = 0.0
                clock.tick()  # Clear time spent handling a start/resume/reset transition.
            self.simulation.advance(self.scaled_frame(frame_dt))
            if self.simulation.finished and not getattr(self.simulation, "is_replay", False) and not self.finish_export_attempted:
                self.finish_export_attempted = True
                saved = self.save_if_needed()
                blocked = any(u.behavior == BehaviorState.BLOCKED for u in self.simulation.units)
                self.notice = self.renderer.label(
                    "运行结束：存在不可达路径，请查看原因。" if blocked else "任务完成，所有运动单位已停止。",
                    "Run finished with blocked routes. Check reasons." if blocked else "Mission completed. All moving units stopped.")
                if self.simulation.scene.rules and not blocked:
                    self.notice = self.renderer.label("本轮结束，可重置、切换场景或回放记录。",
                                                      "Round finished. Reset, select a scene or replay the recording.")
                self.notice += self.renderer.label(" 记录已保存。", " Records saved.") if saved else self.renderer.label(
                    " 记录保存失败。", " Export failed.")
                if saved:
                    self.notice += self.renderer.label(" 工作台“历史与分析”可查看本轮结果。", " See the Workbench History tab for results.")
            self.renderer.replay_available = self.last_run_path is not None
            self.renderer.draw(self.simulation, self.notice, pygame.mouse.get_pos())
            pygame.display.flip()
            self.capture_frame()
        if self.capture_dir:
            print(f"CAPTURE_SAVED: {self.capture_dir} frames={self.capture_frames}")
        return 0 if self.save_if_needed() else 1


def _mission_result(sim: Simulation) -> bool:
    return (not sim.has_missions or sim.finished) and not any(
        u.behavior == BehaviorState.BLOCKED for u in sim.units)


def run_headless(sim: Simulation, max_steps: int, output_dir: Path, scene_path: Path) -> int:
    sim.start()
    for _ in range(max_steps):
        if sim.finished:
            break
        sim.advance(sim.scene.fixed_dt)
    directory = new_run_directory(output_dir)
    export_run(sim, directory, scene_path)
    if _mission_result(sim):
        print(f"HEADLESS_OK steps={sim.step_count} time={sim.sim_time:.6f}s state={sim.state.value} records={directory}")
        return 0
    blocked = [u.id for u in sim.units if u.behavior == BehaviorState.BLOCKED]
    print(f"HEADLESS_INCOMPLETE steps={sim.step_count} state={sim.state.value} blocked={blocked} records={directory}",
          file=sys.stderr)
    return 1


def run(scene_path: Path, smoke_test: bool = False, headless: bool = False,
        max_steps: int = 7200, output_dir: Path | None = None,
        capture_dir: Path | None = None, full_record: bool = False) -> int:
    """Load config before display initialization and return a process exit code."""
    try:
        scene = load_scene(scene_path)
    except SceneConfigError as exc:
        print(f"SCENE_ERROR: {exc}", file=sys.stderr)
        return 2

    if max_steps <= 0:
        print("ARGUMENT_ERROR: max_steps must be positive", file=sys.stderr)
        return 2
    sim = Simulation(scene, record_every_step=full_record)
    if headless:
        try:
            return run_headless(sim, max_steps, output_dir or RUNS_ROOT, scene_path)
        except (OSError, ValueError, RuntimeError) as exc:
            print(f"HEADLESS_ERROR: {exc}", file=sys.stderr)
            return 1

    enable_native_dpi()
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    if smoke_test:
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        os.environ["SDL_AUDIODRIVER"] = "dummy"
    try:
        import pygame
        from .renderer import Renderer, WINDOW_SIZE
        from .visual_assets import set_pygame_icon
    except ImportError as exc:
        print(f"DEPENDENCY_ERROR: {exc}. Install requirements.txt using the project .venv.",
              file=sys.stderr)
        return 3

    try:
        pygame.display.init()
        pygame.font.init()
        pygame.display.set_caption(f"二维协同仿真与实验分析 | {scene.name}")
        set_pygame_icon()
        surface = pygame.display.set_mode(WINDOW_SIZE)
        renderer = Renderer(surface)
        font_name = renderer.font_path or "Pygame built-in (English labels)"
        print(f"UI_FONT: {font_name}")
        if smoke_test:
            sim.start()
            preview_steps = min(300 if sim.has_missions else 30, max_steps)
            for _ in range(preview_steps):
                if sim.finished:
                    break
                sim.advance(scene.fixed_dt)
            stage = artifact_prefix(scene, sim.has_missions)
            smoke_directory = new_run_directory(output_dir or PROJECT_ROOT / 'artifacts' / 'smoke_runs')
            image = smoke_directory / f"{stage}_smoke.png"
            image.parent.mkdir(parents=True, exist_ok=True)
            renderer.draw(sim, renderer.label("无界面冒烟：运动与状态预览。", "Headless smoke: positions and state preview."))
            pygame.image.save(surface, str(image))
            if sim.has_missions:
                while not sim.finished and sim.step_count < max_steps:
                    sim.advance(scene.fixed_dt)
                renderer.draw(sim, renderer.label("无界面冒烟：完整路线运行结果。", "Headless smoke: complete mission result."))
                pygame.image.save(surface, str(image.with_name(f"{stage}_finished.png")))
            records = smoke_directory / 'recording'
            export_run(sim, records, scene_path)
            if scene.rules:
                from .replay import load_replay
                playback = load_replay(records / "run.json")
                playback.seek(playback.count - 1)
                renderer.draw(playback, renderer.label("回放终态：读取保存快照。", "Replay result: recorded snapshots."))
                pygame.image.save(surface, str(image.with_name(f"{stage}_replay.png")))
            if not _mission_result(sim):
                print(f"SMOKE_TEST_FAILED state={sim.state.value} records={records}", file=sys.stderr)
                return 1
            print(f"SMOKE_TEST_OK steps={sim.step_count} time={sim.sim_time:.6f}s state={sim.state.value} image={image} records={records}")
            return 0
        return App(sim, renderer, scene_path, output_dir, capture_dir, full_record).run()
    except (pygame.error, OSError, ValueError, RuntimeError) as exc:
        print(f"APP_ERROR: {exc}", file=sys.stderr)
        return 1
    finally:
        pygame.quit()


def run_replay(path: Path, check_only: bool = False, capture_dir: Path | None = None, seek_time: float | None = None, event_index: int | None = None) -> int:
    """Read recorded facts; no simulation update or export is performed."""
    from .replay import ReplayError, load_replay
    try:
        playback = load_replay(path)
    except ReplayError as exc:
        print(f"REPLAY_ERROR: {exc}", file=sys.stderr)
        return 2
    try:
        if seek_time is not None:
            playback.seek_time(seek_time)
        if event_index is not None:
            playback.seek_event(event_index)
    except (ValueError, TypeError) as exc:
        print(f"REPLAY_ERROR: {exc}", file=sys.stderr)
        return 2
    if check_only:
        playback.seek(playback.count - 1)
        print(f"REPLAY_CHECK_OK frames={playback.count} steps={playback.step_count} "
              f"time={playback.sim_time:.6f}s source_state={playback.source_state.value} "
              f"events={len(playback.events)}")
        return 0
    enable_native_dpi()
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    try:
        import pygame
        from .renderer import Renderer, WINDOW_SIZE
        from .visual_assets import set_pygame_icon
    except ImportError as exc:
        print(f"DEPENDENCY_ERROR: {exc}", file=sys.stderr)
        return 3
    try:
        pygame.display.init()
        pygame.font.init()
        pygame.display.set_caption("协同仿真 | 记录回放")
        set_pygame_icon()
        renderer = Renderer(pygame.display.set_mode(WINDOW_SIZE))
        app = App(playback, renderer, capture_dir=capture_dir)
        app.notice = renderer.label("已载入记录，点击开始回放。", "Recording loaded. Click Start to play.")
        return app.run()
    except (pygame.error, OSError, ValueError, RuntimeError) as exc:
        print(f"REPLAY_ERROR: {exc}", file=sys.stderr)
        return 1
    finally:
        pygame.quit()
