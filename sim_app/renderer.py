"""Read-only Pygame presentation of navigation, trails and run events."""

from dataclasses import dataclass
from collections import OrderedDict
from pathlib import Path
from time import monotonic

import pygame

from .models import BehaviorState, Point, RunState, Team, UnitType
from .simulation import Simulation
from .visual_assets import app_icon_surface, terrain_view_surface, unit_glyph_surface
from .camera import MapCamera
from .font_support import bundled_font_path, pygame_font
from .motion import MotionPresenter
from .geography import scene_world, GeoReference, format_lonlat
from .world_map import WorldMapView
from .equipment import profile_key, equipment_label


WINDOW_SIZE = (1280, 800)
MAP_UNIT_SIZE = 56
MAP_AIR_SIZE = 52
BG = (6, 17, 29)
PANEL = (11, 29, 43)
PANEL_HEADER = (16, 40, 58)
BRAND = (7, 23, 37)
BRAND_LIGHT = (221, 236, 244)
BRAND_MUTED = (143, 170, 188)
EDGE = (29, 80, 101)
TEXT = (221, 236, 244)
MUTED = (143, 170, 188)
ACCENT = (36, 200, 229)
ACCENT_HOVER = (89, 220, 241)
SELECTED = (18, 61, 80)
HOVER = (21, 56, 75)
CONTROL_DISABLED = (10, 27, 39)
DISABLED_TEXT = (93, 119, 137)
MAP = (8, 24, 36)
GRID = (22, 51, 68)
MAP_EDGE = (37, 96, 117)
WARNING = (240, 183, 86)
TEAM_COLORS = {Team.RED: (241, 108, 128), Team.BLUE: (82, 199, 243)}
TEAM_TEXT = TEAM_COLORS


@dataclass(frozen=True)
class Button:
    action: str
    rect: pygame.Rect
    zh: str
    en: str
    shortcut: str


class Renderer:
    """Present live state or recorded snapshots without updating the world."""

    def __init__(self, surface: pygame.Surface):
        self.surface = surface
        font_path = bundled_font_path()
        self.font_path = str(font_path) if font_path else None
        self.chinese = font_path is not None
        self.fonts = {
            size: pygame_font(size)
            for size in (12, 13, 14, 15, 16, 17, 18, 20, 21, 24, 26, 28, 34)
        }
        self.title_fonts = {size: pygame_font(size, bold=True) for size in self.fonts}
        self._text_cache = OrderedDict()
        self.map_rect = pygame.Rect(256, 190, 704, 507)
        self.replay_available = False
        self.selected_unit_id: str | None = None
        self.unit_offset = self.contact_offset = self.event_offset = 0
        self.unit_search = ""
        self.search_active = False
        self._search_pending_text = ""
        self.camera = MapCamera()
        self.world_map = None
        self.map_mode = 'scene'
        self.map_maximized = False
        self.map_notice = ''
        self._map_scene = None
        self._clock = monotonic
        self.motion = MotionPresenter()
        self._display_positions = {}
        self._drawn_model = None
        self._drag_button = self._drag_last = None
        self._drag_origin = None
        self._in_world = False
        self._world_transform = None
        self.show_trails = self.show_labels = True
        self.show_terrain = True
        self.show_grid = False
        self.filter_events = False
        self.playback_speed = 1.0
        self.unit_rows = []
        self.event_rows = []
        self.unit_hit_positions = {}
        self.unit_hit_rects = {}
        self.unit_label_rects = {}
        self.hidden_unit_labels = set()
        self.search_rect = pygame.Rect(32, 263, 176, 28)
        self.unit_scroll_rect = pygame.Rect(16, 294, 208, 170)
        self.contact_scroll_rect = pygame.Rect(992, 439, 272, 264)
        self.event_scroll_rect = pygame.Rect(0, 0, 0, 0)
        self.map_buttons = tuple(Button(action, pygame.Rect(400 + i * 78, 121, 72, 28), zh, en, "")
            for i, (action, zh, en) in enumerate((
                ("zoom_in", "放大 +", "Zoom +"), ("zoom_out", "缩小 -", "Zoom -"),
                ("view_reset", "复位", "Reset view"), ("trails", "轨迹", "Trails"),
                ("labels", "编号", "Labels"), ("event_filter", "单位事件", "Unit events"),
                ("speed", "倍速", "Speed"))))
        self.map_layer_buttons = (
            Button("terrain", pygame.Rect(256, 157, 80, 27), "地形", "Terrain", ""),
            Button("grid", pygame.Rect(344, 157, 80, 27), "网格", "Grid", ""))
        self.map_navigation_buttons = (
            Button("world_view", pygame.Rect(432, 157, 70, 27), "全球", "Global", ""),
            Button("scene_view", pygame.Rect(510, 157, 70, 27), "场景", "Scene", ""),
            Button("focus_scene", pygame.Rect(588, 157, 100, 27), "定位场景", "Locate", ""),
            Button("map_maximize", pygame.Rect(696, 157, 104, 27), "展开地图", "Expand", ""))
        self._normal_map_buttons = self.map_buttons
        self._normal_layer_buttons = self.map_layer_buttons
        self._normal_navigation_buttons = self.map_navigation_buttons
        self.footer_buttons = (
            Button("workbench", pygame.Rect(650, 726, 180, 36), "工作台 / 历史", "Workbench / history", ""),
            Button("open_result", pygame.Rect(842, 726, 172, 36), "打开结果目录", "Open results", ""),
            Button("help", pygame.Rect(1026, 726, 230, 36), "操作帮助", "Help", "H"))
        self.extra_buttons = (
            Button("scene", pygame.Rect(720, 62, 144, 34), "切换场景", "Scene", "C"),
            Button("sharing", pygame.Rect(876, 62, 164, 34), "共享", "Sharing", "S"),
            Button("replay", pygame.Rect(1052, 62, 204, 34), "回放本轮", "Replay run", "L"),
            Button("previous", pygame.Rect(818, 656, 66, 34), "上一帧", "Prev", "←"),
            Button("next", pygame.Rect(892, 656, 66, 34), "下一帧", "Next", "→"),
        )
        self.timeline = pygame.Rect(274, 664, 530, 18)
        self._normal_extra_buttons = self.extra_buttons
        self.buttons = tuple(
            Button(action, pygame.Rect(24 + i * 124, 726, 112, 36), zh, en, key)
            for i, (action, zh, en, key) in enumerate(
                (
                    ("start", "开始", "Start", "Enter"),
                    ("pause", "暂停", "Pause", "Space"),
                    ("resume", "继续", "Resume", "Space"),
                    ("reset", "重置", "Reset", "R"),
                    ("exit", "退出", "Exit", "Esc"),
                )
            )
        )

    def label(self, zh: str, en: str) -> str:
        return zh if self.chinese else en

    @property
    def zoom(self):
        return self.active_camera.zoom

    @zoom.setter
    def zoom(self, value):
        self.active_camera.set_pose(value, self.active_camera.center)

    @property
    def view_center(self):
        return self.active_camera.center

    @view_center.setter
    def view_center(self, value):
        self.active_camera.set_pose(self.active_camera.zoom, value)

    @property
    def active_camera(self):
        return self._world_view().camera if self.map_mode == 'world' else self.camera

    def _world_view(self):
        if self.world_map is None:
            self.world_map = WorldMapView()
        if self.world_map.size != self.map_rect.size:
            self.world_map.resize(self.map_rect.size)
        return self.world_map

    def _prepare_map(self, sim):
        """Keep the two display cameras independent of the simulation."""
        changed = self._map_scene is not sim.scene
        if changed:
            self._map_scene = sim.scene
            self.map_notice = ''
            self.map_mode = 'world' if GeoReference.from_world(scene_world(sim.scene)) else 'scene'
            self._layout_map(sim)
            if self.map_mode == 'world':
                self._world_view().focus_scene(scene_world(sim.scene), animate=False)
        self._layout_map(sim)
        self._bind_camera(sim)
        if changed and getattr(sim.scene,'terrain',None)=='astra_atlas':
            from .terrain_atlas import DETAIL_ZOOM
            self.camera.set_pose(DETAIL_ZOOM,Point(sim.scene.width/2,sim.scene.height/2))

    @staticmethod
    def _move_buttons(buttons, start_x, y, gap=6):
        moved = []
        for button in buttons:
            rect = button.rect.copy()
            rect.topleft = (start_x, y)
            moved.append(Button(button.action, rect, button.zh, button.en, button.shortcut))
            start_x = rect.right + gap
        return tuple(moved)

    def _layout_map(self, sim):
        replay = getattr(sim, 'is_replay', False)
        if self.map_maximized:
            self.map_rect = pygame.Rect(32, 142, 1216, 524 if replay else 574)
            self.map_navigation_buttons = self._move_buttons(self._normal_navigation_buttons, 32, 109)
            self.map_buttons = self._move_buttons(self._normal_map_buttons, 400, 109)
            self.map_layer_buttons = self._move_buttons(self._normal_layer_buttons, 952, 109)
            self.timeline = pygame.Rect(48, 679, 1036, 18)
            self.extra_buttons = (*self._normal_extra_buttons[:3],
                *self._move_buttons(self._normal_extra_buttons[3:], 1100, 672))
            self.search_active = False
            self.unit_rows = []
        else:
            self.map_rect = pygame.Rect(256, 190, 704, 460 if sim.has_missions or replay else 507)
            self.map_buttons = self._normal_map_buttons
            self.map_layer_buttons = self._normal_layer_buttons
            self.map_navigation_buttons = self._normal_navigation_buttons
            self.timeline = pygame.Rect(274, 664, 530, 18)
            self.extra_buttons = self._normal_extra_buttons
        if self.map_mode == 'world':
            self._world_view()

    def _bind_camera(self, sim):
        terrain=getattr(sim.scene,'terrain',None)
        self.camera.max_zoom = 256 if terrain=='astra_atlas' else 64 if terrain in ('detailed_virtual','astra_mountain') else 6
        if self.camera.zoom>self.camera.max_zoom:
            self.camera.set_pose(self.camera.max_zoom,self.camera.center)
        dimensions = (sim.scene.width, sim.scene.height)
        viewport = tuple(self.map_rect)
        if self.camera.world_size != dimensions or self.camera.viewport != viewport:
            self.camera.bind(dimensions, viewport)

    def text(self, value: str, pos: tuple[int, int], size: int = 16,
             color: tuple[int, int, int] = TEXT) -> pygame.Rect:
        rendered = self._render_text(value, size, color)
        return self.surface.blit(rendered, pos)

    def _render_text(self, value, size, color, *, bold=False, max_width=None):
        # Stable chrome and identifiers need no repeated font rasterization.
        # Bound the cache so changing time/coordinates cannot grow it forever.
        key = (value, size, tuple(color), bold, max_width)
        rendered = self._text_cache.get(key)
        if rendered is not None:
            self._text_cache.move_to_end(key)
            return rendered
        font = self.title_fonts[size] if bold else self.fonts[size]
        if max_width is not None and font.size(value)[0] > max_width:
            choices = self.title_fonts if bold else self.fonts
            for native_size in sorted((s for s in choices if s <= size), reverse=True):
                font = choices[native_size]
                if font.size(value)[0] <= max_width:
                    break
            if font.size(value)[0] > max_width:
                # Keep glyphs on their native pixel grid instead of shrinking
                # already-rasterized Chinese strokes into a blurred bitmap.
                left, right = 0, len(value)
                while left < right:
                    middle = (left + right + 1) // 2
                    if font.size(value[:middle] + '…')[0] <= max_width:
                        left = middle
                    else:
                        right = middle - 1
                value = value[:left] + '…' if left else ''
        rendered = font.render(value, True, color)
        if len(self._text_cache) >= 512:
            self._text_cache.popitem(last=False)
        self._text_cache[key] = rendered
        return rendered

    def fitted_text(self, value: str, pos: tuple[int, int], max_width: int,
                    size: int = 16, color: tuple[int, int, int] = TEXT) -> None:
        # Prefer native glyph sizes; unusually long values use an ellipsis.
        rendered = self._render_text(value, size, color, max_width=max_width)
        self.surface.blit(rendered, pos)

    def _draw_button(self, button: Button, title: str, mouse_pos=(-1, -1),
                     *, enabled: bool = True, primary: bool = False,
                     selected: bool = False, size: int = 13) -> None:
        hovered = enabled and button.rect.collidepoint(mouse_pos)
        if not enabled:
            fill, color, edge = CONTROL_DISABLED, DISABLED_TEXT, EDGE
        elif primary:
            fill, color, edge = ACCENT_HOVER if hovered else ACCENT, BG, ACCENT
        elif selected:
            fill, color, edge = HOVER if hovered else SELECTED, ACCENT, ACCENT
        else:
            fill, color, edge = HOVER if hovered else PANEL_HEADER, TEXT, EDGE
        pygame.draw.rect(self.surface, fill, button.rect, border_radius=5)
        pygame.draw.rect(self.surface, edge, button.rect, 1, border_radius=5)
        available = button.rect.width - 12
        rendered = self._render_text(title, size, color, max_width=available)
        self.surface.blit(rendered, rendered.get_rect(center=button.rect.center))

    def _draw_panel(self, rect: pygame.Rect) -> None:
        pygame.draw.rect(self.surface, PANEL, rect, border_radius=6)
        pygame.draw.rect(self.surface, EDGE, rect, 1, border_radius=6)
        header = pygame.Rect(rect.x + 1, rect.y + 1, rect.width - 2, 39)
        pygame.draw.rect(self.surface, PANEL_HEADER, header, border_top_left_radius=5,
                         border_top_right_radius=5)
        pygame.draw.line(self.surface, EDGE, (rect.x + 1, header.bottom),
                         (rect.right - 2, header.bottom))
        pygame.draw.line(self.surface, ACCENT, (rect.x + 10, rect.y),
                         (rect.x + 38, rect.y))

    def _title(self, value, pos, size=17, color=TEXT):
        return self.surface.blit(self._render_text(value, size, color, bold=True), pos)

    def _draw_header(self, sim, scene_name, replay):
        pygame.draw.rect(self.surface, BRAND, (0, 0, WINDOW_SIZE[0], 48))
        pygame.draw.line(self.surface, EDGE, (0, 47), (1279, 47))
        self.surface.blit(app_icon_surface(32), (18, 8))
        self._title(self.label('协同仿真', 'SIMULATION'), (62, 11), 18, BRAND_LIGHT)
        pygame.draw.line(self.surface, (59, 77, 100), (217, 14), (217, 34))
        self.fitted_text(scene_name, (237, 15), 654, 14, BRAND_LIGHT)
        mode = self.label('回放模式' if replay else '实时模式', 'REPLAY' if replay else 'LIVE')
        self.fitted_text(mode, (940, 15), 100, 13, BRAND_MUTED)
        pygame.draw.line(self.surface, (59, 77, 100), (1060, 14), (1060, 34))
        status_color = {RunState.RUNNING: (93, 199, 163), RunState.PAUSED: (234, 188, 99)}.get(
            sim.state, (84, 195, 217))
        pygame.draw.circle(self.surface, status_color, (1093, 24), 3)
        self.fitted_text(self.state_label(sim.state), (1107, 14), 140, 14, BRAND_LIGHT)

    def world_to_screen(self, point: Point, sim: Simulation) -> tuple[int, int] | None:
        if self.map_mode == 'world':
            local = self._world_view().local_to_screen(point.x, point.y, scene_world(sim.scene))
            return None if local is None else (round(local[0] + self.map_rect.x),
                                               round(local[1] + self.map_rect.y))
        if self._in_world and self._drawn_model is sim:
            scale, origin_x, origin_y = self._world_transform
        else:
            self._bind_camera(sim)
            scale = self.camera.base_scale * self.zoom
            center = self.camera.actual_center
            cx, cy = self.camera.screen_center
            origin_x, origin_y = cx - center.x * scale, cy - center.y * scale
        return (round(origin_x + point.x * scale), round(origin_y + point.y * scale))

    def screen_to_world(self, pos, sim):
        if self.map_mode == 'world':
            local = self._world_view().screen_to_local(
                (pos[0] - self.map_rect.x, pos[1] - self.map_rect.y), scene_world(sim.scene))
            return Point(*local) if local is not None else None
        self._bind_camera(sim)
        return self.camera.to_world(pos)

    def sync_motion(self, sim):
        self.motion.sync(sim)

    @staticmethod
    def enabled(action: str, state: RunState) -> bool:
        return {
            "start": state == RunState.READY,
            "pause": state == RunState.RUNNING,
            "resume": state == RunState.PAUSED,
            "reset": True,
            "exit": True,
        }[action]

    def action_at(self, pos: tuple[int, int], state: RunState) -> str | None:
        for button in self.buttons:
            if button.rect.collidepoint(pos) and self.enabled(button.action, state):
                return button.action
        return None

    def extra_enabled(self, action: str, sim) -> bool:
        replay = getattr(sim, "is_replay", False)
        return {
            "scene": sim.state != RunState.RUNNING,
            "sharing": bool(sim.scene.rules) and not replay and sim.state != RunState.FINISHED,
            "replay": not replay and sim.state != RunState.RUNNING and self.replay_available,
            "previous": replay and sim.index > 0 if replay else False,
            "next": replay and sim.index < sim.count - 1 if replay else False,
        }[action]

    def extra_action_at(self, pos, sim) -> str | None:
        for button in (*self.map_buttons, *self.map_layer_buttons,
                       *self.map_navigation_buttons, *self.footer_buttons):
            if button.rect.collidepoint(pos):
                if button.action == 'zoom_in' and self.active_camera.target_zoom >= self.active_camera.max_zoom:
                    return None
                if button.action == 'zoom_out' and self.active_camera.target_zoom <= self.active_camera.min_zoom:
                    return None
                return button.action
        for button in self.extra_buttons:
            if button.rect.collidepoint(pos) and self.extra_enabled(button.action, sim):
                return button.action
        return None

    def seek_index_at(self, pos, replay) -> int | None:
        if not self.timeline.inflate(0, 14).collidepoint(pos):
            return None
        ratio = min(1.0, max(0.0, (pos[0] - self.timeline.x) / (self.timeline.width - 1)))
        return round(ratio * (replay.count - 1))

    def select_at(self, pos, sim) -> bool:
        for rect, unit_id in self.unit_rows if not self.map_maximized else ():
            if rect.collidepoint(pos):
                self.selected_unit_id = unit_id
                self.contact_offset = self.event_offset = 0
                return True
        if not self.map_rect.collidepoint(pos):
            return False
        if self.map_mode == 'world':
            unit_id = self._world_view().pick_unit((pos[0] - self.map_rect.x, pos[1] - self.map_rect.y))
            if unit_id is not None:
                self.selected_unit_id = unit_id
                self.contact_offset = self.event_offset = 0
                return True
            return False
        candidates = []
        for unit in sim.units:
            positions = self.unit_hit_positions if self._drawn_model is sim else {}
            x, y = positions.get(unit.id, self.world_to_screen(unit.position, sim))
            distance = (x - pos[0]) ** 2 + (y - pos[1]) ** 2
            size = self._map_unit_size(unit.unit_type)
            hit_rect = self.unit_hit_rects.get(unit.id, pygame.Rect(x - size // 2, y - size // 2, size, size))
            if hit_rect.collidepoint(pos):
                candidates.append((distance, unit.id))
        if candidates:
            self.selected_unit_id = min(candidates, key=lambda value: value[0])[1]
            self.contact_offset = self.event_offset = 0
            return True
        return False

    def draw(self, sim: Simulation, notice: str = "", mouse_pos=(-1, -1)) -> None:
        self.surface.fill(BG)
        replay = getattr(sim, "is_replay", False)
        game = sim.scene.rules is not None
        from .scenarios import SCENE_LABELS
        business = SCENE_LABELS.get(sim.scene.name)
        self._prepare_map(sim)
        if self.map_mode == 'world':
            self._world_view().update()
        else:
            self.camera.update(self._clock())
        self._display_positions = self.motion.positions(sim)
        self._drawn_model = sim
        self.event_scroll_rect = pygame.Rect(0, 0, 0, 0)
        self.event_rows = []
        if self.selected_unit_id not in {unit.id for unit in sim.units}:
            self.selected_unit_id = sim.units[0].id if sim.units else None
        self._draw_header(sim, self.label(business[0] if business else sim.scene.name,
                                        business[1] if business else sim.scene.name), replay)
        self._title(self.label('仿真态势', 'Simulation view'), (24, 67))
        summary = self.label(f'{len(sim.units)} 个单位 · {len(sim.scene.obstacles)} 个障碍 · 步长 {sim.scene.fixed_dt:g}s',
                             f'{len(sim.units)} units | {len(sim.scene.obstacles)} obstacles | dt {sim.scene.fixed_dt:g}s')
        self.fitted_text(summary,
                         (211, 71), 480, 13, MUTED)
        self._draw_toolbar(sim, mouse_pos)

        if self.map_maximized:
            pygame.draw.rect(self.surface, PANEL, (16, 103, 1248, 613), border_radius=6)
            pygame.draw.rect(self.surface, EDGE, (16, 103, 1248, 613), 1, border_radius=6)
        else:
            self._draw_panel(pygame.Rect(240, 112, 736, 604))
            self._title(self.label('世界地图' if self.map_mode == 'world' else '场景地图',
                                   'World map' if self.map_mode == 'world' else 'Scene map'), (256, 123))
        for button in self.map_buttons:
            title = self.label(button.zh, button.en)
            if button.action == "speed":
                title = f"{self.playback_speed:g}x"
            if button.action == "trails":
                title = self.label("轨迹开" if self.show_trails else "轨迹关", "Trails on" if self.show_trails else "Trails off")
            if button.action == "labels":
                title = self.label("编号开" if self.show_labels else "编号关", "IDs on" if self.show_labels else "IDs off")
            if button.action == "event_filter" and self.filter_events:
                title = self.label("筛选开", "Filtered")
            selected = {"trails": self.show_trails, "labels": self.show_labels,
                        "event_filter": self.filter_events}.get(button.action, False)
            enabled = (self.active_camera.target_zoom < self.active_camera.max_zoom if button.action == 'zoom_in' else
                       self.active_camera.target_zoom > self.active_camera.min_zoom if button.action == 'zoom_out' else True)
            self._draw_button(button, title, mouse_pos, selected=selected, size=12, enabled=enabled)
        for button in self.map_layer_buttons:
            selected = self.show_terrain if button.action == "terrain" else self.show_grid
            title = self.label(button.zh + ("开" if selected else "关"),
                               button.en + (" on" if selected else " off"))
            self._draw_button(button, title, mouse_pos, selected=selected, size=12)
        for button in self.map_navigation_buttons:
            title = self.label(button.zh, button.en)
            if button.action=='focus_scene' and getattr(sim.scene,'terrain',None)=='astra_atlas':
                title=self.label('定位装备','Locate unit')
            if button.action == 'map_maximize' and self.map_maximized:
                title = self.label('恢复侧栏', 'Restore panels')
            selected = ((button.action == 'world_view' and self.map_mode == 'world') or
                        (button.action == 'scene_view' and self.map_mode == 'scene'))
            self._draw_button(button, title, mouse_pos, selected=selected, size=12)
        self.fitted_text(f'{self.zoom * 100:.0f}%',
                         (1136, 115) if self.map_maximized else (874, 161), 112 if self.map_maximized else 78,
                         13, ACCENT)
        if self.map_mode == 'world':
            self._draw_geographic_world(sim, mouse_pos)
        else:
            self._draw_world(sim)
            self._draw_map_readout(sim, mouse_pos)
        if replay:
            self._draw_timeline(sim, mouse_pos)
        elif game and not self.map_maximized:
            selected = next((u for u in sim.units if u.id == self.selected_unit_id), None)
            if selected:
                shared = sum(contact.shared for contact in selected.contacts.values())
                self.fitted_text(self.label(
                    f"选中 {selected.id} · 感知范围 {selected.sensor_range:g} · 已知 {len(selected.contacts)}（共享 {shared}）",
                    f"Selected {selected.id} | Range {selected.sensor_range:g} | Contacts {len(selected.contacts)} ({shared} shared)"),
                    (256, 688), 704, 12, MUTED)
        if not self.map_maximized:
            self._draw_status(sim)
            self._draw_events(sim, mouse_pos)
        pygame.draw.rect(self.surface, PANEL, (0, 721, 1280, 47))
        pygame.draw.line(self.surface, EDGE, (0, 721), (1279, 721))
        pygame.draw.line(self.surface, EDGE, (635, 731), (635, 757))
        self._draw_controls(sim, mouse_pos)
        for button in self.footer_buttons:
            title = self.label(button.zh, button.en)
            if button.shortcut:
                title += f"  {button.shortcut}"
            self._draw_button(button, title, mouse_pos)
        pygame.draw.line(self.surface, EDGE, (24, 768), (1256, 768))
        status = self.map_notice or notice or self.label("地图滚轮锚定缩放 · 左键拖动平移 · 全球 / 场景 / 定位 · 列表滚轮 · H 帮助",
                                      "Scroll map to zoom at cursor | Left drag to pan | Reset view | Scroll lists | H help")
        self.fitted_text(status, (26, 774), 1228, 12, MUTED)

    def _geographic_world(self, sim):
        world = scene_world(sim.scene)
        world['obstacles'] = [dict(id=o.id, x=o.x, y=o.y, width=o.width, height=o.height)
                              for o in sim.scene.obstacles]
        world['sites'] = []
        for team in Team:
            name = self.label('红方' if team == Team.RED else '蓝方', team.value.upper())
            spawn, home = sim.scene.spawn_points[team], sim.scene.return_points[team]
            if spawn == home:
                world['sites'].append(dict(label=self.label(f'{name} 出发 / 返航', f'{name} spawn / home'),
                                           team=team.value, x=spawn.x, y=spawn.y))
            else:
                for point, title in ((spawn, self.label(f'{name} 出发', f'{name} spawn')),
                                     (home, self.label(f'{name} 返航', f'{name} home'))):
                    world['sites'].append(dict(label=title, team=team.value, x=point.x, y=point.y))
        return world

    def _geographic_units(self, sim):
        rows = []
        for unit in sim.units:
            position = self._display_positions.get(unit.id, unit.position)
            trail = unit.trail if position == unit.position else (
                [*unit.trail, position] if getattr(sim, 'is_replay', False)
                else [*unit.trail[:-1], position])
            rows.append(dict(id=unit.id, x=position.x, y=position.y,
                             type=unit.unit_type.value, equipment=profile_key(unit), team=unit.team.value,
                             sensor_range=unit.sensor_range,
                             waypoints=[dict(x=p.x, y=p.y) for p in unit.waypoints],
                             path=[dict(x=p.x, y=p.y) for p in unit.path[unit.path_index:]],
                             trail=[dict(x=p.x, y=p.y) for p in trail]))
        return rows

    def _draw_geographic_world(self, sim, mouse_pos):
        view = self._world_view()
        world = self._geographic_world(sim)
        title = sim.scene.name
        frame = view.render(world, self._geographic_units(sim), selected_id=self.selected_unit_id,
                            show_grid=self.show_grid, show_terrain=self.show_terrain,
                            show_trails=self.show_trails, show_labels=self.show_labels, title=title)
        self.surface.blit(frame, self.map_rect)
        self.unit_hit_rects = {unit_id: rect.move(self.map_rect.x, self.map_rect.y)
                               for unit_id, rect in view.hit_rects.items()}
        self.unit_hit_positions = {unit_id: rect.center for unit_id, rect in self.unit_hit_rects.items()}
        self.unit_label_rects.clear()
        self.hidden_unit_labels = {unit.id for unit in sim.units} - self.unit_hit_rects.keys()
        self._draw_map_readout(sim, mouse_pos)

    def _draw_map_readout(self, sim, mouse_pos):
        if not self.map_rect.collidepoint(mouse_pos):
            if self.map_mode == 'world':
                return
            terrain_label = {'astra_atlas':'广域山区地图', 'astra_mountain': '山区影像地形', 'detailed_virtual': '精细分层地形'}.get(getattr(sim.scene, 'terrain', None), '示意地形')
            value = self.label('虚拟场景 · X/Y 仿真单位 · ' + terrain_label,
                               'Virtual scene | simulation X/Y | illustrative terrain')
        elif self.map_mode == 'world':
            lon, lat = self._world_view().to_lonlat((mouse_pos[0]-self.map_rect.x,
                                                    mouse_pos[1]-self.map_rect.y))
            if not -180 <= lon <= 180 or not -90 <= lat <= 90:
                return
            value = self.label('地图 ', 'Map ') + format_lonlat(lon, lat)
            local = self.screen_to_world(mouse_pos, sim)
            if local is not None and 0 <= local.x <= sim.scene.width and 0 <= local.y <= sim.scene.height:
                value += f'  ·  X {local.x:.1f} / Y {local.y:.1f}'
        else:
            point = self.screen_to_world(mouse_pos, sim)
            value = self.label('仿真 ', 'Simulation ') + f'X {point.x:.1f} / Y {point.y:.1f}'
        text = self._render_text(value, 12, MUTED, max_width=max(1, self.map_rect.width-24))
        rect = text.get_rect(topright=(self.map_rect.right-10,
                                       self.map_rect.y+(56 if self.map_mode == 'world' else 10)))
        old_clip = self.surface.get_clip()
        self.surface.set_clip(self.map_rect)
        pygame.draw.rect(self.surface, MAP, rect.inflate(10, 6), border_radius=3)
        self.surface.blit(text, rect)
        self.surface.set_clip(old_clip)

    def _draw_world(self, sim: Simulation) -> None:
        scale = self.camera.base_scale * self.zoom
        center = self.camera.actual_center
        cx, cy = self.camera.screen_center
        self._world_transform = (scale, cx - center.x * scale, cy - center.y * scale)
        self._in_world = True
        self.unit_hit_positions.clear()
        self.unit_hit_rects.clear()
        self.unit_label_rects.clear()
        self.hidden_unit_labels = {unit.id for unit in sim.units}
        # Aspect ratio is preserved even if a user chooses another world size.
        tl = self.world_to_screen(Point(0, 0), sim)
        br = self.world_to_screen(Point(sim.scene.width, sim.scene.height), sim)
        world_rect = pygame.Rect(tl, (br[0] - tl[0], br[1] - tl[1]))
        old_clip = self.surface.get_clip()
        self.surface.set_clip(self.map_rect)
        pygame.draw.rect(self.surface, MAP, world_rect)
        if self.show_terrain:
            terrain_source = terrain_view_surface
            if getattr(sim.scene,'terrain',None)=='astra_atlas':
                from .terrain_atlas import terrain_view_surface as terrain_source
            elif getattr(sim.scene, 'terrain', None) == 'astra_mountain':
                from .photographic_terrain import terrain_view_surface as terrain_source
            elif getattr(sim.scene, 'terrain', None) == 'detailed_virtual':
                from .detailed_terrain import terrain_view_surface as terrain_source
            terrain = terrain_source((sim.scene.width, sim.scene.height), self.map_rect.size,
                                     self.camera.visible_world())
            if terrain is not None:
                self.surface.blit(terrain, self.map_rect)
        self.surface.set_clip(world_rect.clip(self.map_rect))
        # At most 19 x 13 grid lines, regardless of config dimensions.
        if self.show_grid:
            for i in range(19):
                x, _ = self.world_to_screen(Point(sim.scene.width * i / 18, 0), sim)
                pygame.draw.line(self.surface, GRID, (x, tl[1]), (x, br[1]))
            for i in range(13):
                _, y = self.world_to_screen(Point(0, sim.scene.height * i / 12), sim)
                pygame.draw.line(self.surface, GRID, (tl[0], y), (br[0], y))
        self.surface.set_clip(world_rect.clip(self.map_rect))
        pygame.draw.rect(self.surface, MAP_EDGE, world_rect, width=1)

        for obstacle in sim.scene.obstacles:
            start = self.world_to_screen(Point(obstacle.x, obstacle.y), sim)
            end = self.world_to_screen(
                Point(obstacle.x + obstacle.width, obstacle.y + obstacle.height), sim
            )
            rect = pygame.Rect(start, (max(1, end[0] - start[0]), max(1, end[1] - start[1])))
            pygame.draw.rect(self.surface, (28, 56, 71), rect, border_radius=3)
            pygame.draw.rect(self.surface, (70, 107, 126), rect, width=1, border_radius=3)
            self.fitted_text(obstacle.id, (rect.x + 6, rect.y + 8),
                             max(1, rect.width - 12), 12, MUTED)

        self.surface.set_clip(world_rect.clip(self.map_rect))
        if sim.scene.rules:
            selected = next((u for u in sim.units if u.id == self.selected_unit_id), sim.units[0] if sim.units else None)
            if selected:
                self.selected_unit_id = selected.id
                scale = min(self.map_rect.width / sim.scene.width, self.map_rect.height / sim.scene.height) * self.zoom
                center = self.world_to_screen(self._display_positions[selected.id], sim)
                pygame.draw.circle(self.surface, (37, 94, 112), center,
                                   max(1, round(selected.sensor_range * scale)), 1)
                for contact in selected.contacts.values():
                    known = self.world_to_screen(contact.position, sim)
                    pygame.draw.circle(self.surface, WARNING, known, 8, 1)
                    pygame.draw.line(self.surface, (117, 95, 53), center, known, 1)
        waypoint_labels = []
        for unit in sim.units:
            color = TEAM_COLORS[unit.team]
            display_position = self._display_positions[unit.id]
            if self.show_trails and len(unit.trail) > 1:
                trail_color = tuple(round(v * .70 + MAP[i] * .30) for i, v in enumerate(color))
                points = unit.trail if display_position == unit.position else (
                    [*unit.trail, display_position] if getattr(sim, 'is_replay', False)
                    else [*unit.trail[:-1], display_position])
                pygame.draw.lines(self.surface, trail_color, False,
                                  [self.world_to_screen(p, sim) for p in points],
                                  2 if unit.id == self.selected_unit_id else 1)
            remaining = unit.path[unit.path_index:]
            if remaining:
                pygame.draw.lines(self.surface, (78, 109, 129), False,
                                  [self.world_to_screen(display_position, sim),
                                   *(self.world_to_screen(p, sim) for p in remaining)], 1)
            for index, point in enumerate(unit.waypoints):
                x, y = self.world_to_screen(point, sim)
                pygame.draw.circle(self.surface, color, (x, y), 5, width=1)
                waypoint_labels.append(self.text(str(index + 1), (x + 7, y - 13), 12, color).inflate(4, 4))
        self.surface.set_clip(world_rect.clip(self.map_rect))

        occupied_labels = waypoint_labels
        origin_label = self.label("原点 (0, 0) · x 向右 / y 向下", "Origin (0, 0) | x right / y down")
        origin_rect = self._render_text(origin_label, 12, MUTED).get_rect(
            topleft=(world_rect.x + 10, world_rect.bottom - 26))
        occupied_labels.append(origin_rect.inflate(4, 4))
        for team in Team:
            color = TEAM_COLORS[team]
            spawn = self.world_to_screen(sim.scene.spawn_points[team], sim)
            home = self.world_to_screen(sim.scene.return_points[team], sim)
            pygame.draw.circle(self.surface, color, spawn, 22, width=1)
            hx, hy = home
            pygame.draw.lines(self.surface, color, True,
                              [(hx, hy - 7), (hx + 7, hy), (hx, hy + 7), (hx - 7, hy)], 2)
            name = self.label("红方" if team == Team.RED else "蓝方", team.value.upper())
            if sim.scene.spawn_points[team] == sim.scene.return_points[team]:
                marks = [(spawn, self.label(f"{name} 出发 / 返航", f"{name} spawn / home"))]
            else:
                marks = [(spawn, self.label(f"{name} 出发", f"{name} spawn")),
                         (home, self.label(f"{name} 返航", f"{name} home"))]
            for (px, py), title in marks:
                rendered = self._render_text(title, 12, color)
                # Place site labels above their markers to avoid the nearby unit IDs.
                label_rect = rendered.get_rect(topleft=(px - rendered.get_width() // 2,
                                                        py - (58 if sim.has_missions else 44)))
                label_rect.clamp_ip(world_rect.inflate(-12, -12))
                self.surface.blit(rendered, label_rect)
                occupied_labels.append(label_rect.inflate(6, 4))

        # The world border is not a clipping boundary for an equipment image:
        # a unit at x=0/y=0 still shows its complete art in available map space.
        self.surface.set_clip(self.map_rect)
        label_units = []
        for unit in sim.units:
            display_position = self._display_positions[unit.id]
            x, y = self.world_to_screen(display_position, sim)
            color = TEAM_COLORS[unit.team]
            colocated = [other for other in sim.units if self._display_positions[other.id] == display_position]
            if sim.has_missions and len(colocated) > 1:
                # Leaders preserve the true position when several glyphs share a point.
                real_position = (x, y)
                index = next(i for i, other in enumerate(colocated) if other.id == unit.id)
                if len(colocated) == 2 and colocated[0].unit_type != colocated[1].unit_type:
                    dx, dy = (-28, 18) if unit.unit_type == UnitType.GROUND else (28, -18)
                else:
                    columns = max(1, int(len(colocated) ** .5 + .999))
                    rows = (len(colocated) + columns - 1) // columns
                    dx = round((index % columns - (columns - 1) / 2) * (MAP_UNIT_SIZE + 4))
                    dy = round((index // columns - (rows - 1) / 2) * (MAP_UNIT_SIZE + 4))
                x, y = x + dx, y + dy
                pygame.draw.line(self.surface, color, real_position, (x, y), 1)
            self.unit_hit_positions[unit.id] = (x, y)
            size = self._map_unit_size(unit.unit_type)
            self.unit_hit_rects[unit.id] = pygame.Rect(x - size // 2, y - size // 2, size, size)
            self._unit_icon((x, y), unit, color, size=size, selected=unit.id == self.selected_unit_id)
            if sim.scene.rules and unit.tag_flash_until_step > sim.step_count:
                pygame.draw.circle(self.surface, WARNING, (x, y), size // 2 + 3, 2)
            label_units.append((unit, x, y, color))

        # Draw all glyphs before labels, and reserve the selected ID first.
        # A crowded map can suppress an unselected ID; the list retains every ID.
        icon_boxes = [rect.inflate(8, 8) for rect in self.unit_hit_rects.values()]
        bounds = world_rect.clip(self.map_rect).inflate(-12, -12)
        label_units.sort(key=lambda entry: entry[0].id != self.selected_unit_id)
        for unit, x, y, color in label_units if self.show_labels else ():
            if not self.unit_hit_rects[unit.id].colliderect(bounds):
                continue
            rendered = self._render_text(unit.id, 13, color, max_width=max(1, bounds.width - 6))
            # Air and ground units may share a route/home, so label their two rows separately.
            gap = self._map_unit_size(unit.unit_type) // 2 + 10
            offset_y = -46 if sim.has_missions and unit.unit_type == UnitType.AIR else gap
            width, height = rendered.get_size()
            candidates = [(x + gap, y + offset_y), (x + gap, y + gap),
                          (x + gap, y - height - gap), (x - width - gap, y + gap),
                          (x - width - gap, y - height - gap),
                          (x - width // 2, y + gap), (x - width // 2, y - height - gap)]
            for distance in (50, 74, 98):
                for left in (x + gap, x - width - gap, x - width // 2):
                    candidates.extend(((left, y + distance), (left, y - height - distance)))
            if unit.id == self.selected_unit_id:
                # The selected ID may use the remaining viewport space before
                # lower-priority IDs compete for any positions near the cluster.
                candidates.extend((left, top)
                                  for top in range(bounds.top, bounds.bottom - height + 1, 24)
                                  for left in range(bounds.left, bounds.right - width + 1, 24))
            label_rect = None
            seen = set()
            blockers = occupied_labels + icon_boxes
            for left, top in candidates:
                label_rect = rendered.get_rect(topleft=(left, top))
                label_rect.clamp_ip(bounds)
                position = label_rect.topleft
                if position in seen:
                    label_rect = None
                    continue
                seen.add(position)
                padded = label_rect.inflate(6, 4)
                if not any(padded.colliderect(rect) for rect in blockers):
                    break
                label_rect = None
            if label_rect is None:
                continue
            padded = label_rect.inflate(6, 4)
            occupied_labels.append(padded)
            self.unit_label_rects[unit.id] = padded.copy()
            self.hidden_unit_labels.discard(unit.id)
            pygame.draw.rect(self.surface, MAP, padded, border_radius=3)
            self.surface.blit(rendered, label_rect)
        self.text(origin_label,
                  (world_rect.x + 10, world_rect.bottom - 26), 12, MUTED)
        self.surface.set_clip(old_clip)
        self._in_world = False

    def _draw_toolbar(self, sim, mouse_pos) -> None:
        for button in self.extra_buttons[:3]:
            enabled = self.extra_enabled(button.action, sim)
            title = self.label(button.zh, button.en)
            if button.action == "sharing":
                title += self.label(" 开" if getattr(sim, "sharing_enabled", False) else " 关",
                                    " On" if getattr(sim, "sharing_enabled", False) else " Off")
            selected = button.action == "sharing" and getattr(sim, "sharing_enabled", False)
            self._draw_button(button, f"{title}  {button.shortcut}", mouse_pos,
                              enabled=enabled, selected=selected)

    def _draw_timeline(self, replay, mouse_pos=(-1, -1)) -> None:
        caption_pos = (48, 700) if self.map_maximized else (274, 694)
        caption_width = 1184 if self.map_maximized else 684
        self.fitted_text(self.label(f"快照 {replay.index + 1}/{replay.count} · 记录状态 {self.state_label(replay.source_state)} · 播放 {self.playback_speed:g}x",
                                    f"Frame {replay.index + 1}/{replay.count} | Saved {replay.source_state.value}"),
                         caption_pos, caption_width, 12, MUTED)
        pygame.draw.rect(self.surface, EDGE, self.timeline, border_radius=5)
        ratio = replay.index / max(1, replay.count - 1)
        progress = self.timeline.copy()
        progress.width = max(1, round(progress.width * ratio))
        pygame.draw.rect(self.surface, ACCENT, progress, border_radius=5)
        pygame.draw.circle(self.surface, ACCENT,
                           (self.timeline.x + round((self.timeline.width - 1) * ratio), self.timeline.centery), 6)
        pygame.draw.circle(self.surface, PANEL,
                           (self.timeline.x + round((self.timeline.width - 1) * ratio), self.timeline.centery), 3)
        for button in self.extra_buttons[3:]:
            enabled = self.extra_enabled(button.action, replay)
            self._draw_button(button, self.label(button.zh, button.en), mouse_pos,
                              enabled=enabled, size=12)

    def _unit_icon(self, center, unit_type: UnitType, color, size=MAP_UNIT_SIZE, selected=False) -> None:
        glyph = unit_glyph_surface(unit_type, color, size, selected=selected)
        self.surface.blit(glyph, glyph.get_rect(center=center))

    @staticmethod
    def _map_unit_size(unit_type):
        return MAP_UNIT_SIZE if unit_type == UnitType.GROUND else MAP_AIR_SIZE

    def state_label(self, state):
        value = getattr(state, 'value', state)
        labels = {'READY': '就绪', 'RUNNING': '运行中', 'PAUSED': '已暂停', 'FINISHED': '已结束',
                  'IDLE': '待命', 'NAVIGATING': '导航', 'RETURNING': '返航',
                  'STOPPED': '停止', 'BLOCKED': '受阻'}
        return self.label(labels.get(value, value), value)

    def reason_label(self, value: str) -> str:
        labels = {
            'Returned to the team return point': '已到达阵营返航点',
            'Preset route completed': '预设路线已完成',
            'Target is unreachable: navigation speed must be positive and finite': '路径不可达：导航速度需为正的有限数值',
            'Target is unreachable: starting point is blocked or outside the traversable world': '路径不可达：起点受阻或超出可通行范围',
            'Target is unreachable: destination is blocked or outside the traversable world': '路径不可达：终点受阻或超出可通行范围',
            'Target is unreachable: no collision-free route connects the two points': '路径不可达：起点与终点之间没有无碰撞路线',
        }
        return self.label(labels.get(value, value), value)

    def visible_units(self, sim):
        query = self.unit_search.casefold()
        return [unit for unit in sim.units if query in unit.id.casefold()]

    def ui_event(self, event, sim):
        self._prepare_map(sim)
        if event.type == pygame.MOUSEBUTTONDOWN and event.button in (1, 2, 3) and self.map_rect.collidepoint(event.pos):
            self.active_camera.set_pose(self.zoom, self.active_camera.center)
            self._drag_button, self._drag_last = event.button, event.pos
            self._drag_origin = event.pos
            self.search_active = False
            # A left press still reaches App.select_at. Movement changes only
            # the camera, while an ordinary click keeps selecting equipment.
            return event.button != 1
        if event.type == pygame.MOUSEMOTION and self._drag_button is not None:
            if self._drag_origin is not None:
                if abs(event.pos[0]-self._drag_origin[0]) + abs(event.pos[1]-self._drag_origin[1]) < 4:
                    return True
                self._drag_origin = None
            delta = (event.pos[0] - self._drag_last[0], event.pos[1] - self._drag_last[1])
            if self.map_mode == 'world':
                self._world_view().pan(delta)
            else:
                self.camera.pan(delta, self._clock())
            self._drag_last = event.pos
            return True
        if event.type == pygame.MOUSEBUTTONUP and event.button == self._drag_button:
            self._drag_button = self._drag_last = None
            self._drag_origin = None
            return True
        if event.type in (pygame.WINDOWFOCUSLOST, pygame.WINDOWLEAVE):
            self._drag_button = self._drag_last = None
            self._drag_origin = None
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            self.search_active = not self.map_maximized and self.search_rect.collidepoint(event.pos)
            if self.search_active:
                return True
        if self.search_active and event.type == pygame.TEXTINPUT:
            if event.text != self._search_pending_text:
                self.unit_search += event.text
            self._search_pending_text = ""
            self.unit_offset = 0
            return True
        if self.search_active and event.type == pygame.KEYDOWN:
            if event.key == pygame.K_BACKSPACE:
                self.unit_search = self.unit_search[:-1]
                self.unit_offset = 0
            elif event.key in (pygame.K_RETURN, pygame.K_ESCAPE):
                self.search_active = False
            elif getattr(event, 'unicode', '') and event.unicode.isprintable():
                self.unit_search += event.unicode
                self._search_pending_text = event.unicode
                self.unit_offset = 0
            return True
        if event.type == pygame.MOUSEWHEEL:
            pos = pygame.mouse.get_pos()
            if not self.map_maximized and self.unit_scroll_rect.collidepoint(pos):
                self.unit_offset = max(0, min(max(0, len(self.visible_units(sim)) - 5), self.unit_offset - event.y))
            elif not self.map_maximized and self.contact_scroll_rect.collidepoint(pos):
                self.contact_offset = max(0, self.contact_offset - event.y)
            elif not self.map_maximized and self.event_scroll_rect.collidepoint(pos):
                self.event_offset = max(0, self.event_offset - event.y)
            elif self.map_rect.collidepoint(pos):
                amount = getattr(event, 'precise_y', event.y)
                factor = 1.12 ** max(-40, min(40, amount))
                if self.map_mode == 'world':
                    self._world_view().zoom_by(factor, (pos[0]-self.map_rect.x, pos[1]-self.map_rect.y))
                else:
                    self.camera.zoom_by(factor, pos, self._clock())
            return True
        return False

    def display_action(self, action, sim):
        self._prepare_map(sim)
        self.map_notice = ''
        if action in ('zoom_in', 'zoom_out'):
            factor = 1.2 if action == 'zoom_in' else 1 / 1.2
            if self.map_mode == 'world':
                self._world_view().zoom_by(factor)
            else:
                self.camera.zoom_by(factor, self.map_rect.center, self._clock())
        elif action == 'view_reset':
            self._drag_button = self._drag_last = None
            if self.map_mode == 'world':
                self._world_view().focus_global()
            else:
                self.camera.reset(self._clock())
        elif action == 'world_view':
            self.map_mode = 'world'
            self._world_view().focus_global()
        elif action == 'scene_view':
            self.map_mode = 'scene'
        elif action == 'focus_scene':
            if getattr(sim.scene,'terrain',None)=='astra_atlas':
                from .terrain_atlas import DETAIL_ZOOM
                self.map_mode='scene'
                unit=next((u for u in sim.units if u.id==self.selected_unit_id),None)
                center=unit.position if unit else Point(sim.scene.width/2,sim.scene.height/2)
                self.camera.set_pose(DETAIL_ZOOM,center)
            else:
                self.map_mode = 'world'
                if not self._world_view().focus_scene(scene_world(sim.scene), animate=True):
                    self.map_notice = self.label(
                        '此场景尚未设置地理参考；可浏览全球，点击“场景”返回虚拟地图，或在工作台配置参考。',
                        'Scene has no geographic reference. Browse Global, use Scene for the virtual map, or configure a reference in the Workbench.')
        elif action == 'map_maximize':
            self.map_maximized = not self.map_maximized
            self._layout_map(sim)
        elif action == 'trails':
            self.show_trails = not self.show_trails
        elif action == 'labels':
            self.show_labels = not self.show_labels
        elif action == 'terrain':
            self.show_terrain = not self.show_terrain
        elif action == 'grid':
            self.show_grid = not self.show_grid
        elif action == 'event_filter':
            self.filter_events = not self.filter_events
            self.event_offset = 0
        elif action == 'speed':
            speeds = (0.5, 1.0, 2.0, 4.0)
            self.playback_speed = speeds[(speeds.index(self.playback_speed) + 1) % len(speeds)]
        else:
            return False
        if action in ('world_view', 'scene_view', 'focus_scene', 'map_maximize'):
            self._drag_button = self._drag_last = self._drag_origin = None
        return True

    def _draw_status(self, sim: Simulation) -> None:
        replay = getattr(sim, 'is_replay', False)
        self._draw_panel(pygame.Rect(16, 112, 208, 604))
        self._title(self.label('运行概览', 'Overview'), (32, 123))
        self.fitted_text(self.state_label(sim.state), (151, 127), 57, 12, ACCENT)
        self.text(self.label('仿真时间 / 秒', 'Simulation time / s'), (32, 158), 13, MUTED)
        self.fitted_text(f'{sim.sim_time:.3f}', (32, 179), 116, 26, TEXT)
        steps = f'{sim.step_count} ' + self.label('步', 'steps')
        self.fitted_text(steps, (154, 192), 54, 12, MUTED)
        if replay:
            self.fitted_text(self.label('记录状态: ', 'Recorded: ') + self.state_label(sim.source_state),
                             (32, 220), 176, 13, ACCENT)
        elif sim.scene.rules:
            self.fitted_text(self.label('虚构积分 红/蓝: ', 'Virtual score R/B: ') +
                             f'{sim.scores[Team.RED]}/{sim.scores[Team.BLUE]}', (32, 220), 176, 13, ACCENT)
        self.text(self.label('单位列表', 'Units'), (32, 242), 12, MUTED)
        pygame.draw.rect(self.surface, MAP, self.search_rect, border_radius=4)
        pygame.draw.rect(self.surface, ACCENT if self.search_active else EDGE,
                         self.search_rect, 1, border_radius=4)
        self.fitted_text(self.unit_search or self.label('搜索编号 · 滚轮浏览', 'Search IDs | Scroll'),
                         (40, 269), 160, 12, TEXT if self.unit_search else MUTED)
        units = self.visible_units(sim)
        self.unit_offset = min(self.unit_offset, max(0, len(units) - 5))
        self.unit_rows = []
        for i, unit in enumerate(units[self.unit_offset:self.unit_offset + 5]):
            rect = pygame.Rect(32, 297 + i * 29, 176, 27)
            self.unit_rows.append((rect, unit.id))
            selected = unit.id == self.selected_unit_id
            fill = SELECTED if selected else (PANEL if i % 2 == 0 else MAP)
            pygame.draw.rect(self.surface, fill, rect, border_radius=3)
            if selected:
                pygame.draw.rect(self.surface, ACCENT, (rect.x, rect.y + 3, 3, rect.height - 6))
            self.fitted_text(unit.id, (40, rect.y + 5), 116, 13, TEAM_COLORS[unit.team])
            self.fitted_text(self.state_label(unit.behavior), (164, rect.y + 5), 40, 12, MUTED)
        self.text(f'{self.unit_offset + 1 if units else 0}–{min(len(units), self.unit_offset + 5)} / {len(units)}',
                  (126, 446), 12, MUTED)
        pygame.draw.line(self.surface, EDGE, (32, 473), (208, 473))
        self.text(self.label('场景概要', 'Scene facts'), (32, 485), 14, TEXT)
        air = sum(unit.unit_type == UnitType.AIR for unit in sim.units)
        sharing = self.label('开' if getattr(sim, 'sharing_enabled', False) else '关',
                             'On' if getattr(sim, 'sharing_enabled', False) else 'Off') \
                  if sim.scene.rules else self.label('未配置', 'N/A')
        facts = (
            self.label(f'空/地 {air}/{len(sim.units) - air} · {sim.scene.width:g}×{sim.scene.height:g}',
                       f'Air/ground {air}/{len(sim.units) - air} | {sim.scene.width:g}x{sim.scene.height:g}'),
            self.label(f'障碍 {len(sim.scene.obstacles)} · 步长 {sim.scene.fixed_dt:g}s',
                       f'Blocks {len(sim.scene.obstacles)} | dt {sim.scene.fixed_dt:g}s'),
            self.label(f'共享 {sharing} · 事件 {len(sim.events)}',
                       f'Sharing {sharing} | Events {len(sim.events)}'))
        for i, value in enumerate(facts):
            self.fitted_text(value, (32, 513 + i * 22), 176, 12, MUTED)

        self._draw_panel(pygame.Rect(992, 112, 272, 604))
        self._title(self.label('所选单位', 'Selected unit'), (1008, 123))
        unit = next((u for u in sim.units if u.id == self.selected_unit_id), sim.units[0] if sim.units else None)
        if not unit:
            self.text(self.label('暂无单位', 'No units'), (1008, 169), 14, MUTED)
            return
        self.selected_unit_id = unit.id
        preview_rect = pygame.Rect(1008, 158, 96, 96)
        pygame.draw.rect(self.surface, MAP, preview_rect, border_radius=6)
        pygame.draw.rect(self.surface, EDGE, preview_rect, 1, border_radius=6)
        glyph = unit_glyph_surface(unit, TEAM_COLORS[unit.team], 96)
        self.surface.blit(glyph, preview_rect)
        team = self.label('红方' if unit.team == Team.RED else '蓝方', unit.team.value)
        unit_type = self.label(equipment_label(unit), profile_key(unit))
        self.fitted_text(f'{team} / {unit_type}', (1116, 166), 132, 13, TEAM_COLORS[unit.team])
        self.fitted_text(self.state_label(unit.behavior), (1116, 192), 132, 13, TEXT)
        self.fitted_text(self.label('装备图示', 'Equipment artwork'), (1116, 219), 132, 12, MUTED)
        self.fitted_text(unit.id, (1008, 262), 240, 14, TEAM_COLORS[unit.team])
        pygame.draw.line(self.surface, EDGE, (1008, 286), (1248, 286))
        self.fitted_text(f'X {unit.position.x:.2f}  ·  Y {unit.position.y:.2f}', (1008, 294), 240, 13, MUTED)
        self.fitted_text(self.label('累计路程: ', 'Distance: ') + f'{unit.distance_travelled:.2f}',
                         (1008, 316), 240, 13, MUTED)
        if replay:
            value = getattr(sim, 'recorded_speeds', {}).get(unit.id)
            speed = self.label('不可用（无前一区间）', 'N/A (no prior interval)') if value is None else f'{value:.2f}'
            line = self.label('记录区间平均速度: ', 'Recorded interval speed: ') + speed + ' u/s'
        else:
            moving = sim.state == RunState.RUNNING and unit.behavior in (BehaviorState.NAVIGATING, BehaviorState.RETURNING)
            line = self.label('当前 / 配置速度: ', 'Current / config speed: ') + f'{unit.speed if moving else 0:g} / {unit.speed:g} u/s'
        self.fitted_text(line, (1008, 338), 240, 13, ACCENT)
        self.fitted_text(self.label('航点进度: ', 'Waypoints: ') +
                         f'{min(unit.waypoint_index, len(unit.waypoints))} / {len(unit.waypoints)}',
                         (1008, 360), 240, 13, MUTED)
        return_home = self.label('是' if unit.return_home else '否', str(unit.return_home))
        self.text(self.label('任务后返航: ', 'Return home: ') + return_home, (1008, 382), 13, MUTED)
        self.fitted_text(self.reason_label(unit.reason) or self.label('原因: 无异常', 'Reason: no exception'),
                         (1008, 408), 240, 12, WARNING if unit.behavior == BehaviorState.BLOCKED else MUTED)
        pygame.draw.line(self.surface, EDGE, (1008, 434), (1248, 434))
        contacts = sorted(unit.contacts.values(), key=lambda c: (-c.observed_step, c.target_id))
        self.contact_offset = min(self.contact_offset, max(0, len(contacts) - 2))
        start = self.contact_offset + 1 if contacts else 0
        end = min(len(contacts), self.contact_offset + 2)
        self.fitted_text(self.label('联系人 · 滚轮', 'Contacts | Scroll') + f'  {start}–{end} / {len(contacts)}',
                         (1008, 447), 240, 13, TEXT)
        if not contacts:
            self.text(self.label('暂无观测记录', 'No recorded contacts'), (1008, 482), 13, MUTED)
        for i, contact in enumerate(contacts[self.contact_offset:self.contact_offset + 2]):
            top = 479 + i * 72
            self.fitted_text(contact.target_id, (1008, top), 240, 13, TEXT)
            source = self.label('来自 ', 'From ') + contact.source_id + ' · ' + self.label(
                '共享' if contact.shared else '直接', 'shared' if contact.shared else 'direct')
            self.fitted_text(source, (1014, top + 21), 234, 12, MUTED)
            self.fitted_text(f'({contact.position.x:.1f}, {contact.position.y:.1f}) · ' +
                             self.label('步 ', 'step ') + str(contact.observed_step),
                             (1014, top + 41), 234, 12, MUTED)
        pygame.draw.line(self.surface, EDGE, (1008, 625), (1248, 625))
        self.fitted_text(self.label('观测范围: ', 'Sensor range: ') + f'{unit.sensor_range:g}',
                         (1008, 640), 240, 13, MUTED)
        self.fitted_text(self.label('虚构标记: ', 'Virtual tags: ') + str(unit.tag_count),
                         (1008, 666), 240, 13, MUTED)
        reference = GeoReference.from_world(scene_world(sim.scene))
        if reference is not None:
            lon, lat = reference.to_lonlat(unit.position.x, unit.position.y, sim.scene.width, sim.scene.height)
            self.fitted_text(self.label('显示参考 ', 'Display reference ') + format_lonlat(lon, lat),
                             (1008, 690), 240, 12, MUTED)

    def _draw_events(self, sim: Simulation, mouse_pos=(-1, -1)) -> None:
        titles = {
            "run_started": ("开始", "Started"), "paused": ("暂停", "Paused"),
            "resumed": ("继续", "Resumed"), "route_planned": ("规划路线", "Route planned"),
            "waypoint_reached": ("到达路点", "Waypoint reached"),
            "return_started": ("返航", "Returning"), "unit_stopped": ("停止", "Stopped"),
            "path_blocked": ("路径受阻", "Blocked"), "run_finished": ("运行结束", "Finished"),
            "object_discovered": ("发现对象", "Object discovered"), "info_shared": ("收到共享", "Shared contact"),
            "contact_lost": ("观测失效", "Contact expired"), "virtual_tag": ("虚拟标记 +1", "Virtual tag +1"),
            "sharing_changed": ("共享切换", "Sharing changed"), "round_finished": ("本轮结束", "Round finished"),
        }
        pygame.draw.line(self.surface, EDGE, (32, 581), (208, 581))
        self.text(self.label("最近事件", "Recent events"), (32, 587), 14, TEXT)
        events = [(index, event) for index, event in enumerate(sim.events)
                  if not self.filter_events or event.get('unit_id') == self.selected_unit_id
                  or event.get('details', {}).get('target_id') == self.selected_unit_id]
        count = 3
        self.event_scroll_rect = pygame.Rect(16, 583, 208, 132)
        self.event_offset = min(self.event_offset, max(0, len(events)-count))
        end = len(events)-self.event_offset
        self.event_rows = []
        start = max(0, end - count)
        self.fitted_text(f'{start + 1 if events else 0}–{end}/{len(events)}',
                         (128, 590), 80, 12, MUTED)
        if not events:
            self.text(self.label('暂无事件', 'No events'), (32, 617), 12, MUTED)
            self.fitted_text(self.label('完整记录见工作台', 'Full history: Workbench'), (32, 646), 176, 12, MUTED)
        for i, (index, event) in enumerate(events[max(0,end-count):end]):
            rect = pygame.Rect(32, 607 + i * 36, 176, 35)
            self.event_rows.append((rect, index))
            pygame.draw.rect(self.surface, MAP if i % 2 == 0 else PANEL, rect, border_radius=3)
            if getattr(sim, 'is_replay', False) and rect.collidepoint(mouse_pos):
                pygame.draw.rect(self.surface, HOVER, rect, border_radius=3)
            zh, en = titles.get(event["kind"], (event["kind"], event["kind"]))
            actor = event.get('unit_id') or self.label('系统', 'SYSTEM')
            line = f"{event['time']:.2f}s  {self.label(zh, en)}"
            self.fitted_text(line, (rect.x + 5, rect.y + 1), rect.width - 10, 12, TEXT)
            self.fitted_text(actor, (rect.x + 5, rect.y + 18), rect.width - 10, 12, MUTED)

    def _draw_controls(self, sim: Simulation, mouse_pos) -> None:
        for button in self.buttons:
            enabled = self.enabled(button.action, sim.state)
            title = f"{self.label(button.zh, button.en)}  {button.shortcut}"
            self._draw_button(button, title, mouse_pos, enabled=enabled,
                              primary=button.action in ('start', 'resume') and enabled)

    def _draw_legend(self) -> None:
        self._unit_icon((672, 754), UnitType.GROUND, MUTED)
        self.text(self.label("车", "UGV"), (692, 745), 14, MUTED)
        self._unit_icon((756, 754), UnitType.AIR, MUTED)
        self.text(self.label("机", "UAV"), (775, 745), 14, MUTED)
        pygame.draw.circle(self.surface, MUTED, (850, 754), 11, width=1)
        self.text(self.label("出发", "Spawn"), (870, 745), 14, MUTED)
        pygame.draw.lines(self.surface, MUTED, True,
                          [(970, 746), (978, 754), (970, 762), (962, 754)], 1)
        self.text(self.label("返航", "Home"), (990, 745), 14, MUTED)
        self.text(self.label("矩形为障碍", "Blocks: obstacles"), (1082, 745), 14, MUTED)
