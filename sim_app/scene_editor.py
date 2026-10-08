"""Tk scene editing without editing JSON or touching a running simulation."""

from __future__ import annotations

import math
from base64 import b64encode
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any, Callable

from .models import GameRules
from .geography import GeoReference, format_lonlat
from .scene import SceneConfigError
from .scene_document import CONFIG_ROOT, TEMPLATES, USER_SCENE_ROOT, SceneDocument, blank_scene
from .ui_theme import COLORS, BooleanToggle, CollapsibleFrame, PanelHeader, apply_theme, cached_font
from .tk_motion import Transition
from .visual_assets import draw_tk_unit, terrain_view_ppm, unit_glyph_png
from .world_map import WorldMapView
from .equipment import equipment_choices, equipment_profile, equipment_label, profile_key


class SceneEditor(ttk.Frame):
    """An independent scene draft with a saved-path-only run callback.

    ``load_document`` returns False if an unsaved-change prompt is cancelled.
    ``get_document`` exposes the service; its ``data`` property is a copy.
    ``dirty`` includes unapplied form values. The enclosing workbench should
    call ``confirm_discard`` (or ``can_close``) before closing the window.
    """

    def __init__(self, parent: tk.Misc, on_run: Callable[[str], None]) -> None:
        apply_theme(parent)
        super().__init__(parent)
        self._font_family = self.winfo_toplevel()._workbench_font_family
        self._map_font = cached_font(self, 9)
        self._map_bold_font = cached_font(self, 9, 'bold')
        self.on_run = on_run
        self._document = SceneDocument()
        self._selected: tuple[Any, ...] | None = None
        self._loading = False
        self._refreshing = False
        self._scene_pending = False
        self._item_pending = False
        self._geo_pending = False
        self._drag: dict[str, Any] | None = None
        self._canvas_targets: dict[str, tuple[Any, ...]] = {}
        self._transform = (1.0, 24.0, 24.0)
        self._camera_zoom = self._zoom_target = 1.0
        self._camera_pan = (0.0, 0.0)
        self._camera_view = None
        self._pan_drag = None
        self._space_down = False
        self.pan_mode = tk.BooleanVar(value=False)
        self._redraw_after = None
        self._world_animation_after = None
        self._layout_after = None
        self._world_view = WorldMapView()
        self._world_image = None
        self._world_viewport = None
        self._world_fit_pending = False
        self._map_expanded = False
        self._saved_panes = None
        self._zoom_motion = Transition(self)
        self.bind('<Destroy>', self._cancel_view_callbacks, add='+')
        self._scene_vars: dict[str, tk.Variable] = {}
        self._geo_vars: dict[str, tk.Variable] = {}
        self._item_vars: dict[str, tk.Variable] = {}
        self._item_fields: dict[str, ttk.Widget] = {}
        self._item_labels: dict[str, ttk.Label] = {}
        self._scrolling_contents = []
        self._terrain_images = {}
        self._terrain_image = None
        self._preview_images = {}
        self.preview_image = None
        self.map_mode = tk.StringVar(value='terrain')
        self.map_view = tk.StringVar(value='虚拟场景')
        self.atlas_region = tk.StringVar(value='中央河谷')
        self.grid_visible = tk.BooleanVar(value=False)
        self.template_var = tk.StringVar(value=next(iter(TEMPLATES)))
        # Off by default: rules scenes sample the recording, which keeps a
        # ~45 s demo near a few megabytes instead of tens of megabytes.
        self.full_record_var = tk.BooleanVar(value=False)
        self.status_var = tk.StringVar()
        self.coordinate_var = tk.StringVar(value="拖空白平移 · 拖单位编辑 · 滚轮缩放")
        self.zoom_var = tk.StringVar(value='100%')
        self._build()
        try:
            self._document = SceneDocument.open(TEMPLATES[self.template_var.get()])
        except (SceneConfigError, OSError) as error:
            self.status_var.set(f"模板不可用：{error}；可新建或打开场景")
        if self._document.data['units']:
            self._selected = ('units', 0)
        self._set_loaded_map_view()
        self._refresh()

    @property
    def dirty(self) -> bool:
        return self._document.dirty or self._scene_pending or self._item_pending or self._geo_pending

    def get_document(self) -> SceneDocument:
        return self._document

    def load_document(self, payload: dict[str, Any], *, source_path: str | Path | None = None) -> bool:
        # Check new input before asking to discard a valid current draft.
        candidate = SceneDocument(payload, source_path=source_path)
        candidate.validate()
        if not self.confirm_discard():
            return False
        self._document = candidate
        self._selected = None
        self._zoom_motion.cancel()
        self._camera_zoom = self._zoom_target = 1.0
        self._camera_pan = (0.0, 0.0)
        self._camera_view = None
        self._scene_pending = self._item_pending = self._geo_pending = False
        self._set_loaded_map_view()
        self._refresh()
        self.status_var.set("已载入独立场景草稿；运行前将保存副本")
        return True

    def confirm_discard(self) -> bool:
        if not self.dirty:
            return True
        choice = messagebox.askyesnocancel(
            "未保存的场景修改", "当前场景有未保存修改。\n是：保存后继续\n否：放弃修改\n取消：留在当前场景",
            parent=self.winfo_toplevel())
        if choice is None:
            return False
        if choice:
            return self.save_document() is not None
        return True

    def can_close(self) -> bool:
        return self.confirm_discard()

    def _build(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(2, weight=1)
        toolbar = ttk.Frame(self, padding=(0, 4, 0, 14))
        toolbar.grid(row=0, column=0, sticky="ew")
        ttk.Button(toolbar, text="保存并运行", command=self.run_document,
                   style="Primary.TButton").pack(side="right", padx=(12, 0))
        ttk.Checkbutton(toolbar, text="完整记录（60 Hz）", variable=self.full_record_var).pack(
            side="right", padx=(12, 0))
        ttk.Label(toolbar, text="场景", style="Muted.TLabel").pack(side="left", padx=(0, 6))
        ttk.Combobox(toolbar, textvariable=self.template_var, values=list(TEMPLATES),
                     width=11, state="readonly").pack(side="left")
        for group in ((("载入", self.load_template),),
                      (("撤销", self.undo), ("重做", self.redo), ("校验", self.validate_document)),
                      (("保存", self.save_document),)):
            if group[0][0] != "载入":
                ttk.Separator(toolbar, orient="vertical").pack(side="left", fill="y", padx=9, pady=2)
            for text, command in group:
                style = "Ghost.TButton"
                ttk.Button(toolbar, text=text, command=command, style=style).pack(side="left", padx=(4, 0))
        more = ttk.Menubutton(toolbar, text="场景文件 ▾")
        more.pack(side="left", padx=(8, 0))
        menu = tk.Menu(more, tearoff=False)
        for title, command in (("新建场景", self.new_document), ("打开场景…", self.open_document),
                               ("另存为…", self.save_copy), ("查看场景路径…", self.show_scene_path)):
            menu.add_command(label=title, command=command)
        more.configure(menu=menu)
        body = self.body = ttk.Panedwindow(self, orient="horizontal")
        body.grid(row=2, column=0, sticky="nsew")
        left = ttk.Frame(body, width=200, style="HudPanel.TFrame")
        left.columnconfigure(0, weight=1)
        left.rowconfigure(2, weight=1)
        PanelHeader(left, "场景资源").grid(row=0, column=0, columnspan=2, sticky="ew")
        self.resource_summary = tk.StringVar(value="实体与障碍")
        ttk.Label(left, textvariable=self.resource_summary, style="PanelMuted.TLabel", padding=(10, 8)).grid(row=1, column=0, columnspan=2, sticky="ew")
        self.tree = ttk.Treeview(left, show="tree", selectmode="browse", height=15)
        self.tree.column('#0', width=160, minwidth=120)
        self.tree.grid(row=2, column=0, sticky="nsew")
        scroll = ttk.Scrollbar(left, orient="vertical", command=self.tree.yview)
        scroll.grid(row=2, column=1, sticky="ns")
        self.tree.configure(yscrollcommand=scroll.set)
        self.tree.bind("<<TreeviewSelect>>", self._on_tree_select)
        self.tree.tag_configure('resource_group', foreground=COLORS['accent'], background=COLORS['surface'])
        left_buttons = ttk.Frame(left, padding=(4, 5), style="Panel.TFrame")
        left_buttons.grid(row=3, column=0, columnspan=2, sticky="ew")
        for row, (label, command) in enumerate((("+ 实体", self.add_unit), ("+ 矩形障碍", self.add_obstacle),
                                                ("+ 任务点", self.add_waypoint), ("删除所选", self.delete_selected))):
            ttk.Button(left_buttons, text=label, command=command).grid(row=row // 2, column=row % 2,
                                                                       sticky="ew", padx=2, pady=2)
        left_buttons.columnconfigure((0, 1), weight=1)
        body.add(left, weight=0)
        middle = ttk.Frame(body, style="HudPanel.TFrame")
        middle.columnconfigure(0, weight=1)
        middle.rowconfigure(1, weight=1)
        map_header = ttk.Frame(middle, style='Header.TFrame')
        map_header.grid(row=0, column=0, sticky='ew')
        map_header.columnconfigure(0, weight=1)
        PanelHeader(map_header, "仿真地图").grid(row=0, column=0, sticky='ew')
        map_tools = ttk.Frame(map_header, style='Header.TFrame', padding=(4, 2, 8, 2))
        map_tools.grid(row=0, column=1, sticky='e')
        ttk.Checkbutton(map_tools, text='地形', onvalue='terrain', offvalue='grid',
                        variable=self.map_mode, command=self._draw,
                        style='MapToggle.TCheckbutton').pack(side='left', padx=2)
        ttk.Checkbutton(map_tools, text='网格', variable=self.grid_visible,
                        command=self._draw, style='MapToggle.TCheckbutton').pack(side='left', padx=2)
        self.canvas = tk.Canvas(middle, background=COLORS["map"], highlightthickness=0,
                                highlightbackground=COLORS["border"], width=360, height=420, takefocus=True)
        self.canvas.grid(row=1, column=0, sticky="nsew")
        self.canvas.bind("<Configure>", self._canvas_configure)
        self.canvas.bind("<Button-1>", self._canvas_press)
        self.canvas.bind("<B1-Motion>", self._canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self._canvas_release)
        self.canvas.bind("<Motion>", self._canvas_motion)
        self.canvas.bind('<MouseWheel>', self._canvas_zoom)
        self.canvas.bind('<Button-4>', lambda event: self.zoom_map(1.12, (event.x, event.y)))
        self.canvas.bind('<Button-5>', lambda event: self.zoom_map(1 / 1.12, (event.x, event.y)))
        for button in (2, 3):
            self.canvas.bind(f'<Button-{button}>', self._pan_start)
            self.canvas.bind(f'<B{button}-Motion>', self._pan_move)
            self.canvas.bind(f'<ButtonRelease-{button}>', self._pan_end)
        self.canvas.bind('<KeyPress-space>', self._space_press)
        self.canvas.bind('<KeyRelease-space>', self._space_release)
        self.canvas.bind('<FocusOut>', self._view_focus_out)
        self.canvas.bind('<Leave>', self._pan_end)
        map_footer = ttk.Frame(middle, style='Panel.TFrame')
        map_footer.grid(row=2, column=0, sticky='ew')
        map_footer.columnconfigure(0, weight=1)
        zoom_controls = ttk.Frame(map_footer, style='Panel.TFrame', padding=(6, 3))
        zoom_controls.grid(row=0, column=0, sticky='w')
        self.zoom_out_button = ttk.Button(zoom_controls, text='−', width=2, style='PanelGhost.TButton',
                                          command=lambda: self.zoom_map(1 / 1.2))
        self.zoom_out_button.pack(side='left')
        ttk.Label(zoom_controls, textvariable=self.zoom_var, width=5, anchor='center',
                  style='PanelMuted.TLabel').pack(side='left', padx=2)
        self.zoom_in_button = ttk.Button(zoom_controls, text='+', width=2, style='PanelGhost.TButton',
                                         command=lambda: self.zoom_map(1.2))
        self.zoom_in_button.pack(side='left')
        ttk.Button(zoom_controls, text='适配', command=self.fit_map,
                   style='PanelGhost.TButton').pack(side='left', padx=(4, 0))
        self.pan_button = ttk.Checkbutton(zoom_controls, text='平移模式', variable=self.pan_mode,
                                          command=self._pan_mode_changed, style='MapToggle.TCheckbutton')
        self.pan_button.pack(side='left', padx=(8, 0))
        # Keep map navigation on its own compact row at the minimum window size.
        navigation = ttk.Frame(map_footer, style='Panel.TFrame', padding=(6, 2))
        navigation.grid(row=1, column=0, sticky='w')
        view_selector = ttk.Combobox(navigation, textvariable=self.map_view,
                                    values=('虚拟场景', '世界地图'), width=8, state='readonly')
        view_selector.pack(side='left', padx=(0, 4))
        view_selector.bind('<<ComboboxSelected>>', self._map_view_changed)
        self.global_button = ttk.Button(navigation, text='全球', style='PanelGhost.TButton',
                                        command=self.focus_global)
        self.global_button.pack(side='left')
        self.scene_focus_button = ttk.Button(navigation, text='定位场景', style='PanelGhost.TButton',
                                             command=self.focus_scene)
        self.scene_focus_button.pack(side='left', padx=2)
        self.expand_button = ttk.Button(navigation, text='展开地图', style='PanelGhost.TButton',
                                       command=self.toggle_map_expanded)
        self.expand_button.pack(side='left', padx=2)
        self.region_navigation=ttk.Frame(map_footer,style='Panel.TFrame',padding=(8,2))
        self.region_navigation.grid(row=2,column=0,sticky='w')
        ttk.Label(self.region_navigation,text='区域',style='PanelMuted.TLabel').pack(side='left',padx=(0,6))
        from .terrain_atlas import REGION_NAMES
        self.region_selector=ttk.Combobox(self.region_navigation,textvariable=self.atlas_region,
                                         values=('全域总览','当前装备区',*(name for row in REGION_NAMES for name in row)),
                                         width=16,state='readonly')
        self.region_selector.pack(side='left')
        self.region_selector.bind('<<ComboboxSelected>>',self._atlas_region_changed)
        self.region_navigation.grid_remove()
        coordinate = ttk.Label(map_footer, textvariable=self.coordinate_var, style="PanelMuted.TLabel",
                               padding=(8, 4))
        coordinate.grid(row=3, column=0, sticky="ew")
        middle.bind("<Configure>", lambda event: coordinate.configure(wraplength=max(180, event.width - 16)))
        body.add(middle, weight=1)
        right = ttk.Frame(body, width=280, style="HudPanel.TFrame")
        PanelHeader(right, "属性与规则").pack(fill='x')
        self.properties = ttk.Notebook(right, width=280)
        self.properties.pack(fill='both', expand=True)
        body.add(right, weight=0)
        self._side_panes = (left, right)
        self._build_item_tab()
        self._build_scene_tab()
        # Bind locally so the wheel works over fields and disclosure headers,
        # without hijacking scrolling in the other workbench pages.
        for content, wheel in self._scrolling_contents:
            pending = [content]
            while pending:
                child = pending.pop()
                child.bind("<MouseWheel>", wheel)
                pending.extend(child.winfo_children())
        footer = ttk.Frame(self, padding=(0, 8, 0, 0))
        footer.grid(row=3, column=0, sticky="ew")
        footer.columnconfigure(0, weight=1)
        status = ttk.Label(footer, textvariable=self.status_var, anchor="w", style="Muted.TLabel")
        status.grid(row=0, column=0, sticky="ew")
        ttk.Button(footer, text="场景路径…", command=self.show_scene_path, style="Ghost.TButton").grid(row=0, column=1, padx=(8, 0))
        footer.bind("<Configure>", lambda event: status.configure(wraplength=max(200, event.width - 110)))
        self._pane_layout_width = 0
        def arrange(event: tk.Event) -> None:
            if self._map_expanded or len(body.panes()) != 3:
                return
            if event.width > 650 and event.width != self._pane_layout_width:
                self._pane_layout_width = event.width
                compact = event.width < 1000
                body.sashpos(0, 174 if compact else 200)
                body.sashpos(1, event.width - (246 if compact else 280))
        body.bind("<Configure>", arrange)

    def toggle_map_expanded(self) -> None:
        """Remove panes without recreating their forms or changing the camera."""
        if self._layout_after is not None:
            self.after_cancel(self._layout_after)
            self._layout_after = None
        if not self._map_expanded:
            self.body.update_idletasks()
            panes = self.body.panes()
            self._saved_panes = [(pane, dict(self.body.pane(pane))) for pane in panes]
            self._saved_sashes = tuple(self.body.sashpos(index) for index in range(len(panes)-1))
            self._map_expanded = True
            for pane in self._side_panes:
                self.body.forget(pane)
            self.expand_button.configure(text='恢复面板')
        else:
            for index, (pane, options) in enumerate(self._saved_panes or ()):
                if pane not in self.body.panes():
                    position = index if index < len(self.body.panes()) else 'end'
                    self.body.insert(position, pane, **options)
            self._map_expanded = False
            self.expand_button.configure(text='展开地图')
            self._layout_after = self.after_idle(self._restore_pane_positions)
        self._request_draw()

    def _restore_pane_positions(self) -> None:
        self._layout_after = None
        if self._map_expanded:
            return
        self.body.update_idletasks()
        width = self.body.winfo_width()
        positions = self._saved_sashes
        # Startup can expand before the first sash layout has been committed.
        # Restore usable columns rather than a saved pair of zero positions.
        if width > 650 and len(self.body.panes()) == 3 and (
                len(positions) != 2 or positions[0] < 120 or
                positions[1] < positions[0] + 100 or positions[1] > width - 120):
            compact = width < 1000
            positions = (174 if compact else 200, width - (246 if compact else 280))
        for index, position in enumerate(positions):
            self.body.sashpos(index, position)
        self._request_draw()

    def _scrolling_tab(self, title: str) -> ttk.Frame:
        tab = ttk.Frame(self.properties, style="Panel.TFrame")
        tab.columnconfigure(0, weight=1)
        tab.rowconfigure(0, weight=1)
        canvas = tk.Canvas(tab, highlightthickness=0, width=230, background=COLORS["surface"])
        canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(tab, orient="vertical", command=canvas.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        canvas.configure(yscrollcommand=scrollbar.set)
        content = ttk.Frame(canvas, padding=10, style="Panel.TFrame")
        content.columnconfigure(1, weight=1)
        window = canvas.create_window(0, 0, anchor="nw", window=content)
        content.bind("<Configure>", lambda event: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda event: canvas.itemconfigure(window, width=event.width))
        def wheel(event: tk.Event) -> str:
            canvas.yview_scroll(-int(event.delta / 120), "units")
            return "break"
        canvas.bind("<MouseWheel>", wheel)
        content.bind("<MouseWheel>", wheel)
        self._scrolling_contents.append((content, wheel))
        self.properties.add(tab, text=title)
        return content

    def _variable(self, group: str, key: str, boolean: bool = False) -> tk.Variable:
        variable = tk.BooleanVar(value=False) if boolean else tk.StringVar()
        target = self._scene_vars if group == "scene" else self._item_vars
        target[key] = variable
        variable.trace_add("write", lambda *args: self._mark_pending(group))
        return variable

    def _mark_pending(self, group: str) -> None:
        if self._loading:
            return
        if group == "scene":
            self._scene_pending = True
        elif group == "geo":
            self._geo_pending = True
        else:
            self._item_pending = True
        self.status_var.set("有未应用的表单修改；保存、运行或切换对象前将应用")

    def _field(self, frame: ttk.Frame, row: int, label: str, key: str, *, group: str,
               choices: tuple[str, ...] | None = None, boolean: bool = False) -> ttk.Widget:
        field_label = ttk.Label(frame, text=label, style="Panel.TLabel")
        field_label.grid(row=row, column=0, sticky="w", pady=4, padx=(0, 7))
        var = self._variable(group, key, boolean)
        if boolean:
            widget = BooleanToggle(frame, variable=var)
        elif choices:
            widget = ttk.Combobox(frame, textvariable=var, values=choices, state="readonly", width=10)
        else:
            widget = ttk.Entry(frame, textvariable=var, width=12)
        widget.grid(row=row, column=1, sticky="ew", pady=4)
        if group == "item":
            self._item_fields[key] = widget
            self._item_labels[key] = field_label
        return widget

    def _build_item_tab(self) -> None:
        frame = self._scrolling_tab("所选对象")
        self.selection_var = tk.StringVar(value="从地图或列表选择对象")
        ttk.Label(frame, textvariable=self.selection_var, wraplength=255, style="PanelMuted.TLabel").grid(row=0, column=0, columnspan=2,
                                                                             sticky="w", pady=(0, 7))
        self.item_preview = ttk.Frame(frame, style='Panel.TFrame')
        self.item_preview.grid(row=1, column=0, columnspan=2, sticky='ew', pady=(0, 8))
        self.item_preview.columnconfigure(1, weight=1)
        self.preview_label = ttk.Label(self.item_preview, style='Panel.TLabel')
        self.preview_label.grid(row=0, column=0, rowspan=2, padx=(0, 8))
        self.preview_type = tk.StringVar()
        self.preview_team = tk.StringVar()
        ttk.Label(self.item_preview, textvariable=self.preview_type, style='Panel.TLabel').grid(row=0, column=1, sticky='sw')
        ttk.Label(self.item_preview, textvariable=self.preview_team, style='PanelMuted.TLabel').grid(row=1, column=1, sticky='nw')
        fields = (("编号", "id", None, False), ("阵营", "team", ("red", "blue"), False),
                  ("通行类别", "type", ("ground", "air"), False),
                  ("装备类型", "equipment", tuple(p.label for p in equipment_choices()), False), ("X 坐标", "x", None, False),
                  ("Y 坐标", "y", None, False), ("速度", "speed", None, False),
                  ("观测范围", "sensor_range", None, False), ("任务后返航", "return_home", None, True),
                  ("障碍宽度", "width", None, False), ("障碍高度", "height", None, False))
        for row, (label, key, choices, boolean) in enumerate(fields, 2):
            self._field(frame, row, label, key, group="item", choices=choices, boolean=boolean)
        self.item_hint = ttk.Label(frame, text="速度 / 观测留空时沿用默认值。", style="PanelMuted.TLabel", wraplength=255)
        self._item_fields['equipment'].bind('<<ComboboxSelected>>', self._equipment_selected)
        self._item_fields['type'].bind('<<ComboboxSelected>>', self._movement_selected)
        self.item_hint.grid(row=13, column=0, columnspan=2, sticky="w", pady=7)
        self.item_apply = ttk.Button(frame, text="应用修改", command=self.apply_changes)
        self.item_apply.grid(row=14, column=0, columnspan=2, sticky="ew", pady=5)
        self.waypoint_var = tk.StringVar(value="所选实体的任务点顺序在左侧列表中显示")
        self.waypoint_label = ttk.Label(frame, textvariable=self.waypoint_var, wraplength=255, style="PanelMuted.TLabel")
        self.waypoint_label.grid(row=15, column=0, columnspan=2, sticky="w", pady=5)
        controls = self.waypoint_controls = ttk.Frame(frame, style="Panel.TFrame")
        controls.grid(row=16, column=0, columnspan=2, sticky="ew")
        ttk.Button(controls, text="上移", command=lambda: self.reorder_waypoint(-1)).pack(side="left", expand=True, fill="x")
        ttk.Button(controls, text="下移", command=lambda: self.reorder_waypoint(1)).pack(side="left", expand=True, fill="x", padx=(4, 0))

    def _build_scene_tab(self) -> None:
        frame = self._scrolling_tab("场景与规则")
        for row, (label, key) in enumerate((("场景名称", "name"), ("世界宽度", "world.width"), ("世界高度", "world.height"))):
            self._field(frame, row, label, key, group="scene")
        self.team_section = CollapsibleFrame(frame, "出生与返航点")
        self.team_section.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(12, 0))
        row = 0
        for collection, label in (("spawn_points", "出生"), ("return_points", "返航")):
            for team, team_label in (("red", "红队"), ("blue", "蓝队")):
                for axis in ("x", "y"):
                    self._field(self.team_section.body, row, f"{team_label}{label} {axis.upper()}", f"{collection}.{team}.{axis}", group="scene")
                    row += 1
        self.rule_section = CollapsibleFrame(frame, "虚构交互规则")
        self.rule_section.grid(row=4, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        rules = self.rule_section.body
        self._field(rules, 0, "启用规则", "rules_enabled", group="scene", boolean=True)
        self._field(rules, 1, "共享观察信息", "rules.sharing_enabled", group="scene", boolean=True)
        for row, (label, key) in enumerate((("目标积分", "score_limit"), ("时限（秒）", "time_limit")), 2):
            self._field(rules, row, label, f"rules.{key}", group="scene")
        self.advanced_section = CollapsibleFrame(frame, "高级参数")
        self.advanced_section.grid(row=6, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        advanced = self.advanced_section.body
        self._field(advanced, 0, "固定步长（秒）", "fixed_dt", group="scene")
        for row, (label, key) in enumerate((("标记范围", "tag_range"), ("间隔（秒）", "tag_cooldown"),
                                           ("有效期（秒）", "contact_ttl"), ("转发距离（0=全队）", "sharing_range")), 1):
            self._field(advanced, row, label, f"rules.{key}", group="scene")
        self.geo_section = CollapsibleFrame(frame, "地理参考")
        self.geo_section.grid(row=5, column=0, columnspan=2, sticky='ew', pady=(8, 0))
        for row, (label, key) in enumerate((('启用地理参考', 'enabled'),
                                           ('中心纬度（°）', 'center_latitude'),
                                           ('中心经度（°）', 'center_longitude'),
                                           ('米 / 仿真单位', 'meters_per_unit'))):
            variable = tk.BooleanVar(value=False) if key == 'enabled' else tk.StringVar()
            self._geo_vars[key] = variable
            variable.trace_add('write', lambda *args: self._mark_pending('geo'))
            ttk.Label(self.geo_section.body, text=label, style='Panel.TLabel').grid(
                row=row, column=0, sticky='w', pady=4, padx=(0, 7))
            widget = BooleanToggle(self.geo_section.body, variable=variable) if key == 'enabled' else ttk.Entry(
                self.geo_section.body, textvariable=variable, width=12)
            widget.grid(row=row, column=1, sticky='ew', pady=4)
        ttk.Label(self.geo_section.body, text='定义地图显示映射；通行、速度与观测规则仍采用既有模型。',
                  style='PanelMuted.TLabel', wraplength=235).grid(row=4, column=0, columnspan=2, sticky='w', pady=7)
        ttk.Button(self.geo_section.body, text='应用地理参考', command=self.apply_georeference).grid(
            row=5, column=0, columnspan=2, sticky='ew', pady=4)
        ttk.Label(frame, text="坐标、速度采用仿真单位。\n缩小世界后请先校验。", style="PanelMuted.TLabel",
                  wraplength=255).grid(row=7, column=0, columnspan=2, sticky="w", pady=(12, 8))
        ttk.Button(frame, text="应用修改", command=self.apply_changes).grid(row=8, column=0, columnspan=2,
                                                                           sticky="ew", pady=5)

    def _number(self, variable: tk.Variable, path: str, *, integer: bool = False) -> float | int:
        value = str(variable.get()).strip()
        try:
            if integer:
                number = int(value)
                if str(number) != value and value not in (f"+{number}", f"-{abs(number)}"):
                    raise ValueError
            else:
                number = float(value)
            if not math.isfinite(number):
                raise ValueError
            return number
        except (ValueError, OverflowError):
            raise SceneConfigError("<form>", path, "请输入有限整数" if integer else "请输入有限数值（不接受布尔值、NaN 或无穷大）") from None

    def _apply_pending(self, *, refresh: bool = True) -> bool:
        if not self._scene_pending and not self._item_pending and not self._geo_pending:
            return True
        data = self._document.data
        try:
            if self._scene_pending:
                data["name"] = str(self._scene_vars["name"].get()).strip()
                data["world"].update({axis: self._number(self._scene_vars[f"world.{axis}"], f"world.{axis}")
                                      for axis in ("width", "height")})
                data["fixed_dt"] = self._number(self._scene_vars["fixed_dt"], "fixed_dt")
                for collection in ("spawn_points", "return_points"):
                    for team in ("red", "blue"):
                        data[collection][team] = {axis: self._number(self._scene_vars[f"{collection}.{team}.{axis}"],
                                                                  f"{collection}.{team}.{axis}") for axis in ("x", "y")}
                if self._scene_vars["rules_enabled"].get():
                    rules = {"sharing_enabled": bool(self._scene_vars["rules.sharing_enabled"].get())}
                    for key in ("tag_range", "tag_cooldown", "score_limit", "time_limit", "contact_ttl",
                                "sharing_range"):
                        rules[key] = self._number(self._scene_vars[f"rules.{key}"], f"rules.{key}", integer=key == "score_limit")
                    if rules["sharing_range"] < 0:
                        raise SceneConfigError("<form>", "rules.sharing_range", "请输入不小于 0 的数值；0 表示全队广播")
                    data["rules"] = rules
                else:
                    data.pop("rules", None)
            if self._geo_pending:
                if self._geo_vars['enabled'].get():
                    reference = {key: self._number(self._geo_vars[key], f'world.georeference.{key}')
                                 for key in ('center_latitude', 'center_longitude', 'meters_per_unit')}
                    candidate_world = dict(data['world'], georeference=reference)
                    try:
                        geo = GeoReference.from_world(candidate_world)
                        geo.validate_extent(candidate_world['width'], candidate_world['height'])
                    except (ValueError, TypeError) as error:
                        raise SceneConfigError('<form>', 'world.georeference', str(error)) from error
                    data['world']['georeference'] = reference
                else:
                    data['world'].pop('georeference', None)
            if self._item_pending and self._selected is not None:
                kind = self._selected[0]
                if kind in ("units", "obstacles"):
                    index = self._selected[1]
                    item = data[kind][index]
                    field = f"{kind}[{index}]"
                    item["id"] = str(self._item_vars["id"].get()).strip()
                    for axis in ("x", "y"):
                        item[axis] = self._number(self._item_vars[axis], f"{field}.{axis}")
                    if kind == "units":
                        item["team"] = str(self._item_vars["team"].get())
                        item["type"] = str(self._item_vars["type"].get())
                        label = self._item_vars['equipment'].get()
                        profile = next((p for p in equipment_choices() if p.label == label), None)
                        if profile is None or profile.unit_type.value != item['type']:
                            profile = equipment_profile(item['type'])
                        if 'equipment' in item or profile.key != profile_key(item['type']):
                            item['equipment'] = profile.key
                        item["return_home"] = bool(self._item_vars["return_home"].get())
                        for key in ("speed", "sensor_range"):
                            if str(self._item_vars[key].get()).strip():
                                item[key] = self._number(self._item_vars[key], f"{field}.{key}")
                            else:
                                item.pop(key, None)
                    else:
                        for key in ("width", "height"):
                            item[key] = self._number(self._item_vars[key], f"{field}.{key}")
                elif kind == "waypoints":
                    unit_index, point_index = self._selected[1:]
                    point = data["units"][unit_index]["waypoints"][point_index]
                    for axis in ("x", "y"):
                        point[axis] = self._number(self._item_vars[axis], f"units[{unit_index}].waypoints[{point_index}].{axis}")
                else:
                    collection, team = self._selected
                    for axis in ("x", "y"):
                        data[collection][team][axis] = self._number(self._item_vars[axis], f"{collection}.{team}.{axis}")
        except (SceneConfigError, IndexError, KeyError) as error:
            self._show_error(error)
            return False
        self._document.replace(data)
        self._scene_pending = self._item_pending = self._geo_pending = False
        if refresh:
            self._refresh()
        return True

    def apply_changes(self) -> bool:
        if not self._apply_pending():
            return False
        self._set_status("修改已应用；保存前仍会校验全部字段")
        return True

    def apply_georeference(self) -> bool:
        if not self._apply_pending():
            return False
        self._set_status('地理参考已应用；可定位场景' if self._scene_geo() else '已取消地理参考')
        return True

    def _show_error(self, error: Exception) -> None:
        field = getattr(error, "field", "")
        if field.startswith('world.georeference'):
            if self._map_expanded:
                self.toggle_map_expanded()
            self.geo_section.set_expanded(True)
            self.properties.select(1)
        elif field.startswith(("spawn_points", "return_points")):
            self.team_section.set_expanded(True)
            self.properties.select(1)
        elif field.startswith("rules") or field == "fixed_dt":
            self.rule_section.set_expanded(True)
            self.advanced_section.set_expanded(True)
            self.properties.select(1)
        self.status_var.set(f"错误：{error}")
        messagebox.showerror("场景未保存", str(error), parent=self.winfo_toplevel())

    def _set_status(self, detail: str = "") -> None:
        source = self._document.path or self._document.source_path
        filename = Path(source).name if source else "新场景"
        mark = "未保存修改" if self.dirty else "草稿未修改"
        self.status_var.set(f"{mark} | {filename}" + (f" | {detail}" if detail else ""))

    def show_scene_path(self) -> None:
        """Keep full source information available without a long footer."""
        source = self._document.source_path
        saved = self._document.path
        messagebox.showinfo("场景路径", f"来源：{source or '新建场景'}\n\n编辑副本：{saved or '尚未保存'}",
                            parent=self.winfo_toplevel())

    def _refresh(self) -> None:
        self._refreshing = True
        self._loading = True
        try:
            data = self._document.data
            self.resource_summary.set(f"{len(data['units'])} 实体 · {len(data['obstacles'])} 障碍")
            defaults = GameRules()
            for key, variable in self._scene_vars.items():
                if key == "rules_enabled":
                    value = "rules" in data
                elif key.startswith("rules."):
                    field = key.split(".")[1]
                    value = data.get("rules", {}).get(field, getattr(defaults, field))
                else:
                    value: Any = data
                    for part in key.split("."):
                        value = value[part]
                variable.set(value)
            reference = data['world'].get('georeference') or {}
            for key, variable in self._geo_vars.items():
                variable.set(bool(reference) if key == 'enabled' else reference.get(key, ''))
            self.tree.delete(*self.tree.get_children())
            for kind, title in (("units", "实体"), ("obstacles", "矩形障碍"),
                                ("spawn_points", "出生点"), ("return_points", "返航点")):
                self.tree.insert("", "end", iid=kind, text=title, open=True, tags=('resource_group',))
                if kind in ("units", "obstacles"):
                    for index, item in enumerate(data[kind]):
                        key = f"{kind}:{index}"
                        text = item["id"]
                        # Type and team are shown in the selected-object form;
                        # the tree keeps room for the actual identifier.
                        self.tree.insert(kind, "end", iid=key, text=text, open=True)
                        if kind == "units":
                            for point_index, point in enumerate(item.get("waypoints", [])):
                                self.tree.insert(key, "end", iid=f"waypoints:{index}:{point_index}",
                                                 text=f"任务 {point_index + 1} ({point['x']:g}, {point['y']:g})")
                else:
                    for team, team_label in (("red", "红队"), ("blue", "蓝队")):
                        point = data[kind][team]
                        self.tree.insert(kind, "end", iid=f"{kind}:{team}", text=f"{team_label} ({point['x']:g}, {point['y']:g})")
            self._show_selection()
            if self._selected:
                iid = self._selection_key(self._selected)
                if self.tree.exists(iid):
                    self.tree.selection_set(iid)
                    self.tree.see(iid)
                else:
                    self._selected = None
                    self._show_selection()
            self._draw()
            self._set_status()
        finally:
            self._loading = False
            self._refreshing = False

    @staticmethod
    def _selection_key(selection: tuple[Any, ...]) -> str:
        return ":".join(str(value) for value in selection)

    @staticmethod
    def _decode_key(key: str) -> tuple[Any, ...] | None:
        parts = key.split(":")
        if len(parts) < 2:
            return None
        return tuple([parts[0]] + [int(value) if value.isdecimal() else value for value in parts[1:]])

    def select(self, selection: tuple[Any, ...] | None) -> bool:
        if selection == self._selected:
            return True
        if not self._apply_pending():
            return False
        self._selected = selection
        self._loading = True
        try:
            self._show_selection()
        finally:
            self._loading = False
        self._refreshing = True
        try:
            if selection and self.tree.exists(self._selection_key(selection)):
                self.tree.selection_set(self._selection_key(selection))
                self.tree.see(self._selection_key(selection))
        finally:
            self._refreshing = False
        self.properties.select(0)
        self._draw()
        return True

    def _on_tree_select(self, event: tk.Event | None = None) -> None:
        if self._refreshing:
            return
        values = self.tree.selection()
        selection = self._decode_key(values[0]) if values else None
        if not self.select(selection) and self._selected:
            self._refreshing = True
            self.tree.selection_set(self._selection_key(self._selected))
            self._refreshing = False

    def _show_selection(self) -> None:
        data = self._document.data
        for key, variable in self._item_vars.items():
            variable.set(False if key == "return_home" else "")
            self._item_fields[key].configure(state="disabled")
            self._item_fields[key].grid_remove()
            self._item_labels[key].grid_remove()
        self.item_hint.grid_remove()
        self.item_apply.grid_remove()
        self.waypoint_label.grid_remove()
        self.waypoint_controls.grid_remove()
        self.item_preview.grid_remove()
        self.preview_image = None
        self.preview_label.configure(image='')
        if self._selected is None:
            self.selection_var.set("从地图或列表选择对象")
            self.waypoint_var.set("选中实体后可添加任务点")
            return
        kind = self._selected[0]
        try:
            if kind in ("units", "obstacles"):
                index = self._selected[1]
                item = data[kind][index]
                label = "实体" if kind == "units" else "矩形障碍"
                self.selection_var.set(f"{label}：{item['id']}")
                keys = ("id", "x", "y", "team", "type", "equipment", "speed", "sensor_range", "return_home") if kind == "units" else ("id", "x", "y", "width", "height")
                self.waypoint_var.set(f"任务点 {len(item.get('waypoints', []))} 个 · 按列表顺序执行" if kind == "units" else "X / Y 为障碍左上角坐标。")
                if kind == "units":
                    self.item_hint.grid()
                    self._show_unit_preview(item)
            elif kind == "waypoints":
                unit_index, point_index = self._selected[1:]
                item = data["units"][unit_index]["waypoints"][point_index]
                self.selection_var.set(f"{data['units'][unit_index]['id']} · 任务点 {point_index + 1}")
                keys = ("x", "y")
                self.waypoint_var.set("上移 / 下移调整该点的任务执行顺序")
                self.waypoint_controls.grid()
            else:
                collection, team = self._selected
                item = data[collection][team]
                label = "出生" if collection == "spawn_points" else "返航"
                self.selection_var.set(f"{'红队' if team == 'red' else '蓝队'}{label}点")
                keys = ("x", "y")
                self.waypoint_var.set("队伍的公共点位，可拖动或修改坐标")
            for key in keys:
                value = equipment_label(item) if key == 'equipment' else item.get(key, False if key == "return_home" else "")
                self._item_vars[key].set(value)
                self._item_fields[key].configure(state="readonly" if key in ("team", "type", "equipment") else "normal")
                self._item_fields[key].grid()
                self._item_labels[key].grid()
            self.item_apply.grid()
            self.waypoint_label.grid()
        except (IndexError, KeyError):
            self._selected = None
            self.selection_var.set("对象已删除；请选择其他对象")

    def _show_unit_preview(self, unit) -> None:
        key = (profile_key(unit), unit['team'])
        if key not in self._preview_images:
            pixels = unit_glyph_png(unit, COLORS[unit['team']], size=96)
            self._preview_images[key] = tk.PhotoImage(master=self, data=b64encode(pixels).decode('ascii'))
        self.preview_image = self._preview_images[key]
        self.preview_label.configure(image=self.preview_image)
        self.preview_type.set(equipment_label(unit))
        self.preview_team.set('红方实体' if unit['team'] == 'red' else '蓝方实体')
        self.item_preview.grid()

    def _equipment_selected(self, event=None):
        profile = next((p for p in equipment_choices() if p.label == self._item_vars['equipment'].get()), None)
        if profile is not None:
            self._item_vars['type'].set(profile.unit_type.value)
            if self._selected and self._selected[0] == 'units':
                row = dict(self._document.data['units'][self._selected[1]],
                           type=profile.unit_type.value, equipment=profile.key)
                self._show_unit_preview(row)

    def _movement_selected(self, event=None):
        movement = self._item_vars['type'].get()
        self._item_vars['equipment'].set(equipment_label(movement))
        self._equipment_selected()

    def _terrain_photo(self, size, world_size, viewport):
        terrain_kind = self._document.data['world'].get('terrain')
        key = (size, world_size, viewport, terrain_kind)
        if key not in self._terrain_images:
            source = terrain_view_ppm
            if terrain_kind=='astra_atlas':
                from .terrain_atlas import terrain_view_ppm as source
            elif terrain_kind == 'astra_mountain':
                from .photographic_terrain import terrain_view_ppm as source
            elif terrain_kind == 'detailed_virtual':
                from .detailed_terrain import terrain_view_ppm as source
            pixels = source(world_size=world_size, viewport_size=size, world_viewport=viewport)
            if pixels is None:
                return None
            photo = tk.PhotoImage(master=self.canvas, data=pixels, format='PPM')
            if len(self._terrain_images) >= 4:
                self._terrain_images.pop(next(iter(self._terrain_images)))
            self._terrain_images[key] = photo
        return self._terrain_images[key]

    def new_document(self) -> bool:
        return self.load_document(blank_scene())

    def load_template(self) -> bool:
        try:
            document = SceneDocument.open(TEMPLATES[self.template_var.get()])
            return self.load_document(document.data, source_path=document.source_path)
        except (SceneConfigError, OSError) as error:
            self._show_error(error)
            return False

    def open_document(self) -> bool:
        path = filedialog.askopenfilename(parent=self.winfo_toplevel(), title="打开场景（编辑后保存副本）",
                                          initialdir=str(USER_SCENE_ROOT if USER_SCENE_ROOT.exists() else CONFIG_ROOT),
                                          filetypes=(("场景 JSON", "*.json"), ("所有文件", "*.*")))
        if not path:
            return False
        try:
            document = SceneDocument.open(path)
            return self.load_document(document.data, source_path=path)
        except (SceneConfigError, OSError) as error:
            self._show_error(error)
            return False

    def undo(self) -> bool:
        if self._scene_pending or self._item_pending or self._geo_pending:
            # First undo uncommitted form edits, then use document history.
            self._scene_pending = self._item_pending = self._geo_pending = False
            self._refresh()
            self._set_status("已撤销尚未应用的表单修改")
            return True
        result = self._document.undo()
        self._refresh()
        self._set_status("已撤销" if result else "没有可撤销修改")
        return result

    def redo(self) -> bool:
        if not self._apply_pending():
            return False
        result = self._document.redo()
        self._refresh()
        self._set_status("已重做" if result else "没有可重做修改")
        return result

    def validate_document(self) -> bool:
        if not self._apply_pending():
            return False
        try:
            scene = self._document.validate()
        except SceneConfigError as error:
            self._show_error(error)
            return False
        self._set_status(f"校验通过：{len(scene.units)} 实体，{len(scene.obstacles)} 障碍")
        return True

    def save_document(self, path: str | Path | None = None) -> Path | None:
        if not self._apply_pending():
            return None
        try:
            saved = self._document.save(path)
        except (SceneConfigError, OSError) as error:
            self._show_error(error)
            return None
        self._set_status("已保存")
        return saved

    def save_copy(self) -> Path | None:
        if not self.validate_document():
            return None
        suggested = self._document.suggested_path()
        # The dialog starts in the nearest existing directory and saving will
        # create the default user directory, including for a first-time user.
        try:
            USER_SCENE_ROOT.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            self._show_error(error)
            return None
        path = filedialog.asksaveasfilename(parent=self.winfo_toplevel(), title="保存独立场景副本",
                                           initialdir=str(suggested.parent), initialfile=suggested.name,
                                           defaultextension=".json", filetypes=(("场景 JSON", "*.json"),))
        return self.save_document(path) if path else None

    def run_document(self) -> bool:
        # Every launch has an immutable disk input, even if this editor keeps
        # changing while the Pygame subprocess is active.
        if not self._apply_pending():
            return False
        try:
            self._document.save()
            # Keep the editable saved document separate from the launch copy;
            # even a subsequent Save cannot rewrite this run's disk input.
            run_input = SceneDocument(self._document.data, source_path=self._document.path,
                                      user_root=self._document.user_root)
            saved = run_input.save()
        except (SceneConfigError, OSError) as error:
            self._show_error(error)
            return False
        self._set_status("运行输入已保存")
        self.on_run(str(saved))
        return True

    def add_unit(self) -> bool:
        if not self._apply_pending():
            return False
        index = self._document.add_unit()
        self._selected = ("units", index)
        self._refresh()
        self.properties.select(0)
        return True

    def add_obstacle(self) -> bool:
        if not self._apply_pending():
            return False
        index = self._document.add_obstacle()
        self._selected = ("obstacles", index)
        self._refresh()
        self.properties.select(0)
        return True

    def _selected_unit(self) -> int | None:
        if self._selected and self._selected[0] in ("units", "waypoints"):
            return self._selected[1]
        return None

    def add_waypoint(self, x: float | None = None, y: float | None = None) -> bool:
        if not self._apply_pending():
            return False
        unit_index = self._selected_unit()
        if unit_index is None:
            self._set_status("请先选择一个实体，再添加任务点")
            return False
        data = self._document.data
        unit = data["units"][unit_index]
        points = unit.get("waypoints", [])
        previous = points[-1] if points else unit
        if x is None:
            x = min(float(data["world"]["width"]) - .001, previous["x"] + 60)
        if y is None:
            y = previous["y"]
        point_index = self._document.add_waypoint(unit_index, x, y)
        self._selected = ("waypoints", unit_index, point_index)
        self._refresh()
        self.properties.select(0)
        return True

    def reorder_waypoint(self, offset: int) -> bool:
        if not self._apply_pending():
            return False
        if not self._selected or self._selected[0] != "waypoints":
            self._set_status("请从左侧列表选择要调整顺序的任务点")
            return False
        _, unit_index, point_index = self._selected
        new_index = self._document.move_waypoint(unit_index, point_index, offset)
        self._selected = ("waypoints", unit_index, new_index)
        self._refresh()
        return new_index != point_index

    def delete_selected(self) -> bool:
        if not self._apply_pending():
            return False
        if not self._selected:
            self._set_status("请先选择要删除的实体、障碍或任务点")
            return False
        kind = self._selected[0]
        if kind in ("units", "obstacles"):
            self._document.delete_item(kind, self._selected[1])
            self._selected = None
        elif kind == "waypoints":
            unit_index, point_index = self._selected[1:]
            self._document.delete_waypoint(unit_index, point_index)
            self._selected = ("units", unit_index)
        else:
            self._set_status("出生 / 返航点为必需字段，可移动而不能删除")
            return False
        self._refresh()
        return True

    def _scene_geo(self) -> GeoReference | None:
        world = self._document.data['world']
        try:
            geo = GeoReference.from_world(world)
            if geo is not None:
                geo.validate_extent(world['width'], world['height'])
            return geo
        except (ValueError, TypeError, KeyError, OverflowError):
            return None

    def _is_world_map(self) -> bool:
        return self.map_view.get() == '世界地图'

    def _resize_world_view(self) -> None:
        size = (max(self.canvas.winfo_width(), 120), max(self.canvas.winfo_height(), 120))
        if size != self._world_viewport:
            self._world_view.resize(size)
            self._world_viewport = size
            if self._world_fit_pending and min(size) > 120:
                self._world_view.focus_scene(self._document.data['world'], animate=False)
                self._world_fit_pending = False

    def _set_loaded_map_view(self) -> None:
        self._cancel_world_animation()
        self._world_viewport = None
        self._world_fit_pending = self._scene_geo() is not None
        self._resize_world_view()
        if self._scene_geo() is None:
            self.map_view.set('虚拟场景')
            self._world_view.focus_global(animate=False)
        else:
            self.map_view.set('世界地图')
            self._world_view.focus_scene(self._document.data['world'], animate=False)
        if self._document.data['world'].get('terrain')=='astra_atlas':
            from .terrain_atlas import DETAIL_ZOOM
            self._camera_zoom=self._zoom_target=DETAIL_ZOOM
            self._camera_pan=(0.0,0.0)
            self.atlas_region.set('中央河谷')
            self.region_navigation.grid()
            self.scene_focus_button.configure(text='定位装备')
        else:
            self.region_navigation.grid_remove()
            self.scene_focus_button.configure(text='定位场景')
        self.region_selector.configure(state='disabled' if self._is_world_map() else 'readonly')

    def _map_view_changed(self, event=None) -> None:
        self._drag = self._pan_drag = None
        self._zoom_motion.cancel()
        self._cancel_world_animation()
        self.canvas.configure(cursor='')
        if self._is_world_map():
            self._resize_world_view()
            self.coordinate_var.set('左拖浏览世界 · 点击实体选择 · 坐标编辑请切换虚拟场景')
        else:
            self.coordinate_var.set('拖空白平移 · 拖单位编辑 · 滚轮缩放')
        self.region_selector.configure(state='disabled' if self._is_world_map() else 'readonly')
        self._draw()

    def focus_global(self, *, animate=True) -> None:
        self._world_fit_pending = False
        self.map_view.set('世界地图')
        self._map_view_changed()
        self._world_view.focus_global(animate=animate)
        self._animate_world()

    def focus_scene(self, *, animate=True) -> bool:
        if self._document.data['world'].get('terrain')=='astra_atlas' and self._scene_geo() is None:
            self.atlas_region.set('当前装备区')
            self._atlas_region_changed(animate=animate)
            return True
        if not self._apply_pending():
            return False
        if self._scene_geo() is None:
            if self._map_expanded:
                self.toggle_map_expanded()
            self.geo_section.set_expanded(True)
            self.properties.select(1)
            self._set_status('请先启用并填写地理参考，再定位场景')
            return False
        self._world_fit_pending = False
        self.map_view.set('世界地图')
        self._map_view_changed()
        result = self._world_view.focus_scene(self._document.data['world'], animate=animate)
        self._animate_world()
        return bool(result)

    def _atlas_region_changed(self,event=None,*,animate=True):
        """Navigate photographic regions without writing scene coordinates."""
        world=self._document.data['world']
        if world.get('terrain')!='astra_atlas':
            return
        self.map_view.set('虚拟场景')
        self._cancel_world_animation()
        self.region_selector.configure(state='readonly')
        name=self.atlas_region.get()
        if name=='全域总览':
            self.fit_map(animate=animate)
            return
        from .terrain_atlas import DETAIL_ZOOM,regions
        if name=='当前装备区':
            unit_index=self._selected_unit()
            if unit_index is not None:
                unit=self._document.data['units'][unit_index]
                center=(unit['x'],unit['y'])
            else:
                center=(world['width']/2,world['height']/2)
            zoom=DETAIL_ZOOM
        else:
            region=next((item for item in regions((world['width'],world['height'])) if item['name']==name),None)
            if region is None:
                return
            center,zoom=region['center'],region['zoom']
        self._focus_virtual_view(center,zoom,animate=animate)

    def _focus_virtual_view(self,center,zoom,*,animate=True):
        self._zoom_motion.cancel()
        self._drag=self._pan_drag=None
        cw,ch=max(self.canvas.winfo_width(),120),max(self.canvas.winfo_height(),120)
        if self._camera_view is None or self._camera_view[2:]!=(cw,ch):
            self._draw()
        world=self._document.data['world']
        origin=self._world_xy(cw/2,ch/2)
        origin_zoom=self._camera_zoom
        target=max(1,min(self._max_virtual_zoom(),zoom))
        self._zoom_target=target
        fit=min((cw-64)/world['width'],(ch-64)/world['height'])
        def update(amount):
            self._camera_zoom=origin_zoom+(target-origin_zoom)*amount
            x=origin[0]+(center[0]-origin[0])*amount
            y=origin[1]+(center[1]-origin[1])*amount
            scale=fit*self._camera_zoom
            self._camera_pan=((world['width']/2-x)*scale,(world['height']/2-y)*scale)
            self._draw()
        if animate:
            self._zoom_motion.start(update,duration=160)
        else:
            update(1.0)

    def _cancel_world_animation(self) -> None:
        if self._world_animation_after is not None:
            try:
                self.after_cancel(self._world_animation_after)
            except tk.TclError:
                pass
            self._world_animation_after = None

    def _animate_world(self) -> None:
        # Replacing wheel targets keeps exactly one callback pending.
        self._cancel_world_animation()
        if not self._is_world_map():
            return
        self._world_view.update()
        self._draw()
        if self._world_view.camera.animating:
            self._world_animation_after = self.after(16, self._animate_world)

    def _draw_world(self) -> None:
        self._resize_world_view()
        self.canvas.delete('all')
        self._canvas_targets.clear()
        data = self._document.data
        world = dict(data['world'])
        world['obstacles'] = data['obstacles']
        world['sites'] = [dict(point, team=team, label=('红队' if team == 'red' else '蓝队')+label)
                          for collection, label in (('spawn_points', '出发'), ('return_points', '返航'))
                          for team, point in data[collection].items()]
        if self._scene_geo() is None:
            world.pop('georeference', None)
        selected_index = self._selected_unit()
        selected_id = data['units'][selected_index]['id'] if selected_index is not None else None
        self._world_view.render(world, units_rows=data['units'], selected_id=selected_id,
                                show_grid=self.grid_visible.get(), show_terrain=self.map_mode.get() == 'terrain',
                                title=data['name'])
        self._world_image = tk.PhotoImage(master=self.canvas, data=self._world_view.ppm(), format='PPM')
        self.canvas.create_image(0, 0, anchor='nw', image=self._world_image, tags=('world_background',))
        self.pan_button.configure(state='disabled')
        self._zoom_buttons()

    def _xy(self, x: float, y: float) -> tuple[float, float]:
        scale, left, top = self._transform
        return left + x * scale, top + y * scale

    def _world_xy(self, x: float, y: float) -> tuple[float, float]:
        scale, left, top = self._transform
        return (x - left) / scale, (y - top) / scale

    def _cancel_view_callbacks(self, event):
        if event.widget is self:
            self._zoom_motion.cancel()
            self._cancel_world_animation()
            if self._redraw_after is not None:
                self.after_cancel(self._redraw_after)
                self._redraw_after = None
            if self._layout_after is not None:
                self.after_cancel(self._layout_after)
                self._layout_after = None

    def _request_draw(self):
        if self._redraw_after is None:
            self._redraw_after = self.after_idle(self._draw)

    def _canvas_configure(self, event):
        if self._is_world_map():
            self._request_draw()
            return
        if self._camera_view is None or self._camera_view[2:] != (event.width, event.height):
            self._stop_zoom()
            self._request_draw()

    def _stop_zoom(self):
        self._zoom_motion.cancel()
        self._zoom_target = self._camera_zoom

    def _zoom_buttons(self):
        if self._is_world_map():
            camera = self._world_view.camera
            self.zoom_var.set(f'{camera.zoom:.1f}×')
            self.zoom_out_button.configure(state='disabled' if camera.target_zoom <= camera.min_zoom else 'normal')
            self.zoom_in_button.configure(state='disabled' if camera.target_zoom >= camera.max_zoom else 'normal')
            return
        self.zoom_var.set(f'{self._camera_zoom * 100:.0f}%')
        self.zoom_out_button.configure(state='disabled' if self._zoom_target <= 1 else 'normal')
        self.zoom_in_button.configure(state='disabled' if self._zoom_target >= self._max_virtual_zoom() else 'normal')

    def _max_virtual_zoom(self):
        terrain=self._document.data['world'].get('terrain')
        return 256 if terrain=='astra_atlas' else 64 if terrain in ('detailed_virtual','astra_mountain') else 6

    def zoom_map(self, factor, anchor=None, *, animate=True):
        """Zoom the current view without moving the world point at the anchor."""
        if self._drag is not None or self._pan_drag is not None:
            return 'break'
        if self._is_world_map():
            self._world_fit_pending = False
            self._resize_world_view()
            self._world_view.zoom_by(factor, anchor=anchor)
            self._animate_world()
            return 'break'
        cw, ch = max(self.canvas.winfo_width(), 120), max(self.canvas.winfo_height(), 120)
        if self._camera_view is None or self._camera_view[2:] != (cw, ch):
            self._draw()
        world = self._document.data['world']
        if world['width'] <= 0 or world['height'] <= 0:
            return 'break'
        anchor = anchor or (cw / 2, ch / 2)
        point = self._world_xy(*anchor)
        target = max(1.0, min(self._max_virtual_zoom(), self._zoom_target * factor))
        self._zoom_motion.cancel()
        self._zoom_target = target
        origin = self._camera_zoom
        fit_scale = min((cw-64) / world['width'], (ch-64) / world['height'])
        def update(amount):
            self._camera_zoom = origin + (target-origin) * amount
            scale = fit_scale * self._camera_zoom
            self._camera_pan = (anchor[0] - point[0] * scale - (cw-world['width']*scale) / 2,
                                anchor[1] - point[1] * scale - (ch-world['height']*scale) / 2)
            self._draw()
        if animate and abs(target-origin) > 1e-6:
            self._zoom_motion.start(update, duration=120)
        else:
            update(1.0)
        return 'break'

    def fit_map(self, *, animate=True):
        if self._is_world_map():
            return self.focus_scene(animate=animate)
        if self._document.data['world'].get('terrain')=='astra_atlas':
            self.atlas_region.set('全域总览')
        self._zoom_motion.cancel()
        self._pan_drag = None
        origin, pan = self._camera_zoom, self._camera_pan
        self._zoom_target = 1.0
        def update(amount):
            self._camera_zoom = origin + (1-origin) * amount
            self._camera_pan = (pan[0] * (1-amount), pan[1] * (1-amount))
            self._draw()
        if animate and (origin != 1 or pan != (0.0, 0.0)):
            self._zoom_motion.start(update, duration=120)
        else:
            update(1.0)

    def _canvas_zoom(self, event):
        delta = getattr(event, 'delta', 0)
        return self.zoom_map(1.12 ** (delta / 120), (event.x, event.y)) if delta else 'break'

    def _space_press(self, event):
        self._space_down = True
        self.canvas.configure(cursor='hand2')
        return 'break'

    def _pan_mode_changed(self):
        self._drag = None
        self._pan_end()
        self.coordinate_var.set('平移模式：左键拖动地图 · 滚轮缩放 · 点击“平移模式”返回编辑'
                                if self.pan_mode.get() else '拖空白平移 · 拖单位编辑 · 滚轮缩放')
        self._request_draw()

    def _space_release(self, event):
        self._space_down = False
        self._pan_end(event)
        return 'break'

    def _view_focus_out(self, event):
        self._space_down = False
        self._pan_end(event)
        if self._drag is not None:
            self._drag = None
            self._request_draw()

    def _pan_start(self, event):
        self.canvas.focus_set()
        self._stop_zoom()
        self._drag = None
        if self._is_world_map():
            self._cancel_world_animation()
            self._world_fit_pending = False
            self._pan_drag = (event.x, event.y, None)
            self.canvas.configure(cursor='fleur')
            return 'break'
        self._pan_drag = (event.x, event.y, self._camera_pan)
        self.canvas.configure(cursor='fleur')
        return 'break'

    def _pan_move(self, event):
        if self._pan_drag is None:
            return 'break'
        x, y, origin = self._pan_drag
        if self._is_world_map():
            self._world_view.pan((event.x-x, event.y-y))
            self._pan_drag = (event.x, event.y, None)
            self._canvas_motion(event)
            self._request_draw()
            return 'break'
        pan = (origin[0] + event.x-x, origin[1] + event.y-y)
        world = self._document.data['world']
        cw, ch = max(self.canvas.winfo_width(), 120), max(self.canvas.winfo_height(), 120)
        scale = min((cw-64) / world['width'], (ch-64) / world['height']) * self._camera_zoom
        sw, sh = world['width'] * scale, world['height'] * scale
        # Leave a visible strip of the world even after an excessive drag.
        left = max(32-sw, min(cw-32, (cw-sw) / 2 + pan[0]))
        top = max(32-sh, min(ch-32, (ch-sh) / 2 + pan[1]))
        self._camera_pan = (left-(cw-sw) / 2, top-(ch-sh) / 2)
        self._request_draw()
        return 'break'

    def _pan_end(self, event=None):
        if self._pan_drag is not None:
            self._pan_drag = None
            self._draw()
        self.canvas.configure(cursor='hand2' if self._space_down or self.pan_mode.get() else '')
        return 'break'

    def _draw(self) -> None:
        if not hasattr(self, "canvas"):
            return
        if self._redraw_after is not None:
            self.after_cancel(self._redraw_after)
            self._redraw_after = None
        if self._is_world_map():
            self._draw_world()
            return
        self.pan_button.configure(state='normal')
        self.canvas.delete("all")
        self._canvas_targets.clear()
        data = self._document.data
        width = data["world"]["width"]
        height = data["world"]["height"]
        if not isinstance(width, (int, float)) or not isinstance(height, (int, float)) or width <= 0 or height <= 0:
            self.canvas.create_text(20, 20, anchor="nw", text="世界尺寸无效，请到“场景与规则”修正并校验", fill=COLORS['text'])
            return
        cw, ch = max(self.canvas.winfo_width(), 120), max(self.canvas.winfo_height(), 120)
        fit_scale = min((cw - 64) / width, (ch - 64) / height)
        previous = self._camera_view
        if previous is not None and previous[:2] != (width, height):
            self._stop_zoom()
            self._camera_zoom = self._zoom_target = 1.0
            self._camera_pan = (0.0, 0.0)
        elif previous is not None and previous[2:] != (cw, ch) and (self._camera_zoom != 1 or self._camera_pan != (0.0, 0.0)):
            center = self._world_xy(previous[2] / 2, previous[3] / 2)
            self._camera_pan = ((width / 2-center[0]) * fit_scale * self._camera_zoom,
                                (height / 2-center[1]) * fit_scale * self._camera_zoom)
        self._camera_view = (width, height, cw, ch)
        scale = fit_scale * self._camera_zoom
        self._transform = (scale, (cw - width * scale) / 2 + self._camera_pan[0],
                           (ch - height * scale) / 2 + self._camera_pan[1])
        self._zoom_buttons()
        x0, y0 = self._xy(0, 0)
        x1, y1 = self._xy(width, height)
        self.canvas.create_rectangle(x0, y0, x1, y1, fill=COLORS["map"], outline=COLORS["map_border"])
        self._terrain_image = None
        if self.map_mode.get() == 'terrain':
            # Rendering is bounded by the same world rectangle as coordinates,
            # waypoints and obstacles. It changes no scene geometry or physics.
            vx0, vy0, vx1, vy1 = max(0, x0), max(0, y0), min(cw, x1), min(ch, y1)
            if vx1 > vx0 and vy1 > vy0:
                size = (max(1, round(vx1-vx0)), max(1, round(vy1-vy0)))
                wx, wy = self._world_xy(vx0, vy0)
                viewport = (wx, wy, (vx1-vx0) / scale, (vy1-vy0) / scale)
                self._terrain_image = self._terrain_photo(size, (width, height), viewport)
            if self._terrain_image is not None:
                self.canvas.create_image((vx0+vx1)/2, (vy0+vy1)/2, image=self._terrain_image, tags=('terrain_background',))
                self.canvas.create_rectangle(x0, y0, x1, y1, fill='', outline=COLORS['map_border'])
        if self.grid_visible.get():
            for index in range(1, 20):
                gx, _ = self._xy(width * index / 20, 0)
                _, gy = self._xy(0, height * index / 20)
                if self._terrain_image is not None and index % 2:
                    continue
                color = '#163442' if self._terrain_image is not None else COLORS['grid'] if index % 2 == 0 else '#102632'
                self.canvas.create_line(gx, y0, gx, y1, fill=color, tags=('coordinate_grid',))
                self.canvas.create_line(x0, gy, x1, gy, fill=color, tags=('coordinate_grid',))
        for x, y, sx, sy in ((x0, y0, 1, 1), (x1, y0, -1, 1), (x0, y1, 1, -1), (x1, y1, -1, -1)):
            self.canvas.create_line(x + 18 * sx, y, x, y, x, y + 18 * sy, fill=COLORS['accent'], width=1.5)
        hud_plates = {}
        def hud_text(*args, **kwargs):
            label = self.canvas.create_text(*args, font=self._map_font, **kwargs)
            hud_plates[label] = self.canvas.create_rectangle(0, 0, 0, 0, fill=COLORS['header'], outline='')
            return label
        hud_text(max(8, x0), max(16, y0 - 13), text=f"仿真空间  {width:g} × {height:g}", anchor="w", fill=COLORS["muted"])
        mode_label = {'astra_atlas':'广域山区地图', 'astra_mountain': '山区影像地形', 'detailed_virtual': '精细虚拟地形'}.get(data['world'].get('terrain'), '示意地形') if self._terrain_image is not None else '坐标网格' if self.grid_visible.get() else '仿真坐标'
        hud_text(min(cw-8, x1), max(16, y0 - 13), text=mode_label, anchor='e', fill=COLORS['accent'])
        if cw > 650:
            hud_text(min(cw-8, x1), min(ch-16, y1 + 14), text='X / Y · 仿真单位', anchor='e', fill=COLORS['muted'])
        hud_text(max(8, x0), min(ch-16, y1 + 14), text='红 / 蓝阵营   ○ 任务点   ◇ 出发 / 返航', anchor='w', fill=COLORS['muted'])
        preview = self._drag.get("point") if self._drag else None
        def position(item: dict[str, Any], selection: tuple[Any, ...]) -> tuple[float, float]:
            if preview is not None and self._drag["selection"] == selection:
                return self._xy(*preview)
            return self._xy(item["x"], item["y"])
        def tag(selection: tuple[Any, ...]) -> str:
            value = "target:" + self._selection_key(selection)
            self._canvas_targets[value] = selection
            return value
        occupied = [self.canvas.bbox(item) for item in self.canvas.find_all() if self.canvas.type(item) == 'text']
        def unit_size(unit):
            return 52 if unit['type'] == 'air' else 56
        icon_bounds = []
        for index, unit in enumerate(data['units']):
            px, py = position(unit, ('units', index))
            radius = unit_size(unit) / 2 + 2
            icon_bounds.append((px-radius, py-radius, px+radius, py+radius))

        def object_label(x, y, text, color, tags, *, above=False, emphasis=False, backplate=False):
            if not (-32 <= x <= cw+32 and -32 <= y <= ch+32):
                return
            label = self.canvas.create_text(x + 34, y, text=text, anchor='w', fill=color,
                                            font=self._map_bold_font if emphasis else self._map_font, tags=tags)
            bounds = self.canvas.bbox(label)
            label_width = bounds[2] - bounds[0]
            if label_width > cw - 18:
                self.canvas.itemconfigure(label, width=cw - 20)
                label_width = cw - 18
            offsets = (-36, 0, 30, -60, 54) if above else (0, 30, -36, 54, -60)
            best, best_score = None, float('inf')
            for offset in offsets:
                for dx, anchor in ((34, 'w'), (-34, 'e')):
                    self.canvas.itemconfigure(label, anchor=anchor)
                    self.canvas.coords(label, x + dx, y + offset)
                    b = self.canvas.bbox(label)
                    self.canvas.move(label, max(0, 8 - b[0]) - max(0, b[2] - cw + 8),
                                     max(0, 8 - b[1]) - max(0, b[3] - ch + 8))
                    b = self.canvas.bbox(label)
                    score = sum(max(0, min(b[2]+4, r[2]) - max(b[0]-4, r[0])) *
                                max(0, min(b[3]+3, r[3]) - max(b[1]-3, r[1]))
                                for r in occupied + icon_bounds)
                    if score < best_score:
                        best = (self.canvas.coords(label), anchor, b)
                        best_score = score
                    if not score:
                        break
                if not best_score:
                    break
            coords, anchor, b = best
            self.canvas.itemconfigure(label, anchor=anchor)
            self.canvas.coords(label, *coords)
            occupied.append(b)
            if emphasis or backplate:
                plate = self.canvas.create_rectangle(b[0]-4, b[1]-3, b[2]+4, b[3]+3,
                                                      fill=COLORS['header'],
                                                      outline=COLORS['accent'] if emphasis else '', tags=tags)
                self.canvas.tag_lower(plate, label)

        # Site markers are deliberately quiet; selecting one reveals its label.
        # Units and their labels are drawn above them, at the unchanged position.
        for collection in ('spawn_points', 'return_points'):
            for team in ('red', 'blue'):
                selection = (collection, team)
                x, y = position(data[collection][team], selection)
                selected = selection == self._selected
                color = COLORS[team] if selected else COLORS['site']
                tags = (tag(selection),)
                if collection == 'spawn_points':
                    self.canvas.create_polygon(x, y-12, x+12, y, x, y+12, x-12, y,
                                               fill='', outline=color, width=2 if selected else 1, tags=tags)
                else:
                    self.canvas.create_oval(x-18, y-18, x+18, y+18, outline=color,
                                            width=2 if selected else 1, dash=() if selected else (2, 4), tags=tags)
                if selected:
                    object_label(x, y, ('红队' if team == 'red' else '蓝队') +
                                 ('出发点' if collection == 'spawn_points' else '返航点'),
                                 COLORS[team], tags, above=True)
        for index, item in enumerate(data["obstacles"]):
            selection = ("obstacles", index)
            x, y = position(item, selection)
            selected = selection == self._selected
            tags = (tag(selection),)
            self.canvas.create_rectangle(x, y, x + item["width"] * scale, y + item["height"] * scale,
                                         fill="#183646", outline=COLORS["accent"] if selected else "#527385",
                                         width=3 if selected else 1, tags=tags)
            if x < cw and y < ch and x + item['width'] * scale > 0 and y + item['height'] * scale > 0:
                self.canvas.create_text(max(3, x + 3), max(3, y + 3), anchor="nw", text=item["id"], fill=COLORS['muted'], tags=tags)
        for index, unit in enumerate(data["units"]):
            selection = ("units", index)
            color = COLORS[unit['team']]
            x, y = position(unit, selection)
            focused = self._selected == selection or (self._selected and self._selected[0] == 'waypoints' and self._selected[1] == index)
            radius = unit.get('sensor_range')
            if self._selected == selection and isinstance(radius, (int, float)) and radius > 0:
                screen_radius = radius * scale
                # Keep the visual range overlay out of the ruler and margins.
                # Near a world edge the value remains available in properties.
                if x-screen_radius >= x0 and y-screen_radius >= y0 and x+screen_radius <= x1 and y+screen_radius <= y1:
                    self.canvas.create_oval(x-screen_radius, y-screen_radius, x+screen_radius, y+screen_radius,
                                            outline='#22536A', dash=(4, 5), width=1)
            previous = (x, y)
            for point_index, point in enumerate(unit.get("waypoints", [])):
                point_selection = ("waypoints", index, point_index)
                px, py = position(point, point_selection)
                route_color = color if focused else '#743B49' if unit['team'] == 'red' else '#23566C'
                self.canvas.create_line(*previous, px, py, fill=route_color, dash=(5, 5), width=1.5 if focused else 1)
                selected = point_selection == self._selected
                tags = (tag(point_selection),)
                self.canvas.create_oval(px - 6, py - 6, px + 6, py + 6, fill=COLORS["selected"] if selected else COLORS['map'],
                                        outline=COLORS["accent"] if selected else color, width=3 if selected else 2, tags=tags)
                if (selected or self._selected == selection) and 0 <= px <= cw and 0 <= py <= ch:
                    self.canvas.create_text(px, py - 12, text=str(point_index + 1), fill=color, tags=tags)
                previous = (px, py)
            selected = selection == self._selected
            tags = (tag(selection),)
            draw_tk_unit(self.canvas, (x, y), unit, color, size=unit_size(unit),
                         selected=selected, tags=tags)
            object_label(x, y, unit['id'], color, tags, above=unit['type'] == 'air', emphasis=selected,
                         backplate=self._terrain_image is not None)
        # Text may extend beyond an object near an edge. Keep labels readable
        # using their rendered bounds; entity/point positions and target tags
        # stay intact. Very long identifiers wrap instead of losing characters.
        margin = 8
        for item in self.canvas.find_all():
            if self.canvas.type(item) != "text":
                continue
            bounds = self.canvas.bbox(item)
            if not bounds:
                continue
            if bounds[2] - bounds[0] > cw - 2 * margin:
                self.canvas.itemconfigure(item, width=cw - 2 * margin - 2)
                bounds = self.canvas.bbox(item)
            dx = max(0, margin - bounds[0]) - max(0, bounds[2] - (cw - margin))
            dy = max(0, margin - bounds[1]) - max(0, bounds[3] - (ch - margin))
            self.canvas.move(item, dx, dy)
        for label, plate in hud_plates.items():
            bounds = self.canvas.bbox(label)
            self.canvas.coords(plate, bounds[0]-4, bounds[1]-3, bounds[2]+4, bounds[3]+3)
            self.canvas.tag_raise(plate)
            self.canvas.tag_raise(label)

    def _canvas_motion(self, event: tk.Event) -> None:
        if self._is_world_map():
            lon, lat = self._world_view.to_lonlat((event.x, event.y))
            if not (-180 <= lon <= 180 and -90 <= lat <= 90):
                self.coordinate_var.set('地图外 · 左拖浏览 · 虚拟场景编辑坐标')
                return
            self.coordinate_var.set(f'地图 {format_lonlat(lon, lat)} · 左拖浏览 · 虚拟场景编辑坐标')
            return
        x, y = self._world_xy(event.x, event.y)
        hint = '左键拖动平移 · 滚轮缩放' if self.pan_mode.get() else '拖空白平移 · 拖单位编辑 · Shift 加任务点'
        self.coordinate_var.set(f"X {x:.1f} · Y {y:.1f}  |  {hint}")

    def _canvas_press(self, event: tk.Event) -> None:
        self.canvas.focus_set()
        if self._is_world_map():
            unit_id = self._world_view.pick_unit((event.x, event.y))
            if unit_id is not None:
                for index, unit in enumerate(self._document.data['units']):
                    if unit['id'] == unit_id:
                        if not self.select(('units', index)):
                            return 'break'
                        break
            return self._pan_start(event)
        if self._space_down or self.pan_mode.get():
            return self._pan_start(event)
        self._stop_zoom()
        if event.state & 0x0001:
            x, y = self._world_xy(event.x, event.y)
            world = self._document.data["world"]
            if 0 <= x < world["width"] and 0 <= y < world["height"]:
                self.add_waypoint(round(x, 3), round(y, 3))
            return
        candidates = []
        for item_id in reversed(self.canvas.find_overlapping(event.x - 4, event.y - 4, event.x + 4, event.y + 4)):
            for tag in self.canvas.gettags(item_id):
                if tag in self._canvas_targets:
                    candidate = self._canvas_targets[tag]
                    if candidate not in candidates:
                        candidates.append(candidate)
                    break
        # A list-selected entity stays draggable even when a spawn/return
        # marker or another entity is drawn on top of it.
        selection = self._selected if self._selected in candidates else candidates[0] if candidates else None
        if selection is None:
            return self._pan_start(event)
        if not self.select(selection):
            return
        data = self._document.data
        kind = selection[0]
        if kind in ("units", "obstacles"):
            item = data[kind][selection[1]]
        elif kind == "waypoints":
            item = data["units"][selection[1]]["waypoints"][selection[2]]
        else:
            item = data[kind][selection[1]]
        x, y = self._world_xy(event.x, event.y)
        self._drag = {"selection": selection, "offset": (item["x"] - x, item["y"] - y),
                      "start": (event.x, event.y), "point": None}

    def _canvas_drag(self, event: tk.Event) -> None:
        if self._pan_drag is not None:
            return self._pan_move(event)
        if self._drag is None:
            return
        if abs(event.x - self._drag["start"][0]) + abs(event.y - self._drag["start"][1]) < 3:
            return
        x, y = self._world_xy(event.x, event.y)
        dx, dy = self._drag["offset"]
        self._drag["point"] = (round(x + dx, 3), round(y + dy, 3))
        self.coordinate_var.set(f"移动至 X={x + dx:.1f}，Y={y + dy:.1f}；松开鼠标应用，可撤销")
        self._draw()

    def _canvas_release(self, event: tk.Event) -> None:
        if self._pan_drag is not None:
            return self._pan_end(event)
        drag = self._drag
        self._drag = None
        if not drag or drag["point"] is None:
            return
        selection = drag["selection"]
        x, y = drag["point"]
        kind = selection[0]
        if kind in ("units", "obstacles"):
            self._document.update_item(kind, selection[1], x=x, y=y)
        elif kind == "waypoints":
            self._document.update_waypoint(selection[1], selection[2], x, y)
        else:
            self._document.edit(lambda data: data[kind][selection[1]].update(x=x, y=y))
        self._refresh()
        self._set_status("位置已修改；保存时检查边界，可撤销")
