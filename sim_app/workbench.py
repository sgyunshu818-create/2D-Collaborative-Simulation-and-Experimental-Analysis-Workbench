"""Desktop workbench. Tk and Pygame run in separate processes."""

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
from time import monotonic
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk
from uuid import uuid4

from . import analysis, run_history
from .experiment_results import discover_results, load_results
from .scene_document import SceneDocument
from .scene_editor import SceneEditor
from .tk_runtime import create_root
from .ui_theme import COLORS, UI_FONT, DATA_FONT, CollapsibleFrame, NavigationRail, PanelHeader, apply_theme
from .visual_assets import set_tk_icon


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNS = PROJECT_ROOT / 'artifacts' / 'runs'
TITLE = '二维协同仿真与实验分析工作台'
STATUS = {'READY': '准备', 'RUNNING': '运行中 / 中断保存', 'PAUSED': '暂停 / 中断保存',
          'FINISHED': '结束', 'ERROR': '记录不可读'}
REASONS = {'missions_complete': '所有任务已停止推进（受阻单位仍单独统计）',
           'score_limit': '达到虚构积分上限', 'time_limit': '达到仿真时限'}
UNIT_RESULTS = {'completed': '已完成', 'blocked': '受阻', 'incomplete': '未完成',
                'unassigned': '未分配任务', 'no_task': '未分配任务'}
METRICS = {'duration_s': ('仿真时长', '秒'), 'unit_count': ('单位数', '个'),
           'completed_count': ('任务完成', '个'), 'blocked_count': ('任务受阻', '个'),
           'incomplete_count': ('任务未完成', '个'), 'unassigned_count': ('无任务单位', '个'),
           'distance_total': ('累计路程', '仿真单位'), 'event_count': ('事件数', '条'),
           'snapshot_count': ('快照数', '帧')}


def open_path(path):
    path = Path(path)
    if not path.exists():
        raise OSError(f'文件不存在：{path}')
    if hasattr(os, 'startfile'):
        os.startfile(path)
    else:
        subprocess.Popen(['open' if sys.platform == 'darwin' else 'xdg-open', str(path)])


def display_number(value):
    return f'{value:.3f}' if isinstance(value, float) else str(value)


def local_time(value):
    if not value:
        return '未记录'
    try:
        stamp = datetime.fromisoformat(value.replace('Z', '+00:00')).astimezone()
        offset = stamp.strftime('%z')
        return stamp.strftime('%m-%d %H:%M:%S') + f' UTC{offset[:3]}:{offset[3:]}'
    except (ValueError, TypeError):
        return '时间不可读'


def environment_report():
    try:
        pygame = importlib.metadata.version('pygame')
    except importlib.metadata.PackageNotFoundError:
        pygame = '未安装；python -m pip install -r requirements.txt'
    return {'product': TITLE, 'python': sys.version.split()[0], 'executable': sys.executable,
            'tk': tk.TkVersion, 'pygame': pygame, 'runs_root': str(DEFAULT_RUNS),
            'image_results': [str(p) for p in discover_results(PROJECT_ROOT)],
            'optional_environment': str(PROJECT_ROOT / '.experiment-venv')}


def _tree(parent, columns, widths=None, height=8):
    frame = ttk.Frame(parent)
    tree = ttk.Treeview(frame, columns=columns, show='headings', height=height, selectmode='extended')
    for index, column in enumerate(columns):
        tree.heading(column, text=column)
        tree.column(column, width=widths[index] if widths else 110, minwidth=50)
    y = ttk.Scrollbar(frame, orient='vertical', command=tree.yview)
    x = ttk.Scrollbar(frame, orient='horizontal', command=tree.xview)
    tree.configure(yscrollcommand=y.set, xscrollcommand=x.set)
    tree.grid(row=0, column=0, sticky='nsew')
    y.grid(row=0, column=1, sticky='ns')
    x.grid(row=1, column=0, sticky='ew')
    frame.rowconfigure(0, weight=1)
    frame.columnconfigure(0, weight=1)
    return frame, tree


def _text(parent, height=10, *, technical=False):
    frame = ttk.Frame(parent, style='Panel.TFrame')
    widget = tk.Text(frame, height=height, width=1, wrap='none' if technical else 'word',
                     font=DATA_FONT if technical else 'TkDefaultFont', state='disabled',
                     background=COLORS['surface'], foreground=COLORS['text'],
                     selectbackground=COLORS['selected'], selectforeground=COLORS['text'],
                     relief='flat', borderwidth=0, highlightthickness=0,
                     highlightbackground=COLORS['border'], padx=10, pady=8)
    scroll = ttk.Scrollbar(frame, command=widget.yview)
    widget.configure(yscrollcommand=scroll.set)
    widget.grid(row=0, column=0, sticky='nsew')
    scroll.grid(row=0, column=1, sticky='ns')
    if technical:
        horizontal = ttk.Scrollbar(frame, orient='horizontal', command=widget.xview)
        horizontal.grid(row=1, column=0, sticky='ew')
        widget.configure(xscrollcommand=horizontal.set)
    frame.rowconfigure(0, weight=1)
    frame.columnconfigure(0, weight=1)
    return frame, widget


def _set_text(widget, value):
    widget.configure(state='normal')
    widget.delete('1.0', 'end')
    widget.insert('1.0', value)
    widget.configure(state='disabled')


class Workbench:
    def __init__(self, root, runs_root=DEFAULT_RUNS):
        self.root = root
        self.runs_root = Path(runs_root).resolve()
        self.pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix='workbench-reader')
        self.jobs = []
        self.children = []
        self.history_entries = {}
        self.imported_paths = set()
        self.current_payload = None
        self.current_path = None
        self.comparison_payloads = None
        self.compare_left = self.compare_right = None
        self.chart_summaries = []
        self.loaded_images = None
        self.failure_rows = {}
        self.preview_image = None
        self.selection_generation = 0
        self.closed = False
        self._history_signature = None
        self._last_history_probe = monotonic()
        self._probe_running = False
        self._has_launched = False
        root.title(TITLE)
        # Point-based fonts are already DPI-aware; only the initial workspace
        # extent follows the monitor, so native text keeps room at 150% DPI.
        dpi_scale = max(1.0, root.winfo_fpixels('1i') / 96.0)
        initial_width = min(round(1380 * dpi_scale), max(1040, root.winfo_screenwidth() - 80))
        initial_height = min(round(880 * dpi_scale), max(720, root.winfo_screenheight() - 100))
        root.geometry(f'{initial_width}x{initial_height}')
        root.minsize(1040, 720)
        apply_theme(root)
        set_tk_icon(root)
        self.shell = ttk.Frame(root)
        self.shell.pack(fill='both', expand=True)
        self.shell.rowconfigure(0, weight=1)
        self.shell.columnconfigure(1, weight=1)
        self.navigation = NavigationRail(self.shell, self._select_page, brand_image=root._visual_icon_images[0])
        self.navigation.grid(row=0, column=0, sticky='nsew')
        self.workspace = ttk.Frame(self.shell, padding=(16, 10, 16, 0))
        self.workspace.grid(row=0, column=1, sticky='nsew')
        self.workspace.rowconfigure(1, weight=1)
        self.workspace.columnconfigure(0, weight=1)
        header = ttk.Frame(self.workspace, style='HudHeader.TFrame', padding=(12, 8))
        header.grid(row=0, column=0, sticky='ew')
        self.page_title = tk.StringVar(value='场景编辑')
        self.page_context = tk.StringVar(value='独立场景草稿')
        self.page_title_label = ttk.Label(header, textvariable=self.page_title, style='HudTitle.TLabel')
        self.page_title_label.pack(side='left')
        self.page_context_label = ttk.Label(header, textvariable=self.page_context, style='HeaderMuted.TLabel', width=30)
        self.page_context_label.pack(side='left', padx=14)
        self.process_text = tk.StringVar(value='就绪')
        state_block = ttk.Frame(header, style='Header.TFrame')
        state_block.pack(side='right')
        ttk.Label(state_block, text='运行状态', style='HeaderMuted.TLabel').pack(anchor='w')
        state_row = ttk.Frame(state_block, style='Header.TFrame')
        state_row.pack(anchor='w')
        status_dot = tk.Canvas(state_row, width=10, height=14, background=COLORS['header'], highlightthickness=0)
        status_dot.create_oval(2, 4, 8, 10, fill=COLORS['success'], outline='')
        status_dot.pack(side='left', padx=(0, 5))
        ttk.Label(state_row, textvariable=self.process_text, style='Header.TLabel').pack(side='left')
        self.tabs = ttk.Notebook(self.workspace, style='Shell.TNotebook')
        self.tabs.grid(row=1, column=0, sticky='nsew', pady=(10, 0))
        self.editor = SceneEditor(self.tabs, self.launch_scene)
        self.tabs.add(self.editor, text='场景库与编辑')
        self.history_page = ttk.Frame(self.tabs)
        self.tabs.add(self.history_page, text='历史与分析')
        self.image_page = ttk.Frame(self.tabs)
        self.tabs.add(self.image_page, text='图像实验')
        self.help_page = ttk.Frame(self.tabs)
        self.tabs.add(self.help_page, text='帮助与环境')
        self.status = tk.StringVar(value='准备就绪')
        status_label = ttk.Label(self.workspace, textvariable=self.status, style='Status.TLabel')
        status_label.grid(row=2, column=0, sticky='ew')
        status_label.bind('<Configure>', lambda event: status_label.configure(wraplength=max(400, event.width - 24)))
        self._build_history()
        self._build_images()
        self._build_help()
        self.tabs.bind('<<NotebookTabChanged>>', self._sync_page)
        self.editor._scene_vars['name'].trace_add('write', lambda *_: self._sync_page())
        self._compact_shell = None
        root.bind('<Configure>', self._resize_shell, add='+')
        self._sync_page()
        root.protocol('WM_DELETE_WINDOW', self.close)
        self._after_id = root.after(100, self._tick)
        self.refresh_history()

    def _select_page(self, index):
        self.tabs.select(index)
        self._sync_page()

    def _sync_page(self, *_):
        if not self.tabs.tabs():
            return
        index = self.tabs.index('current')
        self.navigation.select(index)
        self.page_title.set(NavigationRail.LABELS[index])
        context = (self.editor._scene_vars['name'].get(), '运行记录与结果对比', '已有实验结果 · 只读分析', '操作指引与环境信息')[index]
        self.page_context.set(context if len(context) <= 30 else context[:27] + '…')

    def _resize_shell(self, event):
        if event.widget is not self.root:
            return
        compact = event.width < 1180
        if compact != self._compact_shell:
            self._compact_shell = compact
            self.navigation.set_compact(compact)
            self.workspace.configure(padding=(12 if compact else 16, 10, 12 if compact else 16, 0))
            self.page_title_label.configure(style='HudCompactTitle.TLabel' if compact else 'HudTitle.TLabel')
            self.page_context_label.configure(width=22 if compact else 30)

    def _background(self, operation, callback):
        self.jobs.append((self.pool.submit(operation), callback))

    def _error(self, exc):
        self.status.set(str(exc))
        messagebox.showerror('无法完成操作', str(exc), parent=self.root)

    def _tick(self):
        if self.closed:
            return
        pending = []
        for future, callback in self.jobs:
            if future.done():
                try:
                    callback(future.result())
                except Exception as exc:
                    self._error(exc)
            else:
                pending.append((future, callback))
        self.jobs = pending
        ended = []
        active = []
        for child in self.children:
            code = child['process'].poll()
            if code is None:
                active.append(child)
            else:
                ended.append(child)
                child['stream'].close()
                self.status.set(f"{child['label']} 已退出，退出码 {code}；日志：{child['log'].name}")
        self.children = active
        self.process_text.set(f'运行中 · {len(active)} 个窗口' if active else ('运行已结束' if self._has_launched else '就绪'))
        if ended:
            self.refresh_history(select_new=True)
        if monotonic() - self._last_history_probe >= 2 and not self._probe_running:
            self._last_history_probe = monotonic()
            self._probe_running = True
            def probed(signature):
                self._probe_running = False
                previous, self._history_signature = self._history_signature, signature
                if previous is not None and previous != signature:
                    self.refresh_history(select_new=True)
            self._background(self._history_files_signature, probed)
        self._after_id = self.root.after(100, self._tick)

    def _history_files_signature(self):
        result = []
        if self.runs_root.is_dir():
            for path in self.runs_root.glob('*/run.json'):
                try:
                    stat = path.stat()
                    result.append((str(path), stat.st_mtime_ns, stat.st_size))
                except OSError:
                    continue
        return tuple(sorted(result))

    def _launch(self, arguments, label):
        log_dir = PROJECT_ROOT / 'artifacts' / 'workbench_logs'
        log_dir.mkdir(parents=True, exist_ok=True)
        log = log_dir / f'{datetime.now():%Y%m%d_%H%M%S}_{uuid4().hex[:6]}.log'
        stream = log.open('w', encoding='utf-8')
        try:
            process = subprocess.Popen([sys.executable, '-m', 'sim_app', *map(str, arguments)],
                                       cwd=PROJECT_ROOT, stdout=stream, stderr=subprocess.STDOUT)
        except OSError:
            stream.close()
            raise
        self.children.append({'process': process, 'stream': stream, 'log': log, 'label': label})
        self._has_launched = True
        self.status.set(f'{label} 已启动独立窗口；运行后点击结束/退出保存，再到历史查看。')

    def launch_scene(self, path):
        try:
            # Freeze the exact input before starting the child. Later saves cannot race its read.
            payload = json.loads(Path(path).read_text(encoding='utf-8'))
            input_path = PROJECT_ROOT / 'configs' / 'user_scenes' / 'run_inputs' / f'input_{uuid4().hex}.json'
            SceneDocument(payload, source_path=path).save(input_path)
            origin = self.editor.get_document().source_path
            input_path.with_suffix('.origin.json').write_text(json.dumps(
                {'saved_scene': str(Path(path).resolve()), 'source': str(origin) if origin else None},
                ensure_ascii=False, indent=2), encoding='utf-8')
            self._launch(['--scene', input_path, '--output-dir', self.runs_root], '场景运行')
        except Exception as exc:
            self._error(exc)

    def _build_history(self):
        tools = ttk.Frame(self.history_page, padding=(0, 0, 0, 8))
        tools.pack(fill='x')
        groups = (('记录', [('刷新', self.refresh_history), ('导入…', self.import_run)]),
                  ('回放', [('回放记录', self.replay_selected), ('复制输入新建', self.copy_input)]),
                  ('分析', [('比较两条', self.compare_selected), ('导出汇总 CSV', self.export_selected)]))
        for index, (label, actions) in enumerate(groups):
            if index:
                ttk.Separator(tools, orient='vertical').pack(side='left', fill='y', padx=10, pady=2)
            ttk.Label(tools, text=label, style='Muted.TLabel').pack(side='left', padx=(0, 5))
            for title, command in actions:
                ttk.Button(tools, text=title, command=command).pack(side='left', padx=(0, 4))
        more = ttk.Menubutton(tools, text='更多 ▾', direction='below')
        more.pack(side='right')
        menu = tk.Menu(more, tearoff=False)
        menu.add_command(label='选择记录目录…', command=self.choose_run_root)
        menu.add_command(label='查看记录目录路径…', command=self.show_history_path)
        menu.add_separator()
        menu.add_command(label='打开所选记录数据目录', command=self.open_run_directory)
        menu.add_command(label='编辑所选记录备注…', command=self.edit_note)
        more.configure(menu=menu)
        pane = self.history_pane = ttk.Panedwindow(self.history_page, orient='horizontal')
        pane.pack(fill='both', expand=True)
        left = ttk.Frame(pane, style='HudPanel.TFrame', width=330)
        pane.add(left, weight=3)
        PanelHeader(left, '运行记录').pack(fill='x')
        filters = ttk.Frame(left, style='Panel.TFrame', padding=(8, 7))
        filters.pack(fill='x')
        self.history_search = tk.StringVar()
        ttk.Label(filters, text='筛选场景 / 编号 / 备注', style='PanelMuted.TLabel').pack(anchor='w', pady=(0, 3))
        ttk.Entry(filters, textvariable=self.history_search, width=16).pack(fill='x')
        self.history_search.trace_add('write', lambda *_: self._render_history())
        location = ttk.Frame(left, style='Panel.TFrame', padding=(8, 0, 8, 6))
        location.pack(fill='x')
        self.history_location = tk.StringVar(value=self._directory_label())
        ttk.Label(location, textvariable=self.history_location, style='PanelMuted.TLabel').pack(side='left', fill='x', expand=True)
        ttk.Button(location, text='目录…', command=self.choose_run_root).pack(side='right', padx=(6, 0))
        frame, self.history_tree = _tree(left, ('场景', '状态', '秒数', '时间（本地）', '备注'), (125, 65, 60, 150, 140), 18)
        frame.pack(fill='both', expand=True)
        self.history_tree.bind('<<TreeviewSelect>>', self._history_selection)
        ttk.Label(left, text='Ctrl 多选比较 · 备注向右滚动查看', style='PanelMuted.TLabel', padding=(8, 5)).pack(fill='x')
        right = ttk.Frame(pane, style='HudPanel.TFrame')
        pane.add(right, weight=7)
        self.detail_tabs = ttk.Notebook(right)
        self.detail_tabs.pack(fill='both', expand=True)
        summary_page = ttk.Frame(self.detail_tabs, padding=8, style='Panel.TFrame')
        self.detail_tabs.add(summary_page, text='结果摘要')
        frame, self.result_text = _text(summary_page, 16)
        frame.pack(fill='both', expand=True)
        comparison_page = ttk.Frame(self.detail_tabs, padding=8, style='Panel.TFrame')
        self.detail_tabs.add(comparison_page, text='输入与指标比较')
        compare_controls = ttk.Frame(comparison_page, style='Panel.TFrame')
        compare_controls.pack(fill='x', pady=(0, 6))
        ttk.Button(compare_controls, text='设为左侧', command=lambda: self.assign_compare('left')).pack(side='left', padx=(0, 4))
        ttk.Button(compare_controls, text='设为右侧', command=lambda: self.assign_compare('right')).pack(side='left', padx=(0, 4))
        ttk.Button(compare_controls, text='比较两条', command=self.compare_selected).pack(side='left')
        self.compare_selection_text = tk.StringVar(value='Ctrl 选择两条；也可分别设置左侧与右侧。差值 = 右 − 左。')
        compare_hint = ttk.Label(comparison_page, textvariable=self.compare_selection_text, style='PanelMuted.TLabel')
        compare_hint.pack(fill='x', pady=(0, 8))
        comparison_page.bind('<Configure>', lambda event: compare_hint.configure(wraplength=max(240, event.width - 16)))
        frame, self.compare_text = _text(comparison_page, 18)
        frame.pack(fill='both', expand=True)
        chart_page = ttk.Frame(self.detail_tabs, padding=8, style='Panel.TFrame')
        self.detail_tabs.add(chart_page, text='时间曲线')
        self.chart_metric = tk.StringVar(value='累计路程')
        combo = ttk.Combobox(chart_page, textvariable=self.chart_metric, values=('累计路程', '虚构积分', '联系人数量'), state='readonly', width=18)
        combo.pack(anchor='w')
        combo.bind('<<ComboboxSelected>>', lambda *_: self.draw_chart())
        self.chart = tk.Canvas(chart_page, background=COLORS['surface'], highlightthickness=0)
        self.chart.pack(fill='both', expand=True)
        self.chart.bind('<Configure>', lambda *_: self.draw_chart())
        event_page = ttk.Frame(self.detail_tabs, padding=8, style='Panel.TFrame')
        self.detail_tabs.add(event_page, text='事件与回放定位')
        event_tools = ttk.Frame(event_page, style='Panel.TFrame')
        event_tools.pack(fill='x')
        event_tools.columnconfigure(1, weight=1)
        self.event_filter = tk.StringVar()
        ttk.Label(event_tools, text='类型 / 单位', style='Panel.TLabel').grid(row=0, column=0, padx=(0, 6))
        ttk.Entry(event_tools, textvariable=self.event_filter, width=15).grid(row=0, column=1, sticky='ew')
        self.event_filter.trace_add('write', lambda *_: self._render_events())
        ttk.Button(event_tools, text='回放到所选事件', command=self.replay_event).grid(row=0, column=2, padx=(8, 0))
        frame, self.event_tree = _tree(event_page, ('秒数', '类型', '单位', '详情'), (65, 130, 145, 370), 11)
        frame.pack(fill='both', expand=True, pady=(8, 6))
        self.event_details = CollapsibleFrame(event_page, '技术详情 · 原始事件 JSON')
        self.event_details.pack(fill='x')
        frame, self.event_text = _text(self.event_details.body, 6, technical=True)
        frame.pack(fill='both', expand=True)
        self.event_tree.bind('<<TreeviewSelect>>', self._event_selection)
        _set_text(self.result_text, '选择一条记录查看结果。\n\nCtrl 选择两条后点击“比较两条”；输入差异、指标和曲线使用相同计算口径。\n\n运行保存后历史自动刷新。')
        _set_text(self.event_text, '选择事件后可查看完整原始数据。')
        self._history_pane_ready = False
        def arrange(event):
            if not self._history_pane_ready and event.width > 500:
                self._history_pane_ready = True
                pane.sashpos(0, round(event.width * .30))
        pane.bind('<Configure>', arrange)

    def _directory_label(self):
        name = self.runs_root.name
        return '目录：' + (name if len(name) <= 22 else name[:10] + '…' + name[-10:])

    def show_history_path(self):
        messagebox.showinfo('记录目录', str(self.runs_root), parent=self.root)

    def choose_run_root(self):
        path = filedialog.askdirectory(title='选择含运行记录的目录', initialdir=self.runs_root)
        if path:
            self.runs_root = Path(path)
            self.history_location.set(self._directory_label())
            self.refresh_history()

    def import_run(self):
        path = filedialog.askopenfilename(title='打开历史 run.json', filetypes=[('运行记录', '*.json')], initialdir=self.runs_root)
        if path:
            self.imported_paths.add(Path(path).resolve())
            self.refresh_history()

    def refresh_history(self, select_new=False):
        self.status.set('正在后台读取历史索引…')
        known = set(self.history_entries)
        def read():
            entries = run_history.list_runs(self.runs_root)
            for path in self.imported_paths:
                entries.extend(item for item in run_history.list_runs(path.parent) if Path(item['path']).resolve() == path)
            return entries
        def loaded(entries):
            self.history_entries = {str(Path(entry['path']).resolve()): entry for entry in entries}
            self._render_history()
            self.status.set(f'历史记录 {len(self.history_entries)} 条 · 目录：{self.runs_root.name}')
            candidates = [path for path in self.history_entries if path not in known]
            if select_new and candidates:
                path = max(candidates, key=lambda p: self.history_entries[p].get('created_at') or '')
                if self.history_tree.exists(path):
                    self.history_tree.selection_set(path)
                    self.history_tree.see(path)
                self.tabs.select(self.history_page)
        self._background(read, loaded)

    def _render_history(self):
        selected = self.history_tree.selection()
        self.history_tree.delete(*self.history_tree.get_children())
        query = self.history_search.get().casefold()
        for path, entry in self.history_entries.items():
            if query and query not in f"{entry.get('scene_name')} {entry.get('run_id')} {entry.get('note')} {path}".casefold():
                continue
            state = entry.get('state', entry.get('status', 'ERROR'))
            duration = entry.get('duration_s', entry.get('duration'))
            self.history_tree.insert('', 'end', iid=path, values=(entry.get('scene_name', ''),
                STATUS.get(state, state), f'{duration:.3f}' if isinstance(duration, (float, int)) else '—',
                local_time(entry.get('created_at')).split(' UTC')[0], entry.get('note', '')))
        restored = [path for path in selected if self.history_tree.exists(path)]
        if restored:
            self.history_tree.selection_set(restored)

    def _selected_path(self):
        paths = self.history_tree.selection()
        if not paths:
            raise ValueError('请先选择一条历史记录。')
        return Path(paths[0])

    def _history_selection(self, *_):
        paths = self.history_tree.selection()
        if len(paths) != 1:
            return
        path = Path(paths[0])
        entry = self.history_entries[str(path)]
        self.selection_generation += 1
        generation = self.selection_generation
        if entry.get('error'):
            self.current_payload = None
            _set_text(self.result_text, entry['error'])
            return
        self.status.set('正在后台加载所选记录和指标…')
        def read():
            payload = run_history.load_run(path)
            return payload, analysis.summarize_run(payload)
        def loaded(value):
            if generation != self.selection_generation:
                return
            payload, summary = value
            self.current_path, self.current_payload = path, payload
            lines = [f"场景：{payload['scene']['name']}",
                     f"记录状态：{STATUS.get(summary['state'], summary['state'])}",
                     f"时间：{local_time(entry.get('created_at'))}；{entry.get('time_label', '记录时间')}", '']
            lines += [f'{title}：{display_number(summary.get(key, "—"))} {unit}' for key, (title, unit) in METRICS.items()]
            lines += [f"结束原因：{REASONS.get(summary['finish_reason'], summary['finish_reason'] or '尚未正常结束 / 未记录')}"]
            quality = summary['data_quality']
            lines += ['', '数据完整性：' + ('完整' if quality['status'] == 'complete' else '存在缺失或说明'),
                      *quality.get('issues', []), '', '单位结果：']
            lines += [f"{unit['id']} · {UNIT_RESULTS.get(unit['status'], unit['status'])} · 航点 {unit['waypoints_reached']}/{unit['waypoints_total']} · 路程 {unit['distance_travelled']:.3f}" for unit in summary['per_unit']]
            lines += ['', '指标口径：']
            for key, (title, _) in METRICS.items():
                definition = summary.get('definitions', {}).get(key, {})
                if definition:
                    explanation = definition.get('definition', '')
                    for original, replacement in {'distance_travelled': '累计路程', 'STOPPED': '停止', 'BLOCKED': '受阻', 'events': '事件', 'snapshots': '快照'}.items():
                        explanation = explanation.replace(original, replacement)
                    lines.append(f"{title}（{definition.get('unit', '')}）：{explanation}")
            lines += ['', '记录来源：', f'运行文件：{path}',
                      f"运行编号：{summary.get('metadata', {}).get('run_id', '旧记录 / 未记录')}"]
            _set_text(self.result_text, '\n'.join(lines))
            self.chart_summaries = [(payload['scene']['name'], summary)]
            self.comparison_payloads = None
            self.draw_chart()
            self._render_events()
            self.status.set(f'已载入 {path.name}；结果与导出使用同一指标函数。')
        self._background(read, loaded)

    def replay_selected(self):
        try:
            path = self._selected_path()
            if self.history_entries[str(path)].get('error'):
                raise ValueError(self.history_entries[str(path)]['error'])
            self._launch(['--replay', path], '记录回放')
        except Exception as exc:
            self._error(exc)

    def copy_input(self):
        try:
            path = self._selected_path()
            def loaded(payload):
                if self.editor.load_document(payload['scene'], source_path=path):
                    self.tabs.select(self.editor)
                    self.status.set(f'已复制历史输入；另存后运行新实验。来源：{path}')
            self._background(lambda: run_history.load_run(path), loaded)
        except Exception as exc:
            self._error(exc)

    def compare_selected(self):
        selected = self.history_tree.selection()
        paths = selected if len(selected) == 2 else (self.compare_left, self.compare_right)
        if not all(paths) or paths[0] == paths[1]:
            self._error(ValueError('分别选择记录并设为左侧/右侧，或按 Ctrl 选择两条。差值方向为右减左。'))
            return
        self.selection_generation += 1
        self.status.set('正在后台比较输入和指标…')
        def read():
            left, right = (run_history.load_run(path) for path in paths)
            return left, right, analysis.compare_runs(left, right)
        def loaded(value):
            left, right, comparison = value
            self.comparison_payloads = (left, right)
            self.comparison_paths = paths
            lines = [f'左：{paths[0]}', f'右：{paths[1]}', '差值方向：右 − 左', '', '输入差异（首先检查改了什么）：']
            differences = comparison['input_differences']
            lines += [f"{item['path']} [{item['kind']}]\n  左 {json.dumps(item['left'], ensure_ascii=False)}\n  右 {json.dumps(item['right'], ensure_ascii=False)}" for item in differences] if differences else ['输入内容相同。']
            lines += ['', '结果指标差异：']
            for key, item in comparison['metric_differences'].items():
                title, unit = METRICS.get(key, (key, ''))
                if key == 'state':
                    title, left_value, right_value = '记录状态', STATUS.get(item['left'], item['left']), STATUS.get(item['right'], item['right'])
                elif key == 'finish_reason':
                    title, left_value, right_value = '结束原因', REASONS.get(item['left'], item['left'] or '未结束'), REASONS.get(item['right'], item['right'] or '未结束')
                else:
                    left_value, right_value = display_number(item['left']), display_number(item['right'])
                delta = display_number(item['delta']) if item.get('delta') is not None else '不适用'
                lines.append(f"{title}: 左 {left_value} → 右 {right_value}；差值 {delta} {unit}")
            lines += ['', '两条曲线使用各自时间轴；短运行不会补齐到长运行。']
            _set_text(self.compare_text, '\n'.join(lines))
            self.chart_summaries = [(f"左 {left['scene']['name']}", comparison['left']),
                                    (f"右 {right['scene']['name']}", comparison['right'])]
            self.draw_chart()
            self.detail_tabs.select(1)
            self.status.set('输入与指标比较完成；导出汇总 CSV 可导出同一比较结果。')
        self._background(read, loaded)

    def assign_compare(self, side):
        try:
            path = str(self._selected_path())
            if side == 'left':
                self.compare_left = path
            else:
                self.compare_right = path
            self.compare_selection_text.set(f"对比左侧：{Path(self.compare_left).parent.name if self.compare_left else '未选择'}；右侧：{Path(self.compare_right).parent.name if self.compare_right else '未选择'}。差值 = 右 − 左。")
        except Exception as exc:
            self._error(exc)

    def draw_chart(self):
        self.chart.delete('all')
        if not self.chart_summaries:
            self.chart.create_text(25, 25, anchor='nw', text='选择运行或比较两条后查看曲线。', fill=COLORS['muted'])
            return
        key, unit = {'累计路程': ('distance', '仿真单位'), '虚构积分': ('score', '虚构分'), '联系人数量': ('contact', '个')}.get(self.chart_metric.get(), ('distance', '仿真单位'))
        width, height = max(380, self.chart.winfo_width()), max(260, self.chart.winfo_height())
        points = [summary['series'].get(key, []) for _, summary in self.chart_summaries]
        numeric = [(p['time'], p['value']) for series in points for p in series if isinstance(p.get('value'), (float, int))]
        if not numeric:
            self.chart.create_text(25, 25, anchor='nw', text='记录缺少可用时间序列。', fill=COLORS['muted'])
            return
        max_x = max(1e-9, max(x for x, _ in numeric))
        max_y = max(1.0, max(y for _, y in numeric))
        left, top, right, bottom = 65, 60, width-25, height-55
        for i in range(5):
            y = bottom - (bottom-top)*i/4
            self.chart.create_line(left, y, right, y, fill=COLORS['grid'])
            self.chart.create_text(left-8, y, anchor='e', text=f'{max_y*i/4:.2g}', fill=COLORS['muted'])
        self.chart.create_line(left, top, left, bottom, right, bottom, fill=COLORS['map_border'])
        self.chart.create_text(left, 15, anchor='nw', text=f'{self.chart_metric.get()} / {unit}', fill=COLORS['text'])
        self.chart.create_text(right, bottom+30, anchor='e', text=f'仿真时间 / 秒（0 → {max_x:.3f}）', fill=COLORS['muted'])
        for index, ((label, _), series) in enumerate(zip(self.chart_summaries, points)):
            color = (COLORS['blue'], COLORS['red'])[index % 2]
            self.chart.create_text(left, 36+index*18, anchor='w', text=label, fill=color)
            coords = []
            # Preserve turning points visually while avoiding hundreds of thousands of canvas vertices.
            stride = max(1, len(series)//2000)
            sampled = series[::stride]
            if series and sampled[-1] is not series[-1]:
                sampled.append(series[-1])
            for point in sampled:
                if isinstance(point.get('value'), (int, float)):
                    coords.extend((left+(right-left)*point['time']/max_x, bottom-(bottom-top)*point['value']/max_y))
            if len(coords) >= 4:
                self.chart.create_line(*coords, fill=color, width=2)

    def _render_events(self):
        self.event_tree.delete(*self.event_tree.get_children())
        if not self.current_payload:
            return
        query = self.event_filter.get().casefold()
        for index, event in enumerate(self.current_payload.get('events', [])):
            encoded = json.dumps(event, ensure_ascii=False)
            if query and query not in encoded.casefold():
                continue
            self.event_tree.insert('', 'end', iid=str(index), values=(f"{event['time']:.3f}", event['kind'], event.get('unit_id') or '系统', event.get('message', json.dumps(event.get('details', {}), ensure_ascii=False))))

    def _event_selection(self, *_):
        selection = self.event_tree.selection()
        if selection and self.current_payload:
            _set_text(self.event_text, json.dumps(self.current_payload['events'][int(selection[0])], ensure_ascii=False, indent=2))

    def replay_event(self):
        try:
            selection = self.event_tree.selection()
            if not selection or not self.current_payload or not self.current_path:
                raise ValueError('选择一条记录及其中的事件。')
            self._launch(['--replay', self.current_path, '--replay-event', selection[0]], '事件定位回放')
        except Exception as exc:
            self._error(exc)

    def export_selected(self):
        try:
            path = self._selected_path()
            target = filedialog.asksaveasfilename(title='导出统一指标', defaultextension='.csv', filetypes=[('CSV', '*.csv')], initialfile='comparison.csv' if self.comparison_payloads else 'metrics.csv')
            if not target:
                return
            if self.comparison_payloads:
                left, right = self.comparison_payloads
                operation = lambda: analysis.export_comparison(left, right, Path(target), left_source_path=self.comparison_paths[0], right_source_path=self.comparison_paths[1])
            else:
                operation = lambda: analysis.export_metrics(run_history.load_run(path), Path(target), source_path=path)
            self._background(operation, lambda result: self.status.set(f'已导出：{result}'))
        except Exception as exc:
            self._error(exc)

    def open_run_directory(self):
        try:
            open_path(self._selected_path().parent)
        except Exception as exc:
            self._error(exc)

    def edit_note(self):
        try:
            path = self._selected_path()
            note = simpledialog.askstring('运行备注', '备注保存为独立文件，原始运行记录保持完整。', initialvalue=self.history_entries[str(path)].get('note', ''), parent=self.root)
            if note is not None:
                run_history.save_note(path, note)
                self.refresh_history()
        except Exception as exc:
            self._error(exc)

    def _build_images(self):
        toolbar = ttk.Frame(self.image_page, padding=(0, 0, 0, 6))
        toolbar.pack(fill='x')
        ttk.Label(toolbar, text='结果', style='Muted.TLabel').pack(side='left', padx=(0, 6))
        paths = [str(p) for p in discover_results(PROJECT_ROOT)]
        self.image_directory = tk.StringVar(value=paths[0] if paths else '')
        ttk.Combobox(toolbar, textvariable=self.image_directory, values=paths, width=25).pack(side='left', fill='x', expand=True, padx=(0, 6))
        ttk.Button(toolbar, text='选择结果目录', command=self.choose_image_directory).pack(side='left', padx=4)
        ttk.Button(toolbar, text='载入已有结果', command=self.load_images).pack(side='left', padx=4)
        self.image_info = tk.StringVar(value='只读浏览已有实验；主环境无需安装图像实验依赖。')
        info = ttk.Label(self.image_page, textvariable=self.image_info, style='Muted.TLabel')
        info.pack(fill='x', pady=(0, 8))
        info.bind('<Configure>', lambda event: info.configure(wraplength=max(300, event.width)))
        PanelHeader(self.image_page, '实验指标').pack(fill='x')
        frame, self.image_metrics = _tree(self.image_page, ('条件', '方法', '独立原图', '变体数', '调用数', '准确率', '平均 ms'), (150, 120, 90, 90, 90, 90, 110), 6)
        frame.pack(fill='x')
        filters = ttk.Frame(self.image_page)
        filters.pack(fill='x', pady=6)
        self.failure_filter = tk.StringVar()
        ttk.Label(filters, text='失败样本筛选', style='Muted.TLabel').pack(side='left')
        ttk.Entry(filters, textvariable=self.failure_filter, width=20).pack(side='left', fill='x', expand=True, padx=6)
        self.failure_filter.trace_add('write', lambda *_: self._render_failures())
        ttk.Button(filters, text='打开所选原图', command=self.open_failure_image).pack(side='left', padx=5)
        ttk.Button(filters, text='打开原始预测 CSV', command=self.open_predictions).pack(side='left', padx=5)
        pane = ttk.Panedwindow(self.image_page, orient='horizontal')
        pane.pack(fill='both', expand=True)
        frame, self.failure_tree = _tree(pane, ('样本 ID', '方法', '条件', '真实', '预测', '轮次'), (240, 100, 130, 90, 90, 50), 10)
        pane.add(frame, weight=3)
        preview = ttk.Frame(pane, style='HudPanel.TFrame')
        pane.add(preview, weight=2)
        PanelHeader(preview, '样本预览与预测').pack(fill='x')
        self.image_preview = ttk.Label(preview, text='选择失败样本查看原图与预测。', anchor='center', style='PanelMuted.TLabel')
        self.image_preview.pack(fill='x', pady=4)
        self.image_preview.bind('<Configure>', lambda event: self.image_preview.configure(wraplength=max(160, event.width - 16)))
        frame, self.failure_detail = _text(preview, 11, technical=True)
        frame.pack(fill='both', expand=True)
        self.failure_tree.bind('<<TreeviewSelect>>', self._failure_selection)

    def choose_image_directory(self):
        directory = filedialog.askdirectory(title='选择包含 predictions.csv 的结果目录', initialdir=PROJECT_ROOT / 'artifacts')
        if directory:
            self.image_directory.set(directory)
            self.load_images()

    def load_images(self):
        directory = self.image_directory.get()
        if not directory:
            self._error(ValueError('请选择已有图像实验结果目录。'))
            return
        self.status.set('正在后台读取预测和样本清单…')
        def loaded(result):
            self.loaded_images = result
            self.image_metrics.delete(*self.image_metrics.get_children())
            for index, item in enumerate(result['metrics']):
                self.image_metrics.insert('', 'end', iid=str(index), values=(item['condition'], item['algorithm'], item['independent_originals'], item['image_count'], item['call_count'], f"{item['accuracy']:.2%}", f"{item['mean_ms']:.6f}"))
            self.image_info.set(f"{result['independent_originals']} 独立测试原图 · {result['image_count']} 条件变体 · {result['round_count']} 轮 · {len(result['rows'])} 次预测 · {len(result['failures'])} 次失败；{result['timing_definition']}")
            self._render_failures()
            self.status.set('已载入已有结果；选择失败样本可查看原图与原始预测。')
        self._background(lambda: load_results(directory), loaded)

    def _render_failures(self):
        self.failure_tree.delete(*self.failure_tree.get_children())
        self.failure_rows = {}
        if not self.loaded_images:
            return
        query = self.failure_filter.get().casefold()
        for index, row in enumerate(self.loaded_images['failures']):
            if query and query not in f"{row['image_id']} {row['condition']} {row['algorithm']} {row['label']} {row['prediction']}".casefold():
                continue
            self.failure_rows[str(index)] = row
            self.failure_tree.insert('', 'end', iid=str(index), values=(row['image_id'], row['algorithm'], row['condition'], row['label'], row['prediction'], row.get('round', '1')))

    def _failure_selection(self, *_):
        selection = self.failure_tree.selection()
        if not selection:
            return
        row = self.failure_rows[selection[0]]
        _set_text(self.failure_detail, f"原始预测 CSV 第 {row['csv_line']} 行\n" + json.dumps(row, ensure_ascii=False, indent=2) + '\n\n混淆矩阵（行=真实标签，列=预测）：\n' + json.dumps(next(m['confusion'] for m in self.loaded_images['metrics'] if m['algorithm'] == row['algorithm'] and m['condition'] == row['condition']), ensure_ascii=False, indent=2))
        self.preview_image = None
        try:
            if not row.get('image_path'):
                raise tk.TclError('没有找到原样本清单，可选择首次结果目录。')
            image = tk.PhotoImage(data=base64.b64encode(Path(row['image_path']).read_bytes()))
            factor = max(1, image.width()//340, image.height()//220)
            self.preview_image = image.subsample(factor)
            self.image_preview.configure(image=self.preview_image, text='')
        except (tk.TclError, OSError) as exc:
            self.image_preview.configure(image='', text=f'预览不可用：{exc}\n仍可打开原图文件。')

    def open_failure_image(self):
        try:
            selection = self.failure_tree.selection()
            if not selection or not self.failure_rows[selection[0]].get('image_path'):
                raise ValueError('先选择存在原图路径的失败样本。')
            open_path(self.failure_rows[selection[0]]['image_path'])
        except Exception as exc:
            self._error(exc)

    def open_predictions(self):
        try:
            if not self.loaded_images:
                raise ValueError('先载入已有结果。')
            open_path(self.loaded_images['predictions_path'])
        except Exception as exc:
            self._error(exc)

    def _build_help(self):
        toolbar = ttk.Frame(self.help_page, padding=(0, 0, 0, 8))
        toolbar.pack(fill='x')
        ttk.Label(toolbar, text='使用指南', style='AppTitle.TLabel').pack(side='left')
        ttk.Button(toolbar, text='打开快速开始文档', command=lambda: open_path(PROJECT_ROOT / 'docs' / '工作台快速开始.md')).pack(side='right')
        text = ('编辑场景\n'
                '选择地理、基础、障碍或共享模板，或打开已有场景。模板始终保存为独立副本。\n'
                '从列表或地图选择对象，修改后点击“应用修改”。保存、运行和切换对象前会自动应用表单；运行前校验全部字段。\n'
                '拖动可移动对象；选中实体后点击“+ 任务点”，或按 Shift 在地图单击。任务点可改坐标、上移、下移或删除；重叠对象可从左侧列表选择。\n'
                '地图的地形与网格可独立开关。世界地图采用真实 Natural Earth 离线地形，虚拟场景可显示山区影像或示意底图。\n'
                '“全球”浏览完整地球，“定位场景”查看已有地理参考的区域，“展开地图”收起两侧信息。世界地图滚轮缩放1–256倍，左拖浏览；点击单位选择，编辑坐标请切回虚拟场景。\n'
                '“场景与规则”的地理参考可设置中心经纬度与每仿真单位的米数，随场景与记录保存。全球底图为小比例尺数据；通行和任务继续由场景实体与障碍决定。\n'
                '在地图内滚轮可按光标位置缩放，＋/− 以视口中心缩放；100% 表示适配。大型精细场景支持 100%–6400%，原小场景支持 100%–600%。拖空白处平移；开启“平移模式”后可在任意位置左拖。中/右键、Space 左拖也可平移，点击“适应”恢复全场景。\n'
                '速度与观测范围留空会沿用默认值。坐标、范围和速度均使用虚构仿真单位；出生与返航点、高级参数在“场景与规则”中展开。\n\n'
                '运行仿真\n'
                '“保存并运行”打开独立窗口。Enter 开始，Space 暂停 / 继续；退出会保存已开始的运行。单位列表支持搜索和滚动，可检查地图缩放、编号 / 轨迹图层与联系人。\n\n'
                '历史与比较\n'
                '运行保存后历史自动刷新。选择一条记录查看摘要、曲线与事件；展开“技术详情”查看原始事件。回放读取历史，复制输入可开始新实验。\n'
                '按 Ctrl 选择两条并比较，也可在比较页分别设置左侧与右侧。先看输入差异，再看指标和各自时间曲线；CSV 使用同一计算口径。目录、备注入口位于“更多”。\n\n'
                '图像实验\n'
                '选择已有结果目录并载入，筛选失败样本，查看原图、预测和混淆矩阵。实验执行使用独立环境与 README 命令；图像页提供只读分析。')
        frame, widget = _text(self.help_page, 22)
        frame.pack(fill='both', expand=True)
        _set_text(widget, text)
        widget.tag_configure('section', font=(UI_FONT[0], 10, 'bold'), foreground=COLORS['accent'], spacing1=3, spacing3=5)
        for title in ('编辑场景', '运行仿真', '历史与比较', '图像实验'):
            start = widget.search(title, '1.0', stopindex='end')
            if start:
                widget.tag_add('section', start, f'{start}+{len(title)}c')
        environment = CollapsibleFrame(self.help_page, '环境与结果位置')
        environment.pack(fill='x', pady=(8, 0))
        frame, details = _text(environment.body, 6, technical=True)
        frame.pack(fill='both', expand=True)
        _set_text(details, '普通运行：artifacts/runs\n编辑副本：configs/user_scenes\n窗口日志：artifacts/workbench_logs\n\n'
                  '完全虚构的二维教学规则；长记录完整载入内存。历史阶段冻结、报告 / PPT / 图像数据保留为旧版证据。\n'
                  '跨电脑、600 秒长 GUI 性能与完整无障碍验证尚未完成。\n\n' + json.dumps(environment_report(), ensure_ascii=False, indent=2))

    def close(self):
        if not self.editor.can_close():
            return
        if self.children and not messagebox.askyesno('运行窗口仍打开', '关闭工作台后运行窗口仍可操作并保存。是否关闭工作台？', parent=self.root):
            return
        self.closed = True
        self.root.after_cancel(self._after_id)
        self.pool.shutdown(wait=False, cancel_futures=True)
        self.root.destroy()


def main():
    parser = argparse.ArgumentParser(description=TITLE)
    parser.add_argument('--runs-root', type=Path, default=DEFAULT_RUNS)
    parser.add_argument('--scene', type=Path, default=PROJECT_ROOT / 'configs' / 'detailed_scene.json',
                        help='启动时载入的场景；默认大型精细虚拟场景')
    parser.add_argument('--check', action='store_true', help='检查环境与结果入口，不创建窗口')
    args = parser.parse_args()
    if args.check:
        print(json.dumps(environment_report(), ensure_ascii=False, indent=2))
        return 0
    try:
        root = create_root()
        app = Workbench(root, args.runs_root)
        try:
            payload = json.loads(args.scene.read_text(encoding='utf-8-sig'))
            app.editor.load_document(payload, source_path=args.scene)
            if args.scene.resolve() == (PROJECT_ROOT / 'configs/detailed_scene.json').resolve():
                app.editor.template_var.set('大型精细场景')
                def open_world():
                    app.editor.toggle_map_expanded()
                root.after(100, open_world)
        except (OSError, ValueError) as error:
            app.status.set(f'启动场景未载入：{error}；可从场景库重新选择')
        root.mainloop()
    except tk.TclError as exc:
        print(f'WORKBENCH_ERROR: {exc}. 请在有桌面的 Python/Tk 环境启动。', file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
