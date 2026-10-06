"""Shared, dependency-free desktop styling for the simulation workbench.

Fonts use points and native Tk DPI scaling after process awareness is set.
Geometry remains in Tk's coordinate system; no extra DPI multiplier is added.
"""

import tkinter as tk
from tkinter import font as tkfont, ttk

from .font_support import FONT_FAMILY, tk_font_family
from .tk_motion import Transition, blend_color


COLORS = {
    'background': '#06111D', 'surface': '#0B1D2B', 'header': '#10283A',
    'border': '#1D5065', 'text': '#DDECF4', 'muted': '#8FAABC',
    'accent': '#24C8E5', 'accent_hover': '#56D8EC', 'selected': '#123D50',
    'grid': '#163141', 'map_border': '#23556B', 'field': '#071724',
    'map': '#081924', 'site': '#507989', 'success': '#36CFAC',
    'navigation': '#071521', 'navigation_selected': '#103348',
    'navigation_text': '#E7F5FD', 'navigation_muted': '#8FAABC',
    'red': '#F16C80', 'blue': '#52C7F3',
}
UI_FONT = (FONT_FAMILY, 10, 'bold')
DATA_FONT = (FONT_FAMILY, 10, 'bold')


def cached_font(widget, size=10, weight='normal'):
    """Hold named Tk fonts for the interpreter, including transient Canvas text."""
    root = widget.winfo_toplevel()
    cache = getattr(root, '_workbench_font_cache', None)
    if cache is None:
        cache = root._workbench_font_cache = {}
    if weight == 'normal' and 0 < size <= 10:
        weight = 'bold'
    key = (size, weight)
    if key not in cache:
        family = getattr(root, '_workbench_font_family', FONT_FAMILY)
        cache[key] = tkfont.Font(root=root, family=family, size=size, weight=weight)
    return cache[key]


def apply_theme(widget):
    """Apply once per Tcl interpreter, including standalone scene editors."""
    root = widget.winfo_toplevel()
    if getattr(root, '_workbench_theme_ready', False):
        return ttk.Style(root)
    family = tk_font_family(root)
    root._workbench_font_family = family
    font = (family, 10, 'bold')
    for name in tkfont.names(root):
        tkfont.nametofont(name, root=root).configure(family=family)
    for name in ('TkDefaultFont', 'TkTextFont', 'TkMenuFont', 'TkHeadingFont', 'TkFixedFont'):
        tkfont.nametofont(name, root=root).configure(size=10, weight='bold')
    root.option_add('*Font', font)
    root.option_add('*TCombobox*Listbox.font', font)
    for option, value in {'background': COLORS['surface'], 'foreground': COLORS['text'],
                          'activeBackground': COLORS['selected'], 'activeForeground': COLORS['accent'],
                          'disabledForeground': COLORS['muted'], 'relief': 'flat', 'borderWidth': 1}.items():
        root.option_add('*Menu.' + option, value)
    for option, value in {'background': COLORS['field'], 'foreground': COLORS['text'],
                          'selectBackground': COLORS['selected'], 'selectForeground': COLORS['text']}.items():
        root.option_add('*TCombobox*Listbox.' + option, value)
    root.configure(background=COLORS['background'])
    style = ttk.Style(root)
    # Native vista ignores many background maps. Clam keeps the same palette
    # on buttons, tabs and fields without adding a third-party UI framework.
    if 'clam' in style.theme_names():
        style.theme_use('clam')
    style.configure('.', font=font, background=COLORS['background'], foreground=COLORS['text'])
    style.configure('TFrame', background=COLORS['background'])
    style.configure('Panel.TFrame', background=COLORS['surface'])
    style.configure('HudPanel.TFrame', background=COLORS['surface'], bordercolor=COLORS['border'],
                    lightcolor=COLORS['border'], darkcolor=COLORS['border'], borderwidth=1, relief='solid')
    style.configure('HudHeader.TFrame', background=COLORS['header'], bordercolor=COLORS['border'],
                    lightcolor=COLORS['border'], darkcolor=COLORS['border'], borderwidth=1, relief='solid')
    style.configure('Header.TFrame', background=COLORS['header'])
    style.configure('Header.TLabel', background=COLORS['header'], foreground=COLORS['text'])
    style.configure('HeaderMuted.TLabel', background=COLORS['header'], foreground=COLORS['muted'], font=(family, 9, 'bold'))
    style.configure('HudTitle.TLabel', background=COLORS['header'], foreground=COLORS['text'], font=(family, 15, 'bold'))
    style.configure('HudCompactTitle.TLabel', background=COLORS['header'], foreground=COLORS['text'], font=(family, 13, 'bold'))
    style.configure('TLabel', background=COLORS['background'], foreground=COLORS['text'])
    style.configure('Panel.TLabel', background=COLORS['surface'])
    style.configure('Muted.TLabel', foreground=COLORS['muted'], font=(family, 9, 'bold'))
    style.configure('PanelMuted.TLabel', background=COLORS['surface'], foreground=COLORS['muted'], font=(family, 9, 'bold'))
    style.configure('AppTitle.TLabel', font=(family, 16, 'bold'))
    style.configure('CompactTitle.TLabel', font=(family, 13, 'bold'))
    style.configure('PanelHeader.TLabel', background=COLORS['header'], foreground=COLORS['text'], font=(family, 10, 'bold'), padding=(12, 8))
    style.configure('Status.TLabel', foreground=COLORS['muted'], padding=(0, 6), font=(family, 9, 'bold'))
    # Remove the native dotted focus/bevel stack. A one-pixel colour-mapped
    # outline still shows keyboard focus without the old raised-button face.
    style.layout('TButton', [('Button.border', {'sticky': 'nswe', 'border': 1, 'children': [
        ('Button.padding', {'sticky': 'nswe', 'children': [('Button.label', {'sticky': 'nswe'})]})]})])
    style.configure('TButton', background=COLORS['surface'], foreground=COLORS['text'],
                    bordercolor=COLORS['border'], lightcolor=COLORS['border'], darkcolor=COLORS['border'],
                    padding=(9, 6), width=0, borderwidth=1, relief='flat')
    style.map('TButton', background=[('disabled', COLORS['background']), ('pressed', COLORS['selected']), ('active', COLORS['header'])],
              foreground=[('disabled', '#506E80')], bordercolor=[('focus', COLORS['accent']), ('active', '#32839B')],
              lightcolor=[('focus', COLORS['accent']), ('active', '#32839B')], darkcolor=[('focus', COLORS['accent']), ('active', '#32839B')])
    style.configure('Primary.TButton', background=COLORS['accent'], foreground=COLORS['background'],
                    bordercolor=COLORS['accent'], lightcolor=COLORS['accent'], darkcolor=COLORS['accent'], padding=(14, 6))
    style.map('Primary.TButton', background=[('disabled', '#315465'), ('pressed', COLORS['accent_hover']), ('active', COLORS['accent_hover'])],
              foreground=[('disabled', COLORS['muted']), ('!disabled', COLORS['background'])], bordercolor=[('focus', '#D9FAFF'), ('!disabled', COLORS['accent'])],
              lightcolor=[('focus', '#D9FAFF'), ('!disabled', COLORS['accent'])], darkcolor=[('focus', '#D9FAFF'), ('!disabled', COLORS['accent'])])
    style.configure('Ghost.TButton', background=COLORS['background'], bordercolor=COLORS['background'],
                    lightcolor=COLORS['background'], darkcolor=COLORS['background'])
    style.configure('PanelGhost.TButton', background=COLORS['surface'], bordercolor=COLORS['surface'],
                    lightcolor=COLORS['surface'], darkcolor=COLORS['surface'])
    for name in ('Ghost.TButton', 'PanelGhost.TButton'):
        style.map(name, background=[('pressed', COLORS['selected']), ('active', COLORS['selected'])],
                  bordercolor=[('focus', COLORS['accent']), ('active', COLORS['selected'])],
                  lightcolor=[('focus', COLORS['accent']), ('active', COLORS['selected'])],
                  darkcolor=[('focus', COLORS['accent']), ('active', COLORS['selected'])])
    style.configure('TMenubutton', background=COLORS['surface'], foreground=COLORS['text'],
                    bordercolor=COLORS['border'], lightcolor=COLORS['border'], darkcolor=COLORS['border'], padding=(8, 6), width=0, relief='flat', borderwidth=1)
    style.map('TMenubutton', background=[('active', COLORS['selected']), ('pressed', COLORS['selected'])], bordercolor=[('focus', COLORS['accent'])])
    style.configure('Disclosure.TButton', background=COLORS['surface'], anchor='w', padding=(0, 8), font=(family, 10, 'bold'),
                    bordercolor=COLORS['surface'], lightcolor=COLORS['surface'], darkcolor=COLORS['surface'])
    style.map('Disclosure.TButton', background=[('active', COLORS['header']), ('pressed', COLORS['selected'])],
              bordercolor=[('focus', COLORS['accent']), ('!focus', COLORS['surface'])],
              lightcolor=[('focus', COLORS['accent']), ('!focus', COLORS['surface'])],
              darkcolor=[('focus', COLORS['accent']), ('!focus', COLORS['surface'])])
    style.configure('TEntry', fieldbackground=COLORS['field'], foreground=COLORS['text'], insertcolor=COLORS['accent'], bordercolor=COLORS['border'],
                    lightcolor=COLORS['border'], darkcolor=COLORS['border'], padding=(7, 5), borderwidth=1, relief='flat',
                    selectbackground=COLORS['selected'], selectforeground=COLORS['text'])
    style.configure('TCombobox', fieldbackground=COLORS['field'], background=COLORS['header'], bordercolor=COLORS['border'],
                    lightcolor=COLORS['border'], darkcolor=COLORS['border'], arrowcolor=COLORS['muted'], padding=(7, 5), borderwidth=1, relief='flat')
    style.map('TCombobox', fieldbackground=[('readonly', COLORS['field']), ('disabled', COLORS['background'])],
              foreground=[('disabled', '#506E80'), ('readonly', COLORS['text'])], selectbackground=[('readonly', COLORS['selected'])],
              selectforeground=[('readonly', COLORS['text'])])
    for name in ('TEntry', 'TCombobox'):
        style.map(name, bordercolor=[('focus', COLORS['accent'])], lightcolor=[('focus', COLORS['accent'])], darkcolor=[('focus', COLORS['accent'])])
    style.configure('TCheckbutton', background=COLORS['surface'], padding=(0, 2))
    style.map('TCheckbutton', background=[('active', COLORS['surface'])])
    style.layout('MapToggle.TCheckbutton', [('Checkbutton.padding', {'sticky': 'nswe', 'children': [
        ('Checkbutton.focus', {'sticky': 'nswe', 'children': [('Checkbutton.label', {'sticky': 'nswe'})]})]})])
    style.configure('MapToggle.TCheckbutton', background=COLORS['header'], foreground=COLORS['muted'], padding=(9, 5))
    style.map('MapToggle.TCheckbutton', background=[('selected', COLORS['selected']), ('active', COLORS['selected'])],
              foreground=[('selected', COLORS['accent']), ('active', COLORS['text'])])
    style.configure('TNotebook', background=COLORS['surface'], borderwidth=0, bordercolor=COLORS['surface'],
                    lightcolor=COLORS['surface'], darkcolor=COLORS['surface'], tabmargins=(0, 0, 0, 0))
    style.configure('TNotebook.Tab', padding=(11, 9), background=COLORS['surface'], foreground=COLORS['muted'],
                    borderwidth=0, bordercolor=COLORS['surface'], lightcolor=COLORS['surface'], darkcolor=COLORS['surface'])
    style.map('TNotebook.Tab', background=[('selected', COLORS['selected']), ('active', COLORS['header'])],
              foreground=[('selected', COLORS['accent']), ('active', COLORS['text'])])
    style.layout('TNotebook.Tab', [('Notebook.padding', {'sticky': 'nswe', 'children': [
        ('Notebook.focus', {'sticky': 'nswe', 'children': [('Notebook.label', {'sticky': 'nswe'})]})]})])
    style.layout('Shell.TNotebook', [('Notebook.client', {'sticky': 'nswe'})])
    style.layout('Shell.TNotebook.Tab', [])
    style.configure('Shell.TNotebook', background=COLORS['background'], borderwidth=0, bordercolor=COLORS['background'],
                    lightcolor=COLORS['background'], darkcolor=COLORS['background'])
    style.configure('Treeview', background=COLORS['surface'], fieldbackground=COLORS['surface'], foreground=COLORS['text'],
                    rowheight=30, borderwidth=0, relief='flat', bordercolor=COLORS['surface'], lightcolor=COLORS['surface'], darkcolor=COLORS['surface'])
    style.map('Treeview', background=[('selected', COLORS['selected'])], foreground=[('selected', COLORS['accent'])])
    style.configure('Treeview.Heading', background=COLORS['header'], foreground=COLORS['muted'], padding=(7, 7),
                    relief='flat', borderwidth=0, bordercolor=COLORS['header'], lightcolor=COLORS['header'], darkcolor=COLORS['header'])
    style.map('Treeview.Heading', background=[('active', COLORS['selected'])])
    style.configure('TPanedwindow', background=COLORS['background'])
    style.configure('Sash', sashthickness=10, gripcount=0)
    style.configure('TSeparator', background=COLORS['border'])
    for name in ('TScrollbar', 'Vertical.TScrollbar', 'Horizontal.TScrollbar'):
        style.configure(name, background='#24576D', troughcolor=COLORS['surface'], bordercolor=COLORS['surface'],
                        lightcolor='#24576D', darkcolor='#24576D', arrowcolor=COLORS['muted'], arrowsize=10, borderwidth=0, relief='flat')
        style.map(name, background=[('pressed', '#3487A1'), ('active', '#2A6A81')],
                  lightcolor=[('pressed', '#3487A1'), ('active', '#2A6A81')],
                  darkcolor=[('pressed', '#3487A1'), ('active', '#2A6A81')],
                  bordercolor=[('!disabled', COLORS['surface'])], troughcolor=[('!disabled', COLORS['surface'])])
    root._workbench_theme_ready = True
    return style


class CollapsibleFrame(ttk.Frame):
    """Keyboard-accessible disclosure that keeps its contents instantiated."""

    def __init__(self, parent, title, *, expanded=False):
        super().__init__(parent, style='Panel.TFrame')
        self.title = title
        self.expanded = expanded
        self.columnconfigure(0, weight=1)
        self.button = ttk.Button(self, style='Disclosure.TButton', command=self.toggle)
        self.button.grid(row=0, column=0, sticky='ew')
        self.clip = ttk.Frame(self, style='Panel.TFrame')
        self.clip.columnconfigure(0, weight=1)
        self.clip.grid(row=1, column=0, sticky='ew')
        self.body = ttk.Frame(self.clip, style='Panel.TFrame', padding=(8, 6))
        self.body.columnconfigure(1, weight=1)
        self.body.grid(row=0, column=0, sticky='ew')
        self._transition = Transition(self)
        self.set_expanded(expanded)

    def set_expanded(self, expanded, *, animate=False):
        self._transition.cancel()
        initial_height = self.clip.winfo_height() if self.clip.winfo_manager() else 0
        self.expanded = bool(expanded)
        self.button.configure(text=('▾ ' if self.expanded else '▸ ') + self.title)
        def complete():
            self.body.place_forget()
            self.body.grid(row=0, column=0, sticky='ew')
            self.clip.grid_propagate(True)
            if self.expanded:
                self.clip.grid()
            else:
                self.clip.grid_remove()
        if not animate:
            complete()
            return
        target_height = self.body.winfo_reqheight() if self.expanded else 0
        self.clip.grid()
        self.clip.grid_propagate(False)
        self.body.grid_remove()
        self.body.place(x=0, y=0, relwidth=1)
        self._transition.start(lambda amount: self.clip.configure(
            height=max(1, round(initial_height + (target_height-initial_height) * amount))),
            duration=160, complete=complete)

    def toggle(self):
        self.set_expanded(not self.expanded, animate=True)


class PanelHeader(tk.Canvas):
    """A restrained HUD section heading, rather than extra dashboard data."""

    def __init__(self, parent, title, *, detail=None, height=36):
        super().__init__(parent, height=height, width=1, background=COLORS['header'],
                         highlightthickness=0, borderwidth=0)
        self.title = title
        self.detail = detail
        self.family = tkfont.nametofont('TkDefaultFont', root=self).actual('family')
        self.bind('<Configure>', lambda event: self._paint())
        if isinstance(detail, tk.Variable):
            self._detail_trace = detail.trace_add('write', lambda *_: self._paint())
            self.bind('<Destroy>', self._clear_trace)

    def _clear_trace(self, event):
        if event.widget is self:
            self.detail.trace_remove('write', self._detail_trace)

    def _paint(self):
        self.delete('all')
        width = max(1, self.winfo_width())
        self.create_line(0, 35, width, 35, fill=COLORS['border'])
        self.create_line(0, 0, 18, 0, fill=COLORS['accent'], width=2)
        self.create_polygon(12, 15, 17, 10, 22, 15, 17, 20, fill='', outline=COLORS['accent'], width=1.4)
        self.create_text(30, 17, text=self.title, fill=COLORS['text'], anchor='w', font=(self.family, 10, 'bold'))
        detail = self.detail.get() if isinstance(self.detail, tk.Variable) else self.detail
        if detail and width > 230:
            self.create_text(width - 12, 17, text=detail, fill=COLORS['muted'], anchor='e', font=(self.family, 9))


class BooleanToggle(tk.Canvas):
    """Small flat switch, retaining the same BooleanVar form contract."""

    def __init__(self, parent, variable):
        super().__init__(parent, width=42, height=26, background=COLORS['surface'],
                         borderwidth=0, highlightthickness=1, highlightbackground=COLORS['surface'],
                         highlightcolor=COLORS['accent'], takefocus=True, cursor='hand2')
        self.variable = variable
        self._position = float(bool(variable.get()))
        self._transition = Transition(self)
        self._trace = variable.trace_add('write', self._state_changed)
        self.bind('<Button-1>', self._toggle)
        self.bind('<space>', self._toggle)
        self.bind('<Return>', self._toggle)
        self.bind('<Destroy>', self._destroy_trace, add='+')
        self._paint()

    def configure(self, cnf=None, **kwargs):
        result = super().configure(cnf, **kwargs)
        if 'state' in kwargs and hasattr(self, 'variable'):
            self._transition.cancel()
            self._position = float(bool(self.variable.get()))
            self._paint()
        return result

    config = configure

    def _toggle(self, event=None):
        if self.cget('state') != 'disabled':
            self.focus_set()
            self.variable.set(not self.variable.get())
        return 'break'

    def _paint(self):
        self.delete('all')
        enabled = self.cget('state') != 'disabled'
        color = blend_color('#315467', COLORS['accent'], self._position) if enabled else '#203646'
        self.create_line(11, 13, 31, 13, width=20, capstyle='round', fill=color)
        x = 11 + 20 * self._position
        self.create_oval(x - 7, 6, x + 7, 20, fill=COLORS['text'], outline=COLORS['text'])

    def _state_changed(self, *_):
        origin, target = self._position, float(bool(self.variable.get()))
        if self.cget('state') == 'disabled':
            self._transition.cancel()
            self._position = target
            self._paint()
            return
        def update(amount):
            self._position = origin + (target-origin) * amount
            self._paint()
        self._transition.start(update, duration=100)

    def _destroy_trace(self, event):
        if event.widget is self:
            self.variable.trace_remove('write', self._trace)


class NavigationRail(tk.Canvas):
    """Four-page navigation with drawn icons and a compact Chinese layout."""

    LABELS = ('场景编辑', '历史分析', '图像实验', '帮助与环境')
    SHORT_LABELS = ('场景', '历史', '图像', '帮助')

    def __init__(self, parent, on_select, *, brand_image=None):
        super().__init__(parent, width=168, background=COLORS['navigation'],
                         borderwidth=0, highlightthickness=0, takefocus=True, cursor='hand2')
        self.on_select = on_select
        self.brand_image = brand_image
        self.selected = 0
        self.hovered = None
        self.focused = False
        self.compact = False
        self._indicator = 0.0
        self._weights = [1.0, 0.0, 0.0, 0.0]
        self._transition = Transition(self)
        self.family = tkfont.nametofont('TkDefaultFont', root=self).actual('family')
        self.bind('<Configure>', lambda event: self._paint())
        self.bind('<Motion>', self._motion)
        self.bind('<Leave>', lambda event: self._set_hover(None))
        self.bind('<Button-1>', self._click)
        self.bind('<FocusIn>', lambda event: self._set_focus(True))
        self.bind('<FocusOut>', lambda event: self._set_focus(False))
        self.bind('<Up>', lambda event: self._key_select(-1))
        self.bind('<Down>', lambda event: self._key_select(1))
        self.bind('<Return>', lambda event: self.on_select(self.selected))
        self.bind('<space>', lambda event: self.on_select(self.selected))

    def set_compact(self, compact):
        compact = bool(compact)
        if compact != self.compact:
            self._transition.cancel()
            self.compact = compact
            self._indicator = float(self.selected)
            self._weights = [float(i == self.selected) for i in range(4)]
            self.configure(width=72 if compact else 168)
            self._paint()

    def select(self, index):
        index = int(index)
        if not 0 <= index < len(self.LABELS):
            return
        if index == self.selected:
            return
        self.selected = index
        origin, weights = self._indicator, self._weights[:]
        def update(amount):
            self._indicator = origin + (index-origin) * amount
            self._weights = [value + (float(i == index)-value) * amount for i, value in enumerate(weights)]
            self._paint()
        self._transition.start(update, duration=140)

    def _set_focus(self, focused):
        self.focused = focused
        self._paint()

    def _row_at(self, y):
        height = 65 if self.compact else 52
        index = int((y - 112) // height)
        return index if 0 <= index < 4 else None

    def _motion(self, event):
        self._set_hover(self._row_at(event.y))

    def _set_hover(self, index):
        if index != self.hovered:
            self.hovered = index
            self._paint()

    def _click(self, event):
        index = self._row_at(event.y)
        if index is not None:
            self.focus_set()
            self.on_select(index)

    def _key_select(self, offset):
        self.on_select((self.selected + offset) % 4)
        return 'break'

    def _paint_icon(self, kind, x, y, color):
        line = dict(fill=color, width=1.6, capstyle='round', joinstyle='round')
        if kind == 0:
            self.create_line(x - 7, y + 6, x, y - 6, x + 8, y + 4, **line)
            for px, py in ((x - 7, y + 6), (x, y - 6), (x + 8, y + 4)):
                self.create_oval(px - 2, py - 2, px + 2, py + 2, fill=COLORS['navigation'], outline=color, width=1.6)
        elif kind == 1:
            self.create_line(x - 8, y - 8, x - 8, y + 8, x + 9, y + 8, **line)
            self.create_line(x - 4, y + 3, x, y - 1, x + 4, y + 1, x + 9, y - 6, **line)
        elif kind == 2:
            self.create_rectangle(x - 9, y - 8, x + 9, y + 8, outline=color, width=1.6)
            self.create_line(x - 7, y + 6, x - 2, y, x + 2, y + 3, x + 6, y - 1, x + 8, y + 2, **line)
            self.create_oval(x - 6, y - 5, x - 3, y - 2, fill=color, outline=color)
        else:
            self.create_oval(x - 9, y - 9, x + 9, y + 9, outline=color, width=1.6)
            self.create_text(x, y, text='?', fill=color, font=(self.family, 10, 'bold'))

    def _paint(self):
        self.delete('all')
        width = 72 if self.compact else 168
        self.create_line(width - 1, 0, width - 1, max(1, self.winfo_height()), fill=COLORS['border'])
        self.create_line(12, 96, width - 12, 96, fill=COLORS['border'])
        if self.brand_image is not None:
            self.create_image(width / 2 if self.compact else 32, 40, image=self.brand_image)
        else:
            # Same collaborative-node geometry as the application icon.
            cx = width / 2 if self.compact else 32
            self.create_line(cx - 10, 48, cx, 31, cx + 11, 46, fill='#54C3D9', width=2.5, capstyle='round')
            for x, y in ((cx - 10, 48), (cx, 31), (cx + 11, 46)):
                self.create_oval(x - 3, y - 3, x + 3, y + 3, fill='white', outline='white')
        if self.compact:
            self.create_text(width / 2, 79, text='协同仿真', fill=COLORS['navigation_text'], font=(self.family, 9))
        else:
            self.create_text(60, 36, text='协同仿真', anchor='w', fill=COLORS['navigation_text'], font=(self.family, 12, 'bold'))
            self.create_text(60, 57, text='实验工作台', anchor='w', fill=COLORS['navigation_muted'], font=(self.family, 9))
        height = 65 if self.compact else 52
        if self.hovered is not None and self.hovered != self.selected:
            y = 112 + self.hovered * height
            self.create_rectangle(8, y + 3, width - 8, y + height - 3,
                                  fill=COLORS['navigation_selected'], outline='')
        indicator_y = 112 + self._indicator * height
        self.create_rectangle(8, indicator_y + 3, width - 8, indicator_y + height - 3,
                              fill=COLORS['navigation_selected'], outline='')
        self.create_rectangle(0, indicator_y + 12, 3, indicator_y + height - 12, fill=COLORS['accent'], outline='')
        for index, label in enumerate(self.LABELS):
            y = 112 + index * height
            active = index == self.selected
            if active and self.focused:
                self.create_rectangle(8, y + 3, width - 8, y + height - 3, outline='#54C3D9', width=1)
            color = blend_color(COLORS['navigation_muted'], COLORS['navigation_text'], self._weights[index])
            self._paint_icon(index, width / 2 if self.compact else 30, y + (22 if self.compact else height / 2), color)
            self.create_text(width / 2 if self.compact else 52, y + (47 if self.compact else height / 2),
                             text=self.SHORT_LABELS[index] if self.compact else label,
                             anchor='center' if self.compact else 'w', fill=color,
                             font=(self.family, 9 if self.compact else 10))
        if not self.compact:
            self.create_text(20, max(440, self.winfo_height() - 26), text='教学实验 · 二维空间', anchor='w',
                             fill=COLORS['navigation_muted'], font=(self.family, 9))
