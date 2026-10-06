"""Recover Tcl script discovery for Windows Python installed under Unicode paths.

Uses this interpreter's existing resources and their bundled license files.
The process environment is changed only after normal initialization fails.
"""

import hashlib
import os
from pathlib import Path
import shutil
import sys
import tempfile
import tkinter as tk

from .dpi_support import enable_native_dpi


def create_root():
    enable_native_dpi()
    try:
        return tk.Tk()
    except tk.TclError as original:
        if os.name != 'nt' or 'init.tcl' not in str(original):
            raise
        if os.environ.get('TCL_LIBRARY') or os.environ.get('TK_LIBRARY'):
            raise tk.TclError(f'{original}\n已有 TCL_LIBRARY/TK_LIBRARY 配置，请核对与当前 Python 匹配。') from original
        source = Path(sys.base_prefix) / 'tcl'
        tcl, tk_dir = source / f'tcl{tk.TclVersion}', source / f'tk{tk.TkVersion}'
        if not (tcl / 'init.tcl').is_file() or not (tk_dir / 'tk.tcl').is_file():
            raise tk.TclError(f'{original}\n当前 Python 缺少 Tcl/Tk 脚本资源；安装含 Tcl/Tk 的 Python 3.11 后重新建立虚拟环境。') from original
        target_root = Path(tempfile.gettempdir()) / 'sim_workbench_tk'
        if not str(target_root).isascii():
            raise tk.TclError(f'{original}\n当前 Tcl 无法读取 Unicode 安装路径，请使用含 Tcl/Tk 的 Python 或 ASCII 临时目录。') from original
        fingerprint = hashlib.sha256((tcl / 'init.tcl').read_bytes() + (tk_dir / 'tk.tcl').read_bytes()).hexdigest()[:16]
        target = target_root / fingerprint
        for folder in (tcl, tk_dir):
            destination = target / folder.name
            if not destination.is_dir():
                shutil.copytree(folder, destination, dirs_exist_ok=True)
        os.environ['TCL_LIBRARY'] = str(target / tcl.name)
        os.environ['TK_LIBRARY'] = str(target / tk_dir.name)
        try:
            return tk.Tk()
        except tk.TclError:
            os.environ.pop('TCL_LIBRARY', None)
            os.environ.pop('TK_LIBRARY', None)
            raise
