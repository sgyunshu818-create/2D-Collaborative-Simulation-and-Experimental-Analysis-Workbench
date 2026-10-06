"""Application-local Source Han Serif fonts shared by Tk and Pygame.

The original OFL font files ship with the application. On Windows the fonts
are registered only for this process; no system font installation is needed.
"""

from functools import lru_cache
from io import BytesIO
import os
from pathlib import Path


FONT_FAMILY = 'Source Han Serif CN'
FONT_DIRECTORY = Path(__file__).resolve().parent / 'assets' / 'fonts'
_registered = False


def bundled_font_path(bold: bool = False) -> Path | None:
    weight = 'Bold' if bold else 'Regular'
    path = FONT_DIRECTORY / f'SourceHanSerifCN-{weight}.otf'
    return path if path.is_file() else None


@lru_cache(maxsize=2)
def _font_bytes(bold: bool = False) -> bytes | None:
    path = bundled_font_path(bold)
    return path.read_bytes() if path is not None else None


def register_private_fonts() -> bool:
    """Expose bundled fonts to Windows GDI for the lifetime of this process."""
    global _registered
    if _registered:
        return True
    if os.name != 'nt':
        return False
    import ctypes
    from ctypes import wintypes

    add_font = ctypes.WinDLL('gdi32', use_last_error=True).AddFontResourceExW
    add_font.argtypes = (wintypes.LPCWSTR, wintypes.DWORD, wintypes.LPVOID)
    add_font.restype = ctypes.c_int
    paths = [bundled_font_path(False), bundled_font_path(True)]
    if any(path is None for path in paths):
        return False
    # FR_PRIVATE: restricted to the calling process, automatically removed on exit.
    registered = [add_font(str(path), 0x10, None) > 0 for path in paths]
    _registered = all(registered)
    return _registered


def tk_font_family(root) -> str:
    """Choose the bundled family, with an explicit CJK fallback if unavailable."""
    from tkinter import font as tkfont

    register_private_fonts()
    families = set(tkfont.families(root))
    for family in (FONT_FAMILY, '思源宋体 CN', 'Source Han Serif SC',
                   '思源宋体', 'Noto Serif CJK SC', 'Noto Serif SC', 'SimSun'):
        if family in families:
            return family
    return tkfont.nametofont('TkDefaultFont', root=root).actual('family')


def pygame_font(size: int, bold: bool = False):
    """Use real Bold outlines for fine small text, keeping the requested pixels."""
    import pygame

    bold = bold or size <= 14
    data = _font_bytes(bold)
    if data is None and bold:
        data = _font_bytes(False)
    if data is not None:
        # Reading bytes also supports installations under non-ASCII paths.
        return pygame.font.Font(BytesIO(data), size)
    fallback = pygame.font.match_font(
        ['sourcehanserifcn', 'sourcehanserifsc', 'notoserifcjksc', 'notoserifsc', 'simsun'],
        bold=bold,
    )
    return pygame.font.Font(fallback, size)
