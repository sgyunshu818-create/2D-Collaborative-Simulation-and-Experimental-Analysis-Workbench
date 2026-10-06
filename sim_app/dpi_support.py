"""Opt this process into native Windows pixels before any UI is created.

This never changes Windows display settings. Native point-based Tk fonts can
use the display DPI, and Pygame surfaces avoid Windows bitmap enlargement.
"""

import ctypes
import sys


_attempted = False
_enabled = False


def enable_native_dpi() -> bool:
    """Best-effort PMv2, PMv1, then system awareness; safe on other platforms.

    Awareness is a process-wide, one-time setting, so callers must invoke this
    before creating Tk roots or initializing SDL displays. An existing manifest
    or an unavailable API may reject the request; startup still proceeds.
    """
    global _attempted, _enabled
    if _attempted:
        return _enabled
    _attempted = True
    if sys.platform != "win32":
        return False
    candidates = (
        ("user32", "SetProcessDpiAwarenessContext", (ctypes.c_void_p,), ctypes.c_int,
         (ctypes.c_void_p(-4),), False),
        ("shcore", "SetProcessDpiAwareness", (ctypes.c_int,), ctypes.c_long, (2,), True),
        ("user32", "SetProcessDPIAware", (), ctypes.c_int, (), False),
    )
    for library, name, argument_types, result_type, arguments, hresult in candidates:
        try:
            function = getattr(ctypes.WinDLL(library, use_last_error=True), name)
            function.argtypes = argument_types
            function.restype = result_type
            result = function(*arguments)
        except (AttributeError, OSError, TypeError, ValueError):
            continue
        succeeded = result == 0 if hresult else bool(result)
        if succeeded:
            _enabled = True
            return True
    return False
