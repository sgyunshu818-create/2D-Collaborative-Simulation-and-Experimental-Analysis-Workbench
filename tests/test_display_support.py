"""Native DPI startup order and real Source Han Serif small-text outlines."""

import ctypes
from io import BytesIO
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from sim_app import app, dpi_support, font_support, tk_runtime, ui_theme
from sim_app.scene import scene_from_data
from tests.test_scene import valid_config


class DpiSupportTests(unittest.TestCase):
    def setUp(self):
        self.state = patch.multiple(dpi_support, _attempted=False, _enabled=False)
        self.state.start()
        self.addCleanup(self.state.stop)

    def libraries(self, modern=1, shcore=0, legacy=1):
        return {
            "user32": SimpleNamespace(SetProcessDpiAwarenessContext=Mock(return_value=modern),
                                      SetProcessDPIAware=Mock(return_value=legacy)),
            "shcore": SimpleNamespace(SetProcessDpiAwareness=Mock(return_value=shcore)),
        }

    def loader(self, libraries):
        return patch.object(ctypes, "WinDLL", side_effect=lambda name, **_: libraries[name], create=True)

    def test_per_monitor_v2_is_attempted_first_once_with_pointer_sized_context(self):
        libraries = self.libraries()
        with patch.object(dpi_support.sys, "platform", "win32"), self.loader(libraries) as loader:
            self.assertTrue(dpi_support.enable_native_dpi())
            self.assertTrue(dpi_support.enable_native_dpi())
        modern = libraries["user32"].SetProcessDpiAwarenessContext
        self.assertEqual(modern.call_count, 1)
        self.assertEqual(modern.call_args.args[0].value, ctypes.c_void_p(-4).value)
        self.assertEqual(modern.argtypes, (ctypes.c_void_p,))
        self.assertEqual(loader.call_count, 1)
        libraries["shcore"].SetProcessDpiAwareness.assert_not_called()
        libraries["user32"].SetProcessDPIAware.assert_not_called()

    def test_shcore_fallback_treats_hresult_zero_as_success(self):
        libraries = self.libraries(modern=0)
        with patch.object(dpi_support.sys, "platform", "win32"), self.loader(libraries):
            self.assertTrue(dpi_support.enable_native_dpi())
        libraries["shcore"].SetProcessDpiAwareness.assert_called_once_with(2)
        libraries["user32"].SetProcessDPIAware.assert_not_called()

    def test_missing_modern_api_and_failed_hresult_use_legacy_api(self):
        libraries = self.libraries(shcore=-2147024891)
        del libraries["user32"].SetProcessDpiAwarenessContext
        with patch.object(dpi_support.sys, "platform", "win32"), self.loader(libraries):
            self.assertTrue(dpi_support.enable_native_dpi())
        libraries["user32"].SetProcessDPIAware.assert_called_once_with()

    def test_api_failure_never_blocks_startup_or_retries_on_later_roots(self):
        libraries = self.libraries(modern=0, shcore=-1, legacy=0)
        with patch.object(dpi_support.sys, "platform", "win32"), self.loader(libraries) as loader:
            self.assertFalse(dpi_support.enable_native_dpi())
            self.assertFalse(dpi_support.enable_native_dpi())
        self.assertEqual(loader.call_count, 3)
        with patch.multiple(dpi_support, _attempted=False, _enabled=False), \
             patch.object(dpi_support.sys, "platform", "win32"), \
             patch.object(ctypes, "WinDLL", side_effect=OSError("unavailable"), create=True):
            self.assertFalse(dpi_support.enable_native_dpi())

    def test_other_platforms_do_not_load_windows_libraries(self):
        with patch.object(dpi_support.sys, "platform", "linux"), \
             patch.object(ctypes, "WinDLL", create=True) as loader:
            self.assertFalse(dpi_support.enable_native_dpi())
        loader.assert_not_called()

    def test_tk_awareness_is_set_before_any_root_creation_and_api_failure_is_harmless(self):
        order = []
        sentinel = object()
        with patch.object(tk_runtime, "enable_native_dpi", side_effect=lambda: order.append("dpi") or False), \
             patch.object(tk_runtime.tk, "Tk", side_effect=lambda: order.append("root") or sentinel):
            self.assertIs(tk_runtime.create_root(), sentinel)
        self.assertEqual(order, ["dpi", "root"])

    def test_pygame_normal_and_replay_start_set_dpi_before_display_initialization(self):
        for replay in (False, True):
            with self.subTest(replay=replay):
                order = []
                pygame = SimpleNamespace(
                    display=SimpleNamespace(init=lambda: order.append("display"), set_caption=lambda *_: None,
                                            set_mode=lambda *_: object()),
                    font=SimpleNamespace(init=lambda: None),
                    error=RuntimeError, quit=lambda: None)
                renderer = SimpleNamespace(font_path="Source Han Serif", label=lambda text, _: text)
                modules = {"pygame": pygame,
                           "sim_app.renderer": SimpleNamespace(Renderer=lambda *_: renderer, WINDOW_SIZE=(1280, 800)),
                           "sim_app.visual_assets": SimpleNamespace(set_pygame_icon=lambda: None)}
                playback = SimpleNamespace()
                with patch.dict(sys.modules, modules), \
                     patch.object(app, "enable_native_dpi", side_effect=lambda: order.append("dpi")), \
                     patch.object(app, "App") as app_class, \
                     patch.object(app, "load_scene", return_value=scene_from_data(valid_config())), \
                     patch("sim_app.replay.load_replay", return_value=playback):
                    app_class.return_value.run.return_value = 0
                    code = app.run_replay(Path("test.json")) if replay else app.run(Path("test.json"))
                    self.assertEqual(code, 0)
                self.assertEqual(order[:2], ["dpi", "display"])


class FontSupportTests(unittest.TestCase):
    def test_small_pygame_text_uses_real_bold_bytes_without_synthetic_weight_or_resizing(self):
        calls = []
        result = object()
        pygame = SimpleNamespace(font=SimpleNamespace(Font=lambda source, size: calls.append((source, size)) or result))
        for size, requested_bold, expected_bold in ((9, False, True), (14, False, True),
                                                  (15, False, False), (20, True, True)):
            with self.subTest(size=size, bold=requested_bold), patch.dict(sys.modules, {"pygame": pygame}), \
                 patch.object(font_support, "_font_bytes", return_value=b"real font outlines") as font_bytes:
                self.assertIs(font_support.pygame_font(size, requested_bold), result)
                font_bytes.assert_called_once_with(expected_bold)
                self.assertIsInstance(calls[-1][0], BytesIO)
                self.assertEqual(calls[-1][1], size)

    def test_missing_bold_uses_bundled_regular_before_installed_font_fallback(self):
        fallback = Mock()
        pygame = SimpleNamespace(font=SimpleNamespace(Font=Mock(return_value=object()), match_font=fallback))
        with patch.dict(sys.modules, {"pygame": pygame}), \
             patch.object(font_support, "_font_bytes", side_effect=(None, b"regular")) as font_bytes:
            font_support.pygame_font(12)
        self.assertEqual([call.args[0] for call in font_bytes.call_args_list], [True, False])
        fallback.assert_not_called()

    def test_real_bundled_font_renders_cjk_and_matches_bold_outline_at_native_size(self):
        try:
            import pygame
        except ImportError:
            self.skipTest("Pygame unavailable")
        initialized = pygame.font.get_init()
        pygame.font.init()
        try:
            path = font_support.bundled_font_path(True)
            self.assertIsNotNone(path)
            loaded = font_support.pygame_font(14)
            expected = pygame.font.Font(BytesIO(path.read_bytes()), 14)
            self.assertEqual(loaded.get_bold(), expected.get_bold())
            text = "飞机 装甲车 坦克 12000"
            actual_surface = loaded.render(text, True, (220, 235, 244))
            expected_surface = expected.render(text, True, (220, 235, 244))
            self.assertEqual(actual_surface.get_size(), expected_surface.get_size())
            self.assertEqual(pygame.image.tostring(actual_surface, "RGBA"),
                             pygame.image.tostring(expected_surface, "RGBA"))
            self.assertGreater(actual_surface.get_width(), 80)
        finally:
            if not initialized:
                pygame.font.quit()

    def test_tk_theme_keeps_source_han_family_points_and_uses_small_text_weight(self):
        root = SimpleNamespace(option_add=Mock(), configure=Mock())
        root.winfo_toplevel = lambda: root
        font_names = ("TkDefaultFont", "TkTextFont", "TkMenuFont", "TkHeadingFont", "TkFixedFont")
        fonts = {name: Mock() for name in font_names}
        style = Mock()
        style.theme_names.return_value = ("clam",)
        with patch.object(ui_theme, "tk_font_family", return_value=font_support.FONT_FAMILY), \
             patch.object(ui_theme.tkfont, "names", return_value=font_names), \
             patch.object(ui_theme.tkfont, "nametofont", side_effect=lambda name, **_: fonts[name]), \
             patch.object(ui_theme.ttk, "Style", return_value=style):
            ui_theme.apply_theme(root)
        root.option_add.assert_any_call("*Font", (font_support.FONT_FAMILY, 10, "bold"))
        for font in fonts.values():
            font.configure.assert_any_call(family=font_support.FONT_FAMILY)
            font.configure.assert_any_call(size=10, weight="bold")
        self.assertEqual(ui_theme.UI_FONT, ui_theme.DATA_FONT)
        self.assertEqual(ui_theme.UI_FONT[0], font_support.FONT_FAMILY)


if __name__ == "__main__":
    unittest.main()
