"""Cached application branding and equipment artwork shared by Tk and Pygame.

Run ``python -m sim_app.visual_assets`` to regenerate the checked-in icon sizes.
Artwork uses only Pygame, which is already a project dependency. Importing this
module does not initialize a display or create a window.
"""

from base64 import b64encode
from functools import lru_cache
from io import BytesIO
from pathlib import Path
import os
import struct
from math import log2
from .equipment import profile_key


ASSET_DIR = Path(__file__).resolve().parent / "assets"
ILLUSTRATED_DIR = ASSET_DIR / "illustrated"
APP_ICON_PATH = ASSET_DIR / "collaborative_simulation.ico"
ICON_SIZES = (16, 24, 32, 48, 64, 128, 256)
BRAND = (21, 38, 61)
CYAN = (84, 195, 217)
ACCENT = (36, 200, 229)
WHITE = (255, 255, 255)
MAP_BACKGROUND = (8, 24, 36)


@lru_cache(maxsize=12)
def _asset_bytes(name):
    path = ILLUSTRATED_DIR / (name + ".png")
    return path.read_bytes() if path.is_file() else None


@lru_cache(maxsize=12)
def _image_asset(name):
    payload = _asset_bytes(name)
    if payload is None:
        return None
    # Loading bytes avoids the SDL/Tcl Unicode filename loaders on Windows.
    return _pygame().image.load(BytesIO(payload), name + ".png")


def equipment_asset_available(unit_type):
    return _image_asset(_kind(unit_type)) is not None


def _illustrated_unit_surface(kind, color, size, selected):
    source = _image_asset(kind)
    if source is None:
        return None
    pygame = _pygame()
    high = size * 4
    canvas = pygame.Surface((high, high), pygame.SRCALPHA)
    # The artwork retains its materials. Team identity comes from a thin
    # silhouette edge and a ground-plane base, rather than a full image tint.
    inset = 4 if selected else 2
    box = source.get_bounding_rect(min_alpha=8)
    cropped = source.subsurface(box) if box.width and box.height else source
    available = max(1, (size - inset * 2) * 4)
    factor = min(available / cropped.get_width(), available / cropped.get_height())
    image = pygame.transform.smoothscale(cropped, (max(1, round(cropped.get_width() * factor)),
                                                 max(1, round(cropped.get_height() * factor))))
    base = pygame.Rect(round(high * .13), round(high * .77), round(high * .74), round(high * .16))
    pygame.draw.ellipse(canvas, (*color, 48), base)
    pygame.draw.ellipse(canvas, (*color, 195), base, max(3, round(size * .05)))
    image_rect = image.get_rect(center=(high // 2, high // 2 - 2))
    mask = pygame.mask.from_surface(image, 40)
    silhouette = mask.to_surface(setcolor=(*color, 210), unsetcolor=(0, 0, 0, 0))
    edge = max(3, round(size * .05))
    for dx, dy in ((-edge, 0), (edge, 0), (0, -edge), (0, edge)):
        canvas.blit(silhouette, image_rect.move(dx, dy))
    canvas.blit(image, image_rect)
    if selected:
        pygame.draw.rect(canvas, ACCENT, (5, 5, high - 10, high - 10),
                         max(4, round(size * .055)), border_radius=round(high * .20))
    return pygame.transform.smoothscale(canvas, (size, size))


@lru_cache(maxsize=1)
def _terrain_source():
    source = _image_asset("terrain")
    if source is None:
        return None
    pygame = _pygame()
    result = source.copy()
    wash = pygame.Surface(result.get_size(), pygame.SRCALPHA)
    wash.fill((*MAP_BACKGROUND, 64))
    result.blit(wash, (0, 0))
    return result


@lru_cache(maxsize=4)
def _terrain_world_source(world_size):
    source = _terrain_source()
    if source is None:
        return None
    width, height = source.get_size()
    ratio = world_size[0] / world_size[1]
    crop_width, crop_height = width, height
    if width / height > ratio:
        crop_width = max(1, round(height * ratio))
    else:
        crop_height = max(1, round(width / ratio))
    # Preserve the pictured terrain's proportions; differing teaching worlds
    # use a centered cover crop instead of stretching rivers and roads.
    return source.subsurface(((width - crop_width) // 2, (height - crop_height) // 2,
                              crop_width, crop_height))


@lru_cache(maxsize=4)
def _terrain_png(size):
    source = _terrain_world_source(size) if size else _terrain_source()
    if source is None:
        return None
    pygame = _pygame()
    image = pygame.transform.smoothscale(source, size) if size else source
    stream = BytesIO()
    pygame.image.save(image, stream, "terrain.png")
    return stream.getvalue()


def terrain_png(size=None):
    """Return cached illustrative terrain PNG pixels, optionally at an exact size.

    The image adds no obstacles or navigational data. Missing art returns None.
    """
    size = tuple(max(1, int(value)) for value in size) if size is not None else None
    return _terrain_png(size)


@lru_cache(maxsize=8)
def _terrain_mip(world_size, level):
    source = _terrain_world_source(world_size)
    if source is None or level == 0:
        return source
    factor = 2 ** (-level / 2)
    return _pygame().transform.smoothscale(source, (max(1, round(source.get_width()*factor)),
                                                   max(1, round(source.get_height()*factor))))


@lru_cache(maxsize=8)
def _terrain_view_surface(world_size, viewport_size, world_viewport):
    source = _terrain_world_source(world_size)
    if source is None:
        return None
    pygame = _pygame()
    result = pygame.Surface(viewport_size)
    result.fill(MAP_BACKGROUND)
    width, height = world_size
    left, top, view_width, view_height = world_viewport
    # Reuse a modestly supersampled source instead of filtering the full image
    # during every animation frame; coordinates still span the same world.
    ratio = min(source.get_width() / (viewport_size[0]*width/view_width),
                source.get_height() / (viewport_size[1]*height/view_height))
    level = max(0, min(6, int(log2(max(1, ratio))*2)))
    source = _terrain_mip(world_size, level)
    x0, y0 = max(0, left), max(0, top)
    x1, y1 = min(width, left + view_width), min(height, top + view_height)
    if x1 <= x0 or y1 <= y0:
        return result
    src = pygame.Rect(round(x0 / width * source.get_width()), round(y0 / height * source.get_height()),
                      0, 0)
    src.width = max(1, round(x1 / width * source.get_width()) - src.x)
    src.height = max(1, round(y1 / height * source.get_height()) - src.y)
    src.clamp_ip(source.get_rect())
    dest = pygame.Rect(round((x0 - left) / view_width * viewport_size[0]),
                       round((y0 - top) / view_height * viewport_size[1]), 0, 0)
    dest.width = max(1, round((x1 - left) / view_width * viewport_size[0]) - dest.x)
    dest.height = max(1, round((y1 - top) / view_height * viewport_size[1]) - dest.y)
    result.blit(pygame.transform.smoothscale(source.subsurface(src), dest.size), dest)
    return result


def terrain_view_surface(world_size, viewport_size, world_viewport):
    """Map illustrative pixels to a visible world rectangle without huge textures.

    world_viewport is (left, top, width, height) in the simulation's coordinates;
    viewport_size is the requested pixel extent. Cached surfaces are read-only.
    """
    return _terrain_view_surface(tuple(float(v) for v in world_size),
                                 tuple(max(1, int(v)) for v in viewport_size),
                                 tuple(round(float(v), 8) for v in world_viewport))


@lru_cache(maxsize=8)
def _terrain_view_png(world_size, viewport_size, world_viewport):
    image = terrain_view_surface(world_size, viewport_size, world_viewport)
    if image is None:
        return None
    stream = BytesIO()
    _pygame().image.save(image, stream, "terrain.png")
    return stream.getvalue()


def terrain_view_png(world_size, viewport_size, world_viewport):
    """Return only the visible terrain pixels as PNG bytes, suitable for Tk."""
    return _terrain_view_png(tuple(float(v) for v in world_size), tuple(int(v) for v in viewport_size),
                             tuple(round(float(v), 8) for v in world_viewport))


@lru_cache(maxsize=8)
def _terrain_view_ppm(world_size, viewport_size, world_viewport):
    image = terrain_view_surface(world_size, viewport_size, world_viewport)
    if image is None:
        return None
    header = f'P6\n{image.get_width()} {image.get_height()}\n255\n'.encode('ascii')
    return header + _pygame().image.tobytes(image, 'RGB')


def terrain_view_ppm(world_size, viewport_size, world_viewport):
    """Opaque P6 bytes for Tk PhotoImage(format='PPM'), avoiding PNG compression."""
    return _terrain_view_ppm(tuple(float(v) for v in world_size), tuple(int(v) for v in viewport_size),
                             tuple(round(float(v), 8) for v in world_viewport))


def refresh_artwork_cache():
    """Refresh delivered artwork without touching the application icon caches."""
    for cached in (_asset_bytes, _image_asset, _terrain_source, _terrain_world_source, _terrain_mip, _terrain_png,
                   _terrain_view_surface, _terrain_view_png, _terrain_view_ppm, _unit_surface, _unit_png):
        cached.cache_clear()


def _pygame():
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    import pygame
    return pygame


def _rgb(color):
    if isinstance(color, str):
        value = color.removeprefix("#")
        if len(value) != 6:
            raise ValueError("Glyph colors must be #RRGGBB or RGB tuples")
        return tuple(int(value[i:i + 2], 16) for i in (0, 2, 4))
    return tuple(color[:3])


def _kind(unit_type):
    return profile_key(unit_type)


def unit_glyph_geometry(unit_type, center, size=28):
    """Return shared screen geometry for a quadrotor or a wheeled ground unit.

    Shape dictionaries contain ``kind`` (rect/polygon/line/circle), ``role``
    (body/detail/wheel), and either points, rect, or center/radius. All coordinates
    are in screen pixels; no heading is invented when the model has none.
    """
    cx, cy = center
    scale = size / 32

    def point(x, y):
        return (cx + x * scale, cy + y * scale)

    def rect(x, y, width, height, radius, role):
        return {"kind": "rect", "role": role,
                "rect": (*point(x, y), width * scale, height * scale),
                "radius": radius * scale}

    if _kind(unit_type) in ("tank", "armored_car"):
        wheels = tuple(rect(x, y, 4, 6, 1.1, "wheel")
                       for x in (-10, 6) for y in (-8, 3))
        return (*wheels, rect(-7, -11, 14, 22, 3.2, "body"),
                rect(-4.5, -6.5, 9, 4.5, 1.2, "detail"),
                {"kind": "line", "role": "detail",
                 "points": (point(-3.5, 6), point(3.5, 6)), "width": 1.2 * scale})
    if _kind(unit_type) == 'airplane':
        return ({'kind': 'polygon', 'role': 'body',
                 'points': tuple(point(x, y) for x, y in
                                 ((0, -15), (3, -4), (15, 5), (14, 8), (3, 5),
                                  (3, 11), (7, 13), (7, 15), (0, 13), (-7, 15),
                                  (-7, 13), (-3, 11), (-3, 5), (-14, 8), (-15, 5), (-3, -4)))},
                {'kind': 'line', 'role': 'detail', 'points': (point(0, -9), point(0, 8)),
                 'width': 1.2 * scale})
    shapes = []
    for x in (-7.5, 7.5):
        for y in (-7.5, 7.5):
            shapes.append({"kind": "line", "role": "body",
                           "points": (point(0, 0), point(x, y)), "width": 2 * scale})
            shapes.append({"kind": "circle", "role": "body", "center": point(x, y),
                           "radius": 3.6 * scale, "width": 1.4 * scale})
    shapes.append({"kind": "polygon", "role": "body",
                   "points": tuple(point(x, y) for x, y in
                                   ((0, -6), (3.3, -2), (3.3, 3), (0, 6), (-3.3, 3), (-3.3, -2)))})
    shapes.append({"kind": "line", "role": "detail",
                   "points": (point(0, -2.5), point(0, 2.5)), "width": 1.2 * scale})
    return tuple(shapes)


@lru_cache(maxsize=128)
def _unit_surface(kind, color, size, selected):
    illustrated = _illustrated_unit_surface(kind, color, size, selected)
    if illustrated is not None:
        return illustrated
    pygame = _pygame()
    supersample = 4
    canvas = pygame.Surface((size * supersample, size * supersample), pygame.SRCALPHA)
    # Leave a real gap between the selection ring and the unit silhouette.
    glyph_size = size * .80 if selected else size
    shapes = unit_glyph_geometry(kind, (size / 2, size / 2), glyph_size)
    colors = {"body": color, "detail": (221, 236, 244),
              "wheel": tuple(round(value * .76) for value in color)}
    for shape in shapes:
        ink = colors[shape["role"]]
        width = max(1, round(shape.get("width", 1) * supersample))
        if shape["kind"] == "rect":
            box = pygame.Rect(*(round(v * supersample) for v in shape["rect"]))
            pygame.draw.rect(canvas, ink, box, border_radius=round(shape["radius"] * supersample))
        elif shape["kind"] == "circle":
            pygame.draw.circle(canvas, ink, tuple(round(v * supersample) for v in shape["center"]),
                               round(shape["radius"] * supersample), width)
        else:
            points = tuple(tuple(round(v * supersample) for v in p) for p in shape["points"])
            if shape["kind"] == "polygon":
                pygame.draw.polygon(canvas, ink, points)
            else:
                pygame.draw.lines(canvas, ink, False, points, width)
    if selected:
        pygame.draw.circle(canvas, ACCENT, (size * 2, size * 2),
                           round((size / 2 - 1.5) * supersample), max(1, round(1.3 * supersample)))
    return pygame.transform.smoothscale(canvas, (size, size))


def unit_glyph_surface(unit_type, color, size=32, selected=False):
    """Return a cached antialiased transparent Pygame surface; treat as read-only."""
    return _unit_surface(_kind(unit_type), _rgb(color), int(size), bool(selected))


@lru_cache(maxsize=128)
def _unit_png(kind, color, size, selected):
    stream = BytesIO()
    _pygame().image.save(_unit_surface(kind, color, size, selected), stream, "unit.png")
    return stream.getvalue()


def unit_glyph_png(unit_type, color, size=32, selected=False):
    """Return cached PNG bytes for the same artwork used by Pygame."""
    return _unit_png(_kind(unit_type), _rgb(color), int(size), bool(selected))


def draw_tk_unit(canvas, center, unit_type, color, size=28, selected=False, tags=()):
    """Draw one cached Tk image item, carrying every supplied selection/drag tag."""
    import tkinter as tk
    key = (_kind(unit_type), _rgb(color), int(size), bool(selected))
    images = getattr(canvas, "_visual_unit_images", None)
    if images is None:
        images = canvas._visual_unit_images = {}
    if key not in images:
        images[key] = tk.PhotoImage(master=canvas, data=b64encode(_unit_png(*key)).decode("ascii"))
    return canvas.create_image(*center, image=images[key], tags=tags)


@lru_cache(maxsize=16)
def app_icon_surface(size=64):
    """Original three-node cooperation mark, rasterized once at each size."""
    pygame = _pygame()
    size = int(size)
    high = size * 4
    scale = high / 32
    canvas = pygame.Surface((high, high), pygame.SRCALPHA)
    box = pygame.Rect(round(scale), round(scale), round(30 * scale), round(30 * scale))
    pygame.draw.rect(canvas, BRAND, box, border_radius=round(7 * scale))
    nodes = tuple((round(x * scale), round(y * scale)) for x, y in ((9, 21), (16, 10), (24, 20)))
    pygame.draw.lines(canvas, WHITE, False, nodes, max(1, round(2 * scale)))
    for i, node in enumerate(nodes):
        pygame.draw.circle(canvas, CYAN if i == 1 else WHITE, node, round(3 * scale))
    return pygame.transform.smoothscale(canvas, (size, size))


@lru_cache(maxsize=16)
def app_icon_png(size=64):
    stream = BytesIO()
    _pygame().image.save(app_icon_surface(int(size)), stream, "brand.png")
    return stream.getvalue()


def set_tk_icon(root):
    """Apply the original app icon, preserving Tk image references and Unicode paths."""
    import tkinter as tk
    images = tuple(tk.PhotoImage(master=root, data=b64encode(app_icon_png(size)).decode("ascii"))
                   for size in (32, 16, 48, 64, 128, 256))
    root._visual_icon_images = images
    # Feed pixels directly: Tcl's Windows file-icon loader can silently fail
    # for Unicode installation paths even when the ICO itself is valid.
    root.iconphoto(True, *images)


def set_pygame_icon():
    _pygame().display.set_icon(app_icon_surface(64))


def generate_assets():
    """Regenerate PNG, SVG and Windows ICO from this single geometric source."""
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    entries, blocks = [], []
    offset = 6 + 16 * len(ICON_SIZES)
    for size in ICON_SIZES:
        payload = app_icon_png(size)
        (ASSET_DIR / f"collaborative_simulation_{size}.png").write_bytes(payload)
        if size <= 64:
            # Small DIB frames also work with Tk/Win32's legacy LoadImage path.
            pixels = _pygame().image.tobytes(app_icon_surface(size), 'BGRA', True)
            mask = bytes(((size + 31) // 32) * 4 * size)
            header = struct.pack('<IiiHHIIiiII', 40, size, size * 2, 1, 32, 0,
                                 len(pixels) + len(mask), 0, 0, 0, 0)
            payload = header + pixels + mask
        entries.append(struct.pack("<BBBBHHII", size % 256, size % 256, 0, 0, 1, 32, len(payload), offset))
        blocks.append(payload)
        offset += len(payload)
    APP_ICON_PATH.write_bytes(struct.pack("<HHH", 0, 1, len(entries)) + b"".join(entries + blocks))
    (ASSET_DIR / "collaborative_simulation.svg").write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32">'
        '<rect x="1" y="1" width="30" height="30" rx="7" fill="#15263D"/>'
        '<path d="M9 21 16 10 24 20" fill="none" stroke="#FFFFFF" stroke-width="2"/>'
        '<circle cx="9" cy="21" r="3" fill="#FFFFFF"/>'
        '<circle cx="16" cy="10" r="3" fill="#54C3D9"/>'
        '<circle cx="24" cy="20" r="3" fill="#FFFFFF"/></svg>', encoding="utf-8")
    return APP_ICON_PATH


def generate_preview(path):
    pygame = _pygame()
    pygame.font.init()
    canvas = pygame.Surface((860, 330))
    canvas.fill((244, 247, 251))
    font = pygame.font.Font(None, 23)
    canvas.blit(app_icon_surface(256), (24, 24))
    for i, size in enumerate((16, 24, 32, 48, 64)):
        x = 326 + i * 88
        canvas.blit(app_icon_surface(size), (x, 50 + (64 - size) // 2))
        canvas.blit(font.render(str(size) + ' px', True, BRAND), (x, 126))
        for row, kind in enumerate(("air", "ground")):
            canvas.blit(unit_glyph_surface(kind, "#C44F65", size), (x, 170 + row * 60))
            canvas.blit(unit_glyph_surface(kind, "#287FC1", size), (x + size + 4, 170 + row * 60))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pygame.image.save(canvas, str(path))


if __name__ == "__main__":
    print(generate_assets())
    generate_preview(Path(__file__).resolve().parent.parent / "artifacts/ui_premium_20261003/icon_preview.png")
