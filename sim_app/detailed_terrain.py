"""Shared tiled rendering for the fictional 12000 by 8000 valley maps.

Historical scenes retain their native vector terrain. The photographic_terrain
backend selects Astra's ImageGen artwork with a separate saved scene identity.
They are purely visual: no feature in this module becomes a simulation obstacle.
Only Pygame is needed at runtime; neither Pillow nor NumPy is imported here.
"""

from collections import OrderedDict
from io import BytesIO
from math import ceil, floor, hypot, log2
from pathlib import Path
import json


LEGACY_ASSET_DIR = Path(__file__).resolve().parent / "assets" / "detailed"
PHOTOGRAPHIC_ASSET_DIR = LEGACY_ASSET_DIR.parent / "terrain_astra" / "runtime"
ASSET_DIR = LEGACY_ASSET_DIR
WORLD_SIZE = (12000.0, 8000.0)
BACKGROUND = (19, 38, 42)
TILE_CACHE_LIMIT = 48
_tiles = OrderedDict()
_metadata = None
_feature_bins = None
_last_pose = None
_last_surface = None
_last_ppm = None
_label_cache = OrderedDict()
_active_directory = None


def _pygame():
    import pygame
    return pygame


def _activate_directory(directory):
    """A scene switch invalidates tiles and pose pixels from the previous map."""
    global _active_directory
    directory = Path(directory)
    if directory != _active_directory:
        clear_cache()
        _active_directory = directory


def metadata(asset_directory=None):
    """Read the map description and its stable, fictional landmark coordinates."""
    global _metadata
    if asset_directory is not None:
        _activate_directory(asset_directory)
    elif _active_directory is None:
        _activate_directory(ASSET_DIR)
    if _metadata is None:
        path = _active_directory / "terrain.json"
        if not path.is_file():
            return None
        _metadata = json.loads(path.read_text(encoding="utf-8"))
    return _metadata


def available():
    return (ASSET_DIR / "terrain.json").is_file()


def clear_cache():
    global _metadata, _feature_bins, _last_pose, _last_surface, _last_ppm
    global _active_directory
    _tiles.clear()
    _label_cache.clear()
    _metadata = _feature_bins = _last_pose = _last_surface = _last_ppm = None
    _active_directory = None


def cache_info():
    """Small diagnostics used by performance checks and the scene workbench."""
    return {"tiles": len(_tiles), "tile_limit": TILE_CACHE_LIMIT,
            "tile_bytes": sum(s.get_width() * s.get_height() * s.get_bytesize()
                              for s in _tiles.values()),
            "pose_surfaces": int(_last_surface is not None),
            "label_bitmaps": len(_label_cache),
            "ppm_bytes": len(_last_ppm) if _last_ppm is not None else 0}


def _tile(level, col, row):
    key = (level, col, row)
    if key in _tiles:
        _tiles.move_to_end(key)
        return _tiles[key]
    path = (_active_directory or ASSET_DIR) / str(level) / f"{col}_{row}.png"
    if not path.is_file():
        return None
    # SDL's path decoder does not reliably support every Windows Unicode path.
    result = _pygame().image.load(BytesIO(path.read_bytes()), path.name)
    _tiles[key] = result
    if len(_tiles) > TILE_CACHE_LIMIT:
        _tiles.popitem(last=False)
    return result


def _bounds(feature):
    if "points" in feature:
        pts = feature["points"]
        pad = feature.get("width", 0) / 2 + 12
        return (min(p[0] for p in pts) - pad, min(p[1] for p in pts) - pad,
                max(p[0] for p in pts) + pad, max(p[1] for p in pts) + pad)
    if "polygon" in feature:
        pts = feature["polygon"]
        return (min(p[0] for p in pts) - 8, min(p[1] for p in pts) - 8,
                max(p[0] for p in pts) + 8, max(p[1] for p in pts) + 8)
    radius = feature.get("r", 12) * 1.6
    return (feature["x"] - radius, feature["y"] - radius,
            feature["x"] + radius, feature["y"] + radius)


def _visible_features(rect):
    global _feature_bins
    data = metadata()
    if _feature_bins is None:
        _feature_bins = {}
        for index, feature in enumerate(data["features"]):
            x0, y0, x1, y1 = _bounds(feature)
            for row in range(floor(y0 / 512), floor(y1 / 512) + 1):
                for col in range(floor(x0 / 512), floor(x1 / 512) + 1):
                    _feature_bins.setdefault((col, row), []).append(index)
    left, top, width, height = rect
    result = set()
    for row in range(floor(top / 512), floor((top + height) / 512) + 1):
        for col in range(floor(left / 512), floor((left + width) / 512) + 1):
            result.update(_feature_bins.get((col, row), ()))
    for index in sorted(result):
        feature = data["features"][index]
        x0, y0, x1, y1 = _bounds(feature)
        if x1 >= left and y1 >= top and x0 <= left + width and y0 <= top + height:
            yield feature


def _line(surface, color, points, width):
    pygame = _pygame()
    width = max(1, round(width))
    if len(points) < 2:
        return
    pygame.draw.lines(surface, color, False, points, width)
    if width >= 3:
        r = width // 2
        for point in points:
            if -r <= point[0] <= surface.get_width() + r and -r <= point[1] <= surface.get_height() + r:
                pygame.draw.circle(surface, color, point, r)


def _dashes(surface, points, scale, color, dash=24, gap=27):
    """Dash phase belongs to the world geometry, so panning does not move markings."""
    pygame = _pygame()
    travelled = 0.0
    step = dash + gap
    for a, b in zip(points, points[1:]):
        distance = hypot(b[0] - a[0], b[1] - a[1])
        if not distance:
            continue
        cursor = -travelled % (step * scale)
        if cursor > gap * scale:
            cursor -= step * scale
        while cursor < distance:
            start, end = max(0, cursor), min(distance, cursor + dash * scale)
            if end > start:
                p = (round(a[0] + (b[0] - a[0]) * start / distance),
                     round(a[1] + (b[1] - a[1]) * start / distance))
                q = (round(a[0] + (b[0] - a[0]) * end / distance),
                     round(a[1] + (b[1] - a[1]) * end / distance))
                pygame.draw.line(surface, color, p, q, max(1, round(scale)))
            cursor += step * scale
        travelled += distance


def _draw_features(surface, view):
    pygame = _pygame()
    left, top, width, height = view
    sx, sy = surface.get_width() / width, surface.get_height() / height
    scale = min(sx, sy)
    if scale < .27:
        return
    def point(p):
        return (round((p[0] - left) * sx), round((p[1] - top) * sy))
    for feature in _visible_features(view):
        kind = feature["type"]
        pts = [point(p) for p in feature.get("points", feature.get("polygon", []))]
        if kind == "river":
            _line(surface, (44, 66, 63), pts, (feature["width"] + 26) * scale)
            _line(surface, (58, 104, 110), pts, feature["width"] * scale)
            _line(surface, (72, 125, 131), pts, feature["width"] * .57 * scale)
            if scale > 1.5:
                _line(surface, (91, 147, 148), pts, max(1, scale * 1.8))
        elif kind == "apron":
            pygame.draw.polygon(surface, feature["color"], pts)
            pygame.draw.lines(surface, (133,141,124), True, pts, max(1,round(scale*2)))
            if scale > .6:
                a,b,c,d=feature["polygon"]
                for t in (.2,.4,.6,.8):
                    u=(a[0]+(b[0]-a[0])*t,a[1]+(b[1]-a[1])*t)
                    v=(d[0]+(c[0]-d[0])*t,d[1]+(c[1]-d[1])*t)
                    pygame.draw.line(surface,(168,170,133),point(u),point(v),max(1,round(scale)))
        elif kind == "lake":
            pygame.draw.polygon(surface, (46, 85, 92), pts)
            pygame.draw.lines(surface, (100, 132, 115), True, pts, max(1, round(scale * 4)))
            if scale > 1:
                pygame.draw.lines(surface, (61, 111, 121), True, pts, max(1, round(scale)))
        elif kind == "field":
            pygame.draw.polygon(surface, feature["color"], pts)
            pygame.draw.lines(surface, (66, 74, 52), True, pts, max(1, round(3 * scale)))
            if scale > .7:
                a, b, c, d = feature["polygon"]
                for t in range(1, feature.get("rows", 14)):
                    u = t / feature.get("rows", 14)
                    pygame.draw.line(surface, feature["row_color"],
                                     point((a[0] + (d[0]-a[0])*u, a[1] + (d[1]-a[1])*u)),
                                     point((b[0] + (c[0]-b[0])*u, b[1] + (c[1]-b[1])*u)),
                                     max(1, round(scale * 1.3)))
        elif kind in ("road", "bridge"):
            road_width = feature["width"] * scale
            _line(surface, (40, 49, 44), pts, road_width + 8 * scale)
            _line(surface, (131, 128, 107) if kind == "bridge" else (113, 110, 92), pts, road_width)
            _line(surface, (149, 143, 117), pts, max(1, road_width - 5 * scale))
            if scale > .65 and feature["width"] >= 18:
                _dashes(surface, pts, scale, (207, 197, 150))
            if kind == "bridge" and scale > .7:
                # Paired guardrails follow the bridge's road tangent.
                for side in (-1, 1):
                    rail=[]
                    for i,p in enumerate(feature["points"]):
                        a=feature["points"][max(0,i-1)]
                        b=feature["points"][min(len(pts)-1,i+1)]
                        length=hypot(b[0]-a[0],b[1]-a[1]) or 1
                        offset=side*(feature["width"]/2-2)
                        rail.append(point((p[0]-(b[1]-a[1])*offset/length,
                                           p[1]+(b[0]-a[0])*offset/length)))
                    _line(surface,(194,180,139),rail,max(1,scale*1.4))
        elif kind == "runway":
            pygame.draw.polygon(surface, (38, 46, 47), pts)
            pygame.draw.lines(surface, (135, 144, 133), True, pts, max(1, round(scale * 3)))
            center = [point(p) for p in feature["centerline"]]
            _dashes(surface, center, scale, (218, 216, 183), dash=38, gap=35)
            if scale > .3:
                a, b, c, d = feature["polygon"]
                for u in (.035, .055, .075, .925, .945, .965):
                    p = (a[0]+(b[0]-a[0])*u, a[1]+(b[1]-a[1])*u)
                    q = (d[0]+(c[0]-d[0])*u, d[1]+(c[1]-d[1])*u)
                    pygame.draw.line(surface, (211, 209, 179), point(p), point(q), max(1, round(scale * 7)))
        elif kind == "building":
            if scale < .4:
                continue
            offset = max(2, round(6 * scale))
            pygame.draw.polygon(surface, (29, 40, 37), [(x+offset, y+offset) for x,y in pts])
            color = feature["color"]
            pygame.draw.polygon(surface, color, pts)
            pygame.draw.lines(surface, tuple(max(0, v-24) for v in color), True, pts, max(1, round(1.8 * scale)))
            a, b, c, d = pts
            if feature.get("kind") == "house":
                p = ((a[0]+d[0])//2, (a[1]+d[1])//2)
                q = ((b[0]+c[0])//2, (b[1]+c[1])//2)
                pygame.draw.line(surface, tuple(min(255,v+22) for v in color), p, q, max(1, round(scale * 2)))
            elif scale >= 1:
                for t in (.25, .5, .75):
                    p = (round(a[0]+(d[0]-a[0])*t), round(a[1]+(d[1]-a[1])*t))
                    q = (round(b[0]+(c[0]-b[0])*t), round(b[1]+(c[1]-b[1])*t))
                    pygame.draw.line(surface, tuple(min(255,v+17) for v in color), p, q, max(1, round(scale)))
            if scale >= 2:
                for t in (.3, .7):
                    x=round(a[0]*.5+d[0]*.5+(b[0]-a[0])*t)
                    y=round(a[1]*.5+d[1]*.5+(b[1]-a[1])*t)
                    pygame.draw.rect(surface,(54,68,65),
                                     (x-round(scale*2),y-round(scale*3),max(2,round(scale*4)),max(2,round(scale*6))))
                    pygame.draw.line(surface,(159,177,173),(x-round(scale),y-round(scale*2)),
                                     (x+round(scale),y-round(scale*2)),max(1,round(scale*.6)))
        elif kind == "tree" and scale >= .45:
            x, y = point((feature["x"], feature["y"]))
            r = max(2, round(feature["r"] * scale))
            pygame.draw.circle(surface, (23, 43, 34), (x+max(1,r//3), y+max(1,r//3)), r)
            if feature.get("kind") == "pine":
                pygame.draw.polygon(surface, feature["color"], [(x,y-r),(x+r,y+r//2),(x-r,y+r//2)])
                pygame.draw.line(surface, (96, 106, 62), (x,y-r//2), (x,y+r//2), max(1,r//6))
            else:
                pygame.draw.circle(surface, feature["color"], (x,y), r)
                pygame.draw.circle(surface, tuple(min(255,v+15) for v in feature["color"]),
                                   (x-r//3,y-r//3), max(1,r*2//3))
                if scale >= 2:
                    pygame.draw.circle(surface, tuple(max(0,v-10) for v in feature["color"]),
                                       (x+r//3,y+r//4), max(1,r//3))
        elif kind == "rock" and scale >= 1:
            pygame.draw.polygon(surface, feature["color"], pts)
            pygame.draw.line(surface, (157, 159, 141), pts[0], pts[1], max(1, round(scale)))


def _draw_labels(surface, view):
    """Cache only rendered words: font handles never survive pygame.quit()."""
    pygame=_pygame()
    left,top,width,height=view
    sx,sy=surface.get_width()/width,surface.get_height()/height
    scale=min(sx,sy)
    data=metadata()
    scale *= data.get('label_scale', 1)
    overview=set(data.get('overview_landmarks',("lake","valley_town","central_bridge","airport")))
    pending=[]
    for key, landmark in data["landmarks"].items():
        if key not in overview and scale < .22:
            continue
        x,y=landmark["position"]
        px,py=round((x-left)*sx),round((y-top)*sy)
        if not (0 <= px < surface.get_width() and 0 <= py < surface.get_height()):
            continue
        font_size=14 if scale < .22 else 16
        cache_key=(landmark["name"],font_size)
        pending.append((cache_key,px,py))
    missing=[key for key,x,y in pending if key not in _label_cache]
    if missing:
        if not pygame.font.get_init():
            pygame.font.init()
        path=LEGACY_ASSET_DIR.parent/"fonts"/"SourceHanSerifCN-Bold.otf"
        payload=path.read_bytes() if path.is_file() else None
        for key in missing:
            # A local lifetime avoids stale SDL_ttf resources after app restarts.
            font=pygame.font.Font(BytesIO(payload),key[1]) if payload else pygame.font.Font(None,key[1])
            text=font.render(key[0],True,(233,226,199))
            bitmap=pygame.Surface((text.get_width()+14,text.get_height()+6),pygame.SRCALPHA)
            pygame.draw.rect(bitmap,(22,40,40,225),bitmap.get_rect(),border_radius=4)
            pygame.draw.rect(bitmap,(127,150,126,195),bitmap.get_rect(),1,border_radius=4)
            bitmap.blit(text,(7,3))
            _label_cache[key]=bitmap
            if len(_label_cache)>data.get('label_cache_limit',24):
                _label_cache.popitem(last=False)
            del font
    for key,x,y in pending:
        _label_cache.move_to_end(key)
        bitmap=_label_cache[key]
        surface.blit(bitmap,(x-bitmap.get_width()//2,y+14))


def _normalize(world_size, viewport_size, world_viewport):
    world = tuple(float(v) for v in world_size)
    size = tuple(max(1, int(v)) for v in viewport_size)
    view = tuple(float(v) for v in world_viewport)
    if len(world) != 2 or len(size) != 2 or len(view) != 4:
        raise ValueError("Expected world and viewport sizes plus a 4-coordinate visible rectangle")
    if min(world) <= 0 or min(view[2:]) <= 0:
        raise ValueError("World and visible extents must be positive")
    return world, size, view


def terrain_view_surface(world_size, viewport_size, world_viewport, *, asset_directory=None):
    """Return a read-only viewport-sized surface, with an LRU of at most 48 tiles.

    The visible rectangle uses world coordinates. The canonical features scale
    proportionally for other world sizes. No full-world runtime texture exists.
    A single last-pose surface is shared with the PPM export used by Tk.
    """
    global _last_pose, _last_surface, _last_ppm
    _activate_directory(asset_directory if asset_directory is not None else ASSET_DIR)
    world, size, view = _normalize(world_size, viewport_size, world_viewport)
    pose = (world, size, view)
    if pose == _last_pose:
        return _last_surface
    data = metadata()
    if data is None:
        return None
    pygame = _pygame()
    surface = pygame.Surface(size)
    surface.fill(BACKGROUND)
    left, top, width, height = view
    native_world = data.get('world_size', WORLD_SIZE)
    canonical = (left * native_world[0]/world[0], top * native_world[1]/world[1],
                 width * native_world[0]/world[0], height * native_world[1]/world[1])
    native_x = data["raster_size"][0] / native_world[0]
    native_y = data["raster_size"][1] / native_world[1]
    pixel_ratio = min(native_x * canonical[2] / size[0], native_y * canonical[3] / size[1])
    level = max(0, min(len(data["levels"])-1, floor(log2(max(1, pixel_ratio)))))
    spec = data["levels"][level]
    rx, ry = spec["width"] / native_world[0], spec["height"] / native_world[1]
    raster_left, raster_top = canonical[0] * rx, canonical[1] * ry
    raster_width, raster_height = canonical[2] * rx, canonical[3] * ry
    x0, y0 = max(0, raster_left), max(0, raster_top)
    x1 = min(spec["width"], raster_left + raster_width)
    y1 = min(spec["height"], raster_top + raster_height)
    if x1 > x0 and y1 > y0:
        dx, dy = size[0] / raster_width, size[1] / raster_height
        tile_size = data["tile_size"]
        # Stitch only the visible source footprint plus a one-pixel filter
        # margin, then resample once. Tile boundaries cannot become independent
        # clamped filtering edges or shift against each other while panning.
        src_left, src_top = max(0,floor(x0)-1), max(0,floor(y0)-1)
        src_right, src_bottom = min(spec["width"],ceil(x1)+1), min(spec["height"],ceil(y1)+1)
        mosaic=pygame.Surface((src_right-src_left,src_bottom-src_top))
        mosaic.fill(BACKGROUND)
        for row in range(floor(src_top/tile_size), ceil(src_bottom/tile_size)):
            for col in range(floor(src_left/tile_size), ceil(src_right/tile_size)):
                tile = _tile(level, col, row)
                if tile is None:
                    continue
                ox, oy = col * tile_size, row * tile_size
                sx0 = max(0,src_left-ox)
                sy0 = max(0,src_top-oy)
                sx1 = min(tile.get_width(),src_right-ox)
                sy1 = min(tile.get_height(),src_bottom-oy)
                if sx1 > sx0 and sy1 > sy0:
                    mosaic.blit(tile,(ox+sx0-src_left,oy+sy0-src_top),
                                (sx0,sy0,sx1-sx0,sy1-sy0))
        dest_x, dest_y = round((src_left-raster_left)*dx), round((src_top-raster_top)*dy)
        dest_w = max(1,round((src_right-raster_left)*dx)-dest_x)
        dest_h = max(1,round((src_bottom-raster_top)*dy)-dest_y)
        surface.blit(pygame.transform.smoothscale(mosaic,(dest_w,dest_h)),(dest_x,dest_y))
        # Native vector layers share the exact world-to-viewport transform.
        clip = pygame.Rect(round((max(0,left)-left)*size[0]/width),
                           round((max(0,top)-top)*size[1]/height),
                           round((min(world[0],left+width)-max(0,left))*size[0]/width),
                           round((min(world[1],top+height)-max(0,top))*size[1]/height))
        surface.set_clip(clip)
        # Photographic artwork already contains its roads, woods and buildings.
        # Flat vector terrain would mask those materials and duplicate geography.
        if data.get("render_mode") != "photographic":
            _draw_features(surface, canonical)
        _draw_labels(surface, canonical)
        surface.set_clip(None)
    _last_pose, _last_surface, _last_ppm = pose, surface, None
    return surface


def terrain_view_ppm(world_size, viewport_size, world_viewport, *, asset_directory=None):
    """Return binary P6 pixels, directly accepted by Tk PhotoImage(format='PPM')."""
    global _last_ppm
    surface = terrain_view_surface(world_size, viewport_size, world_viewport,
                                   asset_directory=asset_directory)
    if surface is None:
        return None
    if _last_ppm is None:
        header = f"P6\n{surface.get_width()} {surface.get_height()}\n255\n".encode("ascii")
        _last_ppm = header + _pygame().image.tobytes(surface, "RGB")
    return _last_ppm
