"""Astra's mountain artwork, with a separate identity for saved scenes/replays."""
from . import detailed_terrain as atlas

ASSET_DIR = atlas.PHOTOGRAPHIC_ASSET_DIR
BACKGROUND = atlas.BACKGROUND
TILE_CACHE_LIMIT = atlas.TILE_CACHE_LIMIT


def metadata():
    return atlas.metadata(ASSET_DIR)


def available():
    return (ASSET_DIR / 'terrain.json').is_file()


def clear_cache():
    atlas.clear_cache()


def cache_info():
    return atlas.cache_info()


def terrain_view_surface(world_size, viewport_size, world_viewport):
    return atlas.terrain_view_surface(world_size, viewport_size, world_viewport,
                                     asset_directory=ASSET_DIR)


def terrain_view_ppm(world_size, viewport_size, world_viewport):
    return atlas.terrain_view_ppm(world_size, viewport_size, world_viewport,
                                 asset_directory=ASSET_DIR)
