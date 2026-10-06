"""A 25-region photographic atlas; coordinates remain fictional simulation units."""
from . import detailed_terrain as tiles

ASSET_DIR = tiles.LEGACY_ASSET_DIR.parent / 'terrain_atlas' / 'runtime'
WORLD_SIZE = (240000, 160000)
SOURCE_SIZE = (1536, 1024)
OVERLAP = (128, 86)
STRIDE = (1408, 938)
RASTER_SIZE = (7168, 4776)
DETAIL_ZOOM = RASTER_SIZE[0] / SOURCE_SIZE[0]
MAX_ZOOM = 256
REGION_NAMES = (
    ('西北高原','北岭山群','北源峡谷','北麓丘陵','东北群山'),
    ('西岭峡谷','西北林海','北段河谷','东北湖区','东岭山群'),
    ('西境山脉','西侧支谷','中央河谷','东侧支谷','东境山脉'),
    ('西南林区','西南丘陵','南段河谷','东南丘陵','东南林区'),
    ('西南群山','西南盆地','南部平原','东南田野','东南山群'),
)


def source_to_world(x, y, world_size=WORLD_SIZE):
    """Map the previous 12000×8000 scene into the preserved central source."""
    return ((2*STRIDE[0]+x*SOURCE_SIZE[0]/12000)*world_size[0]/RASTER_SIZE[0],
            (2*STRIDE[1]+y*SOURCE_SIZE[1]/8000)*world_size[1]/RASTER_SIZE[1])


def regions(world_size=WORLD_SIZE):
    data=metadata() if available() else None
    if data is not None and 'regions' in data:
        native=data['world_size']
        return [{**item,'center':[item['center'][0]*world_size[0]/native[0],
                                 item['center'][1]*world_size[1]/native[1]]}
                for item in data['regions']]
    result = []
    for row in range(5):
        for col in range(5):
            x=(col*STRIDE[0]+SOURCE_SIZE[0]/2)*world_size[0]/RASTER_SIZE[0]
            y=(row*STRIDE[1]+SOURCE_SIZE[1]/2)*world_size[1]/RASTER_SIZE[1]
            result.append({'id':f'r{row}_c{col}','name':REGION_NAMES[row][col],
                           'row':row,'col':col,'center':[x,y],'zoom':DETAIL_ZOOM})
    return result


def metadata():
    return tiles.metadata(ASSET_DIR)


def available():
    return (ASSET_DIR / 'terrain.json').is_file()


def clear_cache():
    tiles.clear_cache()


def cache_info():
    return tiles.cache_info()


def terrain_view_surface(world_size, viewport_size, world_viewport):
    return tiles.terrain_view_surface(world_size,viewport_size,world_viewport,asset_directory=ASSET_DIR)


def terrain_view_ppm(world_size, viewport_size, world_viewport):
    return tiles.terrain_view_ppm(world_size,viewport_size,world_viewport,asset_directory=ASSET_DIR)
