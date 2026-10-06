"""Package the downloaded Natural Earth raster as bounded offline JPEG tiles."""
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import sys

import pygame

ROOT = Path(__file__).resolve().parents[1]
source = ROOT / 'artifacts/world_map_20261005/natural_earth_source/NE1_50M_SR_W/NE1_50M_SR_W.tif'
target = ROOT / 'sim_app/assets/world'
target.mkdir(parents=True, exist_ok=True)
payload = source.read_bytes()
converted = source.with_suffix('.png')
if converted.is_file():
    image = pygame.image.load(BytesIO(converted.read_bytes()), 'source.png')
else:
    # This build-only dependency avoids the SDL TIFF decoder; the app itself
    # only opens its packaged JPEG tiles and needs no Pillow installation.
    try:
        from PIL import Image
    except ImportError:
        raise SystemExit('地图构建需要 Pillow，或先把源 TIFF 无损转换为同名 PNG；应用运行不需要 Pillow。')
    with Image.open(source) as raster:
        rgb = raster.convert('RGB')
        image = pygame.image.frombytes(rgb.tobytes(), rgb.size, 'RGB')
print('Source raster', image.get_size(), flush=True)
files = []
for level in range(4):
    size = (1024 * 2**level, 512 * 2**level)
    scaled = pygame.transform.smoothscale(image, size)
    for x in range(size[0] // 512):
        for y in range(size[1] // 512):
            tile = scaled.subsurface((x*512, y*512, 512, 512))
            stream = BytesIO()
            pygame.image.save(tile, stream, 'tile.jpg')
            name = f'{level}_{x}_{y}.jpg'
            blob = stream.getvalue()
            (target/name).write_bytes(blob)
            files.append({'name':name, 'bytes':len(blob), 'sha256':sha256(blob).hexdigest()})
    print('Built level', level, size, flush=True)
metadata = {'dataset':'Natural Earth I with Shaded Relief and Water', 'version':'3.2.0',
    'source_scale':'1:50,000,000', 'source_size':list(image.get_size()), 'projection':'Plate Carree, WGS84',
    'extent':[-180,-90,180,90], 'source_sha256':sha256(payload).hexdigest(),
    'download_url':'https://naciscdn.org/naturalearth/50m/raster/NE1_50M_SR_W.zip',
    'source_page':'https://www.naturalearthdata.com/downloads/50m-raster-data/50m-natural-earth-1/',
    'license':'Public domain', 'license_page':'https://www.naturalearthdata.com/about/terms-of-use/',
    'packaged_date':'2026-10-05', 'tile_size':512, 'levels':4, 'base_size':[1024,512],
    'processing':'Downsampled to four JPEG tile levels; no invented geographic detail.', 'tiles':files}
(target/'sources.json').write_text(json.dumps(metadata,indent=2), encoding='utf-8')
print('Tiles',len(files), 'bytes',sum(f['bytes'] for f in files))
