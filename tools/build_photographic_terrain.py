"""Package unmodified ImageGen artwork into native-resolution offline tiles.

No upscaling, relief generation or image-content editing. Only smaller mip
levels and crops are produced for the existing bounded viewport renderer.
Requires only the application's Pygame dependency.
"""
import argparse
from hashlib import sha256
from io import BytesIO
import json
import math
import os
from pathlib import Path
import tempfile

os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')
import pygame

ROOT = Path(__file__).resolve().parents[1]
WORLD = (12000, 8000)
TILE_SIZE = 512


def package(source, destination, landmarks=None):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    payload = source.read_bytes()
    original = pygame.image.load(BytesIO(payload), source.name)
    width, height = original.get_size()
    if width * 2 != height * 3:
        raise ValueError('The world requires a native 3:2 source; do not stretch the artwork.')
    # A RGB copy preserves the source pixels while discarding unused alpha.
    image = pygame.Surface(original.get_size(), depth=24)
    image.blit(original, (0, 0))
    destination.mkdir(parents=True, exist_ok=True)
    levels, tiles = [], []
    current = image
    level = 0
    while True:
        w, h = current.get_size()
        columns, rows = math.ceil(w / TILE_SIZE), math.ceil(h / TILE_SIZE)
        levels.append(dict(width=w, height=h, columns=columns, rows=rows))
        folder = destination / str(level)
        folder.mkdir(exist_ok=True)
        for row in range(rows):
            for col in range(columns):
                rect = pygame.Rect(col*TILE_SIZE, row*TILE_SIZE,
                                   min(TILE_SIZE, w-col*TILE_SIZE), min(TILE_SIZE, h-row*TILE_SIZE))
                path = folder / f'{col}_{row}.png'
                # Byte streams support non-ASCII project paths on Windows.
                stream = BytesIO()
                pygame.image.save(current.subsurface(rect), stream, 'tile.png')
                data = stream.getvalue()
                path.write_bytes(data)
                tiles.append(dict(name=path.relative_to(destination).as_posix(),
                                  size=rect.size, bytes=len(data), sha256=sha256(data).hexdigest()))
        if max(w, h) <= TILE_SIZE:
            break
        level += 1
        current = pygame.transform.smoothscale(image, (max(1, width//2**level), max(1, height//2**level)))
    if landmarks is None:
        landmark_data = {}
    else:
        landmark_data = json.loads(Path(landmarks).read_text(encoding='utf-8-sig'))
        if 'landmarks' in landmark_data:
            landmark_data = landmark_data['landmarks']
        if isinstance(landmark_data, list):
            aliases = {'qinglan_lake':'lake', 'hegu_town':'valley_town',
                       'east_bank_town':'riverbank_town', 'central_road_bridge':'central_bridge',
                       'hegu_airport':'airport', 'northeast_village':'northern_village',
                       'south_road_bridge':'southern_bridge'}
            landmark_data = {aliases[item['id']]:dict(name=item['name'],position=item['world'])
                             for item in landmark_data if item['id'] in aliases}
        for key, item in landmark_data.items():
            x,y = item['position']
            if not (isinstance(item['name'], str) and 0<=x<=WORLD[0] and 0<=y<=WORLD[1]):
                raise ValueError(f'Invalid world landmark {key}')
    metadata = dict(name='Astra 山区河谷试验区', render_mode='photographic',
                    world_size=list(WORLD), raster_size=[width,height], native_source_size=[width,height],
                    tile_size=TILE_SIZE, levels=levels, features=[], landmarks=landmark_data,
                    feature_counts={}, source=dict(path=source.relative_to(ROOT).as_posix()
                                                   if source.is_relative_to(ROOT) else str(source),
                                                   bytes=len(payload), sha256=sha256(payload).hexdigest()),
                    tiles=tiles, processing='Original artwork unchanged; native tile crops and downsampled mip levels only.')
    # Publish only after every tile is ready, so new windows cannot see a partial map.
    descriptor, staged = tempfile.mkstemp(prefix='terrain-', suffix='.json.tmp', dir=destination)
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8') as stream:
            json.dump(metadata, stream, ensure_ascii=False, indent=2)
            stream.write('\n')
        os.replace(staged, destination / 'terrain.json')
    finally:
        if Path(staged).exists():
            Path(staged).unlink()
    return metadata


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT/'sim_app/assets/terrain_astra/terrain_astra.png')
    parser.add_argument('--destination', type=Path, default=ROOT/'sim_app/assets/terrain_astra/runtime')
    parser.add_argument('--landmarks', type=Path, default=ROOT/'sim_app/assets/terrain_astra/landmarks.json')
    args=parser.parse_args()
    data=package(args.source,args.destination,args.landmarks)
    print(json.dumps(dict(source_pixels=data['native_source_size'],levels=len(data['levels']),
                          tiles=len(data['tiles']),tile_bytes=sum(t['bytes'] for t in data['tiles']),
                          destination=str(args.destination)),ensure_ascii=False))


if __name__=='__main__':
    main()
