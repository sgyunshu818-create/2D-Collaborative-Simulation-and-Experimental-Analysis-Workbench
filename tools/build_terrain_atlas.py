"""Package 25 independent ImageGen terrain sources into a streamed map atlas.

Source files remain unchanged. Normal texture composition feathers neighboring
overlaps, crops native tiles and builds smaller mip levels; no upsampling is used.
Only the application's Pygame dependency is needed, including for packaging.
"""
import argparse
from hashlib import sha256
from io import BytesIO
import json
import math
import os
from pathlib import Path
import sys
import tempfile

os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT','1')
import pygame

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from sim_app import terrain_atlas as atlas


def _load(path):
    payload=path.read_bytes()
    return pygame.image.load(BytesIO(payload),path.name),payload


def _feather(image, overlap, row, col):
    """A left/top source-over ramp, with opaque image interiors and outer edges."""
    width,height=image.get_size()
    tile=pygame.Surface((width,height),pygame.SRCALPHA)
    tile.blit(image,(0,0))
    for horizontal,enabled,extent in ((True,col>0,overlap[0]),(False,row>0,overlap[1])):
        if not enabled or extent == 0:
            continue
        mask=pygame.Surface((width,height),pygame.SRCALPHA)
        mask.fill((255,255,255,255))
        for position in range(extent):
            t=position/max(1,extent-1)
            alpha=round(255*t*t*(3-2*t))
            start=(position,0) if horizontal else (0,position)
            end=(position,height-1) if horizontal else (width-1,position)
            pygame.draw.line(mask,(255,255,255,alpha),start,end)
        tile.blit(mask,(0,0),special_flags=pygame.BLEND_RGBA_MULT)
    return tile


def build(sources, destination, *, overlap=atlas.OVERLAP, world_size=atlas.WORLD_SIZE,
          central_landmarks=None, preview_path=None, horizontal_offsets=None):
    sources,destination=Path(sources).resolve(),Path(destination).resolve()
    inputs=[]
    size=None
    # Preflight before writing a published atlas: missing, wrong-size and repeated
    # assets must not silently become enlarged copies or an incomplete map.
    for row in range(5):
        for col in range(5):
            path=sources/f'r{row}_c{col}.png'
            image,payload=_load(path)
            if size is None:
                size=image.get_size()
                if size[0]*2 != size[1]*3:
                    raise ValueError('Native region sources must have a 3:2 aspect ratio')
            if image.get_size()!=size:
                raise ValueError(f'Native size mismatch: {path.name}: {image.get_size()} vs {size}')
            inputs.append({'row':row,'col':col,'path':path,'bytes':len(payload),
                           'sha256':sha256(payload).hexdigest(),'native_size':list(size)})
    if len({entry['sha256'] for entry in inputs})!=25:
        raise ValueError('The atlas needs 25 distinct source images; repeating one is not new detail')
    if any(n<0 or n>=s for n,s in zip(overlap,size)):
        raise ValueError('Overlap must lie inside the native region size')
    stride=(size[0]-overlap[0],size[1]-overlap[1])
    offsets=horizontal_offsets or {}
    if offsets.get('r2_c2',0)!=0:
        raise ValueError('Keep the central source fixed so saved routes retain their alignment')
    positions={f'r{row}_c{col}':col*stride[0]+int(offsets.get(f'r{row}_c{col}',0))
               for row in range(5) for col in range(5)}
    for row in range(5):
        if positions[f'r{row}_c0']!=0 or positions[f'r{row}_c4']!=4*stride[0]:
            raise ValueError('Outer atlas boundaries must remain fixed')
        for col in range(1,5):
            overlap_x=positions[f'r{row}_c{col-1}']+size[0]-positions[f'r{row}_c{col}']
            if not 0<overlap_x<size[0]:
                raise ValueError('Neighboring sources must overlap without reversing or leaving holes')
    width,height=4*stride[0]+size[0],4*stride[1]+size[1]
    area_scale=width*height/(size[0]*size[1])
    if area_scale<20:
        raise ValueError('The native atlas must contain at least 20 times the previous pixel area')
    image=pygame.Surface((width,height),depth=24)
    image.fill((19,38,42))
    for entry in inputs:
        source,_=_load(entry['path'])
        row,col=entry['row'],entry['col']
        key=f'r{row}_c{col}'
        actual_overlap=(positions[f'r{row}_c{col-1}']+size[0]-positions[key]) if col else overlap[0]
        image.blit(_feather(source,(actual_overlap,overlap[1]),row,col),
                   (positions[key],row*stride[1]))
    if preview_path is not None:
        path=Path(preview_path)
        path.parent.mkdir(parents=True,exist_ok=True)
        preview_width=min(1536,width)
        preview=pygame.transform.smoothscale(image,(preview_width,round(height*preview_width/width)))
        stream=BytesIO()
        pygame.image.save(preview,stream,'preview.png')
        path.write_bytes(stream.getvalue())
    destination.mkdir(parents=True,exist_ok=True)
    levels,tiles=[],[]
    level,current=0,image
    while True:
        w,h=current.get_size()
        cols,rows=math.ceil(w/512),math.ceil(h/512)
        levels.append(dict(width=w,height=h,columns=cols,rows=rows))
        folder=destination/str(level)
        folder.mkdir(exist_ok=True)
        for row in range(rows):
            for col in range(cols):
                rect=pygame.Rect(col*512,row*512,min(512,w-col*512),min(512,h-row*512))
                path=folder/f'{col}_{row}.png'
                stream=BytesIO()
                pygame.image.save(current.subsurface(rect),stream,'tile.png')
                payload=stream.getvalue()
                path.write_bytes(payload)
                tiles.append(dict(name=path.relative_to(destination).as_posix(),size=list(rect.size),
                                  bytes=len(payload),sha256=sha256(payload).hexdigest()))
        if max(w,h)<=512:
            break
        level+=1
        current=pygame.transform.smoothscale(image,(max(1,width//2**level),max(1,height//2**level)))
    region_data=[]
    landmarks={}
    overview=[]
    for row in range(5):
        for col in range(5):
            key=f'r{row}_c{col}'
            center=[(positions[key]+size[0]/2)*world_size[0]/width,
                    (row*stride[1]+size[1]/2)*world_size[1]/height]
            region_data.append(dict(id=key,row=row,col=col,name=atlas.REGION_NAMES[row][col],
                                    center=center,zoom=width/size[0]))
            landmarks[key]=dict(name=atlas.REGION_NAMES[row][col],position=center)
            overview.append(key)
    if central_landmarks is not None:
        old=json.loads(Path(central_landmarks).read_text(encoding='utf-8'))['landmarks']
        for key,item in old.items():
            x,y=item['position']
            position=[(2*stride[0]+x*size[0]/12000)*world_size[0]/width,
                      (2*stride[1]+y*size[1]/8000)*world_size[1]/height]
            landmarks[key]=dict(name=item['name'],position=position)
        overview.extend(key for key in ('lake','valley_town','central_bridge','airport') if key in old)
    source_records=[{**entry,'path':entry['path'].relative_to(ROOT).as_posix()
                     if entry['path'].is_relative_to(ROOT) else str(entry['path'])} for entry in inputs]
    data=dict(name='Astra 广域山地河谷地图',render_mode='photographic',terrain_id='astra_atlas',
              world_size=list(world_size),raster_size=[width,height],tile_size=512,
              levels=levels,tiles=tiles,features=[],feature_counts={},landmarks=landmarks,
              overview_landmarks=overview,label_cache_limit=64,label_scale=width/size[0],
              regions=region_data,atlas_grid=[5,5],overlap=list(overlap),
              horizontal_offsets=offsets,
              region_native_size=list(size),native_imagery_area_scale=area_scale,
              source_tiles=source_records,
              processing='Native source-over texture composition at overlaps; source files unchanged; native crops and smaller mip levels; no upsampling.')
    descriptor,staged=tempfile.mkstemp(prefix='atlas-',suffix='.json.tmp',dir=destination)
    try:
        with os.fdopen(descriptor,'w',encoding='utf-8') as stream:
            json.dump(data,stream,ensure_ascii=False,indent=2)
            stream.write('\n')
        os.replace(staged,destination/'terrain.json')
    finally:
        if Path(staged).exists():
            Path(staged).unlink()
    return data


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sources',type=Path,default=ROOT/'sim_app/assets/terrain_atlas/sources')
    parser.add_argument('--destination',type=Path,default=atlas.ASSET_DIR)
    parser.add_argument('--preview',type=Path,default=ROOT/'artifacts/terrain_atlas_20261006/atlas-overview.png')
    parser.add_argument('--alignment',type=Path,help='JSON with measured horizontal_offsets for interior sources')
    args=parser.parse_args()
    result=build(args.sources,args.destination,preview_path=args.preview,
                 central_landmarks=ROOT/'sim_app/assets/terrain_astra/runtime/terrain.json',
                 horizontal_offsets=json.loads(args.alignment.read_text(encoding='utf-8'))['horizontal_offsets']
                 if args.alignment else None)
    print(json.dumps({key:result[key] for key in ('world_size','raster_size','native_imagery_area_scale')},ensure_ascii=False))
    print(json.dumps({'source_images':len(result['source_tiles']),'levels':len(result['levels']),
                      'tiles':len(result['tiles']),'tile_bytes':sum(tile['bytes'] for tile in result['tiles'])}))


if __name__=='__main__':
    main()
