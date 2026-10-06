"""Move the previous demonstration into the preserved central atlas region."""
import argparse
from copy import deepcopy
import json
from math import sqrt
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from sim_app import terrain_atlas as atlas
from sim_app.scene import scene_from_data


def enlarge(data):
    if (data['world']['width'],data['world']['height'])!=(12000,8000):
        raise ValueError('Use the previous 12000×8000 source scene; do not migrate an atlas twice')
    result=deepcopy(data)
    result['name']='广域山地河谷协同巡检 · 25 分区'
    result['world']={**result['world'],'width':atlas.WORLD_SIZE[0],
                     'height':atlas.WORLD_SIZE[1],'terrain':'astra_atlas'}
    if result['world'].get('georeference') is not None:
        raise ValueError('Atlas migration requires fictional local coordinates, not a geographic source')
    def migrate(point):
        point['x'],point['y']=atlas.source_to_world(point['x'],point['y'])
    for field in ('spawn_points','return_points'):
        for point in result[field].values():
            migrate(point)
    scale=sqrt((atlas.SOURCE_SIZE[0]/12000*atlas.WORLD_SIZE[0]/atlas.RASTER_SIZE[0])*
               (atlas.SOURCE_SIZE[1]/8000*atlas.WORLD_SIZE[1]/atlas.RASTER_SIZE[1]))
    for unit in result['units']:
        migrate(unit)
        for point in unit['waypoints']:
            migrate(point)
        for field in ('speed','sensor_range'):
            if field in unit:
                unit[field]=round(unit[field]*scale,6)
    for obstacle in result['obstacles']:
        migrate(obstacle)
        obstacle['width']*=atlas.SOURCE_SIZE[0]/12000*atlas.WORLD_SIZE[0]/atlas.RASTER_SIZE[0]
        obstacle['height']*=atlas.SOURCE_SIZE[1]/8000*atlas.WORLD_SIZE[1]/atlas.RASTER_SIZE[1]
    scene_from_data(result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=ROOT/'configs/mountain_scene.json')
    parser.add_argument('--output',type=Path,default=ROOT/'configs/atlas_scene.json')
    parser.add_argument('--activate',action='store_true',help='Replace the default only after the complete atlas is available')
    args=parser.parse_args()
    if args.activate and not atlas.available():
        raise ValueError('Build all atlas assets before activating the new default')
    data=enlarge(json.loads(args.source.read_text(encoding='utf-8')))
    output=ROOT/'configs/detailed_scene.json' if args.activate else args.output
    output.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'scene':data['name'],'world':data['world'],'output':str(output)},ensure_ascii=False))


if __name__=='__main__':
    main()
