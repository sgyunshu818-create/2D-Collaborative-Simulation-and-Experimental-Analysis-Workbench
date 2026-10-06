"""Run the real Windows SDL atlas sample with verified input provenance."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.benchmark_detailed_pygame import main as run_sample
from sim_app.models import Point
from sim_app.terrain_atlas import DETAIL_ZOOM, WORLD_SIZE

ARTIFACT=ROOT/'artifacts/terrain_atlas_20261006'
SCENE_PATH=ROOT/'configs/detailed_scene.json'
SOURCE_PATHS={
    'scene':SCENE_PATH,
    'atlas_descriptor':ROOT/'sim_app/assets/terrain_atlas/runtime/terrain.json',
    'wrapper':Path(__file__).resolve(),
    'benchmark_driver':ROOT/'tools/benchmark_detailed_pygame.py',
}


def source_records(*,allow_read_errors=False):
    records={}
    for role,path in SOURCE_PATHS.items():
        try:
            payload=path.read_bytes()
        except OSError as error:
            if not allow_read_errors:
                raise
            records[role]={'path':path.relative_to(ROOT).as_posix(),
                           'read_error':f'{type(error).__name__}: {error}'}
            continue
        records[role]={'path':path.relative_to(ROOT).as_posix(),
                       'bytes':len(payload),'sha256':hashlib.sha256(payload).hexdigest()}
    return records


def main():
    before=source_records()
    preflight_utc=datetime.now(timezone.utc).isoformat()
    result=run_sample(artifact=ARTIFACT,scene_path=SCENE_PATH,
        phase_specs=[('expanded_atlas_overview',1.0,Point(WORLD_SIZE[0]/2,WORLD_SIZE[1]/2)),
                     ('expanded_atlas_central_valley',DETAIL_ZOOM,Point(WORLD_SIZE[0]/2,WORLD_SIZE[1]/2))])
    after=source_records(allow_read_errors=True)
    postflight_utc=datetime.now(timezone.utc).isoformat()
    changed=[role for role in before if before[role]!=after[role]]
    timing_path=ARTIFACT/'pygame_detailed_timing.json'
    output=json.loads(timing_path.read_text(encoding='utf-8'))
    measured_scene_matches=output.get('scene_sha256')==before['scene']['sha256']
    unchanged=not changed and measured_scene_matches
    output['measurement_sources']={
        'schema_version':1,'preflight_utc':preflight_utc,'postflight_utc':postflight_utc,
        'before':before,'after':after,'unchanged':unchanged,
        'changed_roles':changed,'measured_scene_sha256_matches_preflight':measured_scene_matches,
        'status':'validated' if unchanged else 'invalid_source_changed',
        'scope':'Scene, atlas descriptor, wrapper and shared benchmark driver are hashed immediately before and after the sample. Descriptor SHA256 identifies its recorded runtime tile/source hashes; this check is not a complete runtime tile rehash or a cold-disk benchmark.',
    }
    timing_path.write_text(json.dumps(output,ensure_ascii=False,indent=2),encoding='utf-8')
    if not unchanged:
        raise RuntimeError('Benchmark sources changed during measurement; timing evidence marked invalid')
    print('ATLAS_PROVENANCE_OK scene/descriptor/wrapper/benchmark_driver SHA256 unchanged')
    return result


if __name__=='__main__':
    raise SystemExit(main())
