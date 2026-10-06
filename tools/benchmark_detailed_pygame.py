"""Short, real Windows SDL App.run timing sample; production files are untouched.

This opens its own 1280 x 800 window and closes it after 440 frames. It measures
Renderer.draw and draw-start intervals, not monitor scanout or long-term FPS.
Run from Exp with .venv/Scripts/python.exe tools/benchmark_detailed_pygame.py.
"""
from datetime import datetime, timezone
from pathlib import Path
import csv
import hashlib
import json
import math
import os
import platform
import statistics
import sys
from time import monotonic, perf_counter

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT","1")
# This benchmark must exercise the actual Windows window backend.
os.environ.pop("SDL_VIDEODRIVER",None)
os.environ.pop("SDL_AUDIODRIVER",None)

import pygame
from sim_app.app import App
from sim_app.dpi_support import enable_native_dpi
from sim_app.models import Point
from sim_app.renderer import Renderer, WINDOW_SIZE
from sim_app.scene import load_scene
from sim_app.simulation import Simulation
from sim_app import detailed_terrain
from sim_app.visual_assets import set_pygame_icon

ARTIFACT=ROOT/"artifacts"/"detailed_map_20261006"
SCENE_PATH=ROOT/"configs"/"detailed_scene.json"
WARMUP=20
SAMPLES=200
PHASE_LENGTH=WARMUP+SAMPLES


def distribution(values):
    values=sorted(values)
    def quantile(p):
        position=(len(values)-1)*p
        lower=math.floor(position)
        upper=math.ceil(position)
        return values[lower]+(values[upper]-values[lower])*(position-lower)
    return {"count":len(values),"median_ms":statistics.median(values),
            "p95_ms":quantile(.95),"maximum_ms":max(values),"minimum_ms":min(values),
            "mean_ms":statistics.mean(values)}


def main(*, artifact=ARTIFACT, scene_path=SCENE_PATH, phase_specs=None):
    ARTIFACT,SCENE_PATH=Path(artifact),Path(scene_path)
    if platform.system()!="Windows":
        raise RuntimeError("This experiment explicitly requires the Windows SDL backend")
    ARTIFACT.mkdir(parents=True,exist_ok=True)
    enable_native_dpi()
    pygame.display.init()
    pygame.font.init()
    pygame.display.set_caption("详细地形短时性能抽样 · 自动关闭")
    set_pygame_icon()
    surface=pygame.display.set_mode(WINDOW_SIZE)
    driver=pygame.display.get_driver()
    if driver in ("dummy","offscreen"):
        raise RuntimeError(f"Expected a real window, received SDL driver {driver}")
    scene=load_scene(SCENE_PATH)
    scene_digest=hashlib.sha256(SCENE_PATH.read_bytes()).hexdigest()
    simulation=Simulation(scene)
    if len(simulation.units)!=8:
        raise RuntimeError("The sample must use all 8 units in the new detailed scene")
    renderer=Renderer(surface)
    renderer._prepare_map(simulation)
    renderer.display_action("map_maximize",simulation)
    renderer._prepare_map(simulation)
    app=App(simulation,renderer,SCENE_PATH,ARTIFACT/"pygame_detailed_runs")
    simulation.start()
    app._sync_presentation()
    rows=[]
    originals={"draw":renderer.draw,"flip":pygame.display.flip}
    frame_index=0
    last_start=None
    started=perf_counter()
    wall_start=datetime.now(timezone.utc).isoformat()
    phase_specs=phase_specs or [("expanded_overview",1.0,Point(scene.width/2,scene.height/2)),
                 ("expanded_town_near_8x",8.0,Point(scene.width*.653333,scene.height*.34))]
    if len(phase_specs)!=2:
        raise ValueError('This short sample expects exactly two camera cases')

    def measured_draw(sim,notice,mouse_pos):
        nonlocal frame_index,last_start
        phase_index=frame_index//PHASE_LENGTH
        local_index=frame_index%PHASE_LENGTH
        phase,base_zoom,center=phase_specs[phase_index]
        if local_index==0:
            renderer.camera.set_pose(base_zoom,center)
            last_start=None
        # Drive the real bounded camera: continuous panning, interrupted smooth
        # anchored zooms. Every frame changes the visible terrain rectangle.
        zoom_starts=(28,73,118,163,208)
        zoom_in_progress=any(0<=local_index-start<12 for start in zoom_starts)
        if local_index>0 and not zoom_in_progress:
            renderer.camera.pan((math.cos(local_index/17)*1.1,
                                 math.sin(local_index/19)*.65),monotonic())
        if local_index in zoom_starts:
            target=base_zoom*(1.10 if local_index in (28,118,208) else 1.0)
            renderer.camera.zoom_by(target/renderer.camera.target_zoom,
                                    renderer.camera.screen_center,monotonic())
        begin=perf_counter()
        originals["draw"](sim,notice,mouse_pos)
        end=perf_counter()
        rect=renderer.camera.visible_world()
        rows.append({"frame_index":frame_index,"phase":phase,"phase_frame":local_index,
                     "warmup":local_index<WARMUP,"sample_frame":local_index-WARMUP,
                     "draw_start_seconds":begin-started,"draw_ms":(end-begin)*1000,
                     "frame_interval_ms":(begin-last_start)*1000 if last_start is not None else None,
                     "display_flip_ms":None,"zoom":renderer.camera.zoom,
                     "center_x":renderer.camera.actual_center.x,"center_y":renderer.camera.actual_center.y,
                     "view_left":rect[0],"view_top":rect[1],"view_width":rect[2],"view_height":rect[3],
                     "visible_unit_hit_rects":sum(rect.colliderect(renderer.map_rect)
                                                   for rect in renderer.unit_hit_rects.values()),
                     "simulation_steps":sim.step_count})
        last_start=begin
        frame_index+=1
        if frame_index>=2*PHASE_LENGTH:
            app.active=False

    def measured_flip():
        begin=perf_counter()
        originals["flip"]()
        rows[-1]["display_flip_ms"]=(perf_counter()-begin)*1000

    renderer.draw=measured_draw
    pygame.display.flip=measured_flip
    try:
        result=app.run()
        elapsed=perf_counter()-started
        backend={"pygame":pygame.version.ver,"sdl":list(pygame.get_sdl_version()),
                 "driver":driver,"display_size":list(surface.get_size()),
                 "map_rectangle":list(renderer.map_rect),
                 "platform":platform.platform(),"python":platform.python_version(),
                 "processor":platform.processor(),"display_flags":surface.get_flags(),
                 "display_bitsize":surface.get_bitsize()}
        cache=detailed_terrain.cache_info()
    finally:
        pygame.display.flip=originals["flip"]
        renderer.draw=originals["draw"]
        pygame.quit()
    output={"experiment":"Real Windows SDL short App.run detailed-terrain timing sample",
            "started_utc":wall_start,"elapsed_seconds_including_final_save":elapsed,
            "app_exit_code":result,"backend":backend,
            "scene":str(SCENE_PATH),"scene_sha256":scene_digest,
            "unit_count":len(simulation.units),"unit_ids":[u.id for u in simulation.units],
            "window_count_created":1,"window_closed_after_sample":True,
            "warmup_frames_per_case":WARMUP,"recorded_frames_per_case":SAMPLES,
            "application_loop":"Unmodified App.run; pygame.time.Clock.tick(60); live Simulation.advance; Renderer.draw; pygame.display.flip",
            "measurement_definition":{
                "draw_ms":"Wall duration of Renderer.draw only; excludes scripted camera calls, display.flip, simulation.advance, and Clock.tick waiting.",
                "frame_interval_ms":"Difference between adjacent Renderer.draw start timestamps; includes App.run pacing, simulation, event handling, prior draw and display.flip. This is not monitor scanout timing.",
                "display_flip_ms":"Wall duration of the genuine SDL pygame.display.flip call.",
                "p95":"Linear interpolation of the sorted samples at (n - 1) * 0.95; warmup rows excluded."},
            "scope_note":"仅本机、本次配置的短时200帧抽样；不能据此推断长期稳定60帧、其它机器性能或屏幕实际呈现帧率。",
            "camera_script":"Each case pans for 159 of 220 frames and animates five anchored zoom commands over 12-frame stretches. Camera origins are listed in cases; zoom spans base_zoom to 1.1 times base_zoom. All 8 example units remain in the live simulation; viewports clip units outside their bounds.",
            "cache_at_end":cache,"cases":[]}
    for name,base_zoom,center in phase_specs:
        sample=[row for row in rows if row["phase"]==name and not row["warmup"]]
        if len(sample)!=SAMPLES:
            raise RuntimeError(f"Incomplete case {name}: {len(sample)} of {SAMPLES}")
        output["cases"].append({"name":name,"base_zoom":base_zoom,
                                "origin_world":[center.x,center.y],
                                "draw":distribution([r["draw_ms"] for r in sample]),
                                "frame_interval":distribution([r["frame_interval_ms"] for r in sample]),
                                "display_flip":distribution([r["display_flip_ms"] for r in sample]),
                                "zoom_range":[min(r["zoom"] for r in sample),max(r["zoom"] for r in sample)],
                                "visible_unit_hit_rect_range":[min(r["visible_unit_hit_rects"] for r in sample),max(r["visible_unit_hit_rects"] for r in sample)]})
    (ARTIFACT/"pygame_detailed_timing.json").write_text(json.dumps(output,ensure_ascii=False,indent=2),encoding="utf-8")
    (ARTIFACT/"pygame_detailed_frames.json").write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding="utf-8")
    with (ARTIFACT/"pygame_detailed_frames.csv").open("w",encoding="utf-8-sig",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    print(json.dumps(output,ensure_ascii=False,indent=2))
    return result


if __name__=="__main__":
    raise SystemExit(main())
