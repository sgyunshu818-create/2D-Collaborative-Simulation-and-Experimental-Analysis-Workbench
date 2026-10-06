import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from sim_app.geography import GeoReference, distance_m, scene_world
from sim_app.models import Point
from sim_app.recording import export_run
from sim_app.replay import load_replay
from sim_app.scene import SceneConfigError, load_scene, scene_from_data
from sim_app.simulation import Simulation
from sim_app.world_map import WorldMapView, _tile

ROOT = Path(__file__).resolve().parents[1]


class GeographyTests(unittest.TestCase):
    def setUp(self):
        self.data=json.loads((ROOT/'configs/geographic_scene.json').read_text(encoding='utf-8'))
        self.scene=scene_from_data(self.data,'geo')

    def test_reference_roundtrip_and_direction(self):
        ref=self.scene.georeference
        self.assertEqual(ref.to_lonlat(600000,400000,1200000,800000),(12.5,43.5))
        for x,y in [(0,0),(1199999,799999),(250000,600000)]:
            lon,lat=ref.to_lonlat(x,y,self.scene.width,self.scene.height)
            actual=ref.to_local(lon,lat,self.scene.width,self.scene.height)
            self.assertAlmostEqual(x,actual[0],places=6)
            self.assertAlmostEqual(y,actual[1],places=6)
        west,south,east,north=ref.bounds(self.scene.width,self.scene.height)
        self.assertLess(west,east);self.assertLess(south,north)
        self.assertLess(west,12.5);self.assertGreater(north,43.5)

    def test_invalid_reference_and_dateline_rejected(self):
        for key,value in [('center_latitude',True),('center_latitude',90),('center_longitude',181),
                          ('meters_per_unit',0),('meters_per_unit',float('nan'))]:
            data=copy.deepcopy(self.data);data['world']['georeference'][key]=value
            with self.assertRaises(SceneConfigError):scene_from_data(data,'geo')
        self.data['world']['georeference']['center_longitude']=179
        with self.assertRaisesRegex(SceneConfigError,'180'):scene_from_data(self.data,'geo')

    def test_old_scenes_have_no_invented_geography(self):
        old=load_scene(ROOT/'configs/basic_scene.json')
        self.assertIsNone(old.georeference)
        self.assertNotIn('georeference',scene_world(old))
        view=WorldMapView()
        self.assertIsNone(view.local_to_screen(80,150,scene_world(old)))
        self.assertFalse(view.focus_scene(scene_world(old)))

    def test_export_and_replay_keep_reference(self):
        sim=Simulation(self.scene);sim.start();sim.advance(.3)
        original=sim.snapshot()
        with tempfile.TemporaryDirectory() as folder:
            paths=export_run(sim,folder)
            replay=load_replay(paths['run'])
            self.assertEqual(self.scene.georeference,replay.scene.georeference)
            self.assertEqual(scene_world(self.scene),scene_world(replay.scene))
        self.assertEqual(original,sim.snapshot())

    def test_atlas_zoom_pan_resize_hit_and_bounded_tiles(self):
        import pygame
        world=scene_world(self.scene)
        row=copy.deepcopy(self.data['units'][0])
        view=WorldMapView();view.resize((1200,580));view.focus_scene(world)
        position=view.local_to_screen(row['x'],row['y'],world)
        view.render(world,[row],row['id'])
        self.assertEqual(view.pick_unit(position),row['id'])
        actual=view.screen_to_local(position,world)
        self.assertAlmostEqual(actual[0],row['x']);self.assertAlmostEqual(actual[1],row['y'])
        view.pan((120,60))
        moved=view.local_to_screen(row['x'],row['y'],world)
        self.assertAlmostEqual(moved[0]-position[0],120)
        self.assertAlmostEqual(moved[1]-position[1],60)
        center=view.camera.actual_center
        view.resize((760,420))
        self.assertEqual(center,view.camera.actual_center)
        view.render(world,[row])
        self.assertLessEqual(_tile.cache_info().currsize,48)
        self.assertEqual(view._last_surface.get_size(),(760,420))
        self.assertTrue(view.ppm().startswith(b'P6\n760 420\n255\n'))
        self.assertEqual(row,self.data['units'][0])

    def test_global_raster_landmarks_use_correct_projection(self):
        import pygame
        view=WorldMapView();view.resize((1024,512));view.focus_global(animate=False)
        self.assertEqual(view.to_screen(-180,90),(0,0))
        self.assertEqual(view.to_screen(180,-90),(1024,512))
        self.assertEqual(view.to_screen(0,0),(512,256))
        surface=view.render({},show_grid=False)
        self.assertTrue((ROOT/'sim_app/assets/world/0_0_0.jpg').is_file())
        # Physical dataset has land at the Sahara and water in the Atlantic.
        land=surface.get_at(tuple(round(v) for v in view.to_screen(20,22)))
        water=surface.get_at(tuple(round(v) for v in view.to_screen(-30,22)))
        self.assertNotEqual(land,water)
        self.assertGreater(land.r,water.r)

    def test_geographic_distance_varies_with_latitude(self):
        equator=distance_m(0,0,1,0);north=distance_m(0,60,1,60)
        self.assertAlmostEqual(equator,111195,delta=10)
        self.assertAlmostEqual(north/equator,.5,delta=.001)

    def test_map_text_survives_graphics_shutdown_and_reinitialization(self):
        # Exercise the native resource lifetime in a child process so a stale
        # SDL_ttf handle produces a reported failure instead of killing QA.
        import subprocess,sys
        code = '''
import pygame
from sim_app.world_map import WorldMapView
v=WorldMapView()
v.render({})
pygame.quit()
pygame.font.init()
v.render({},title='new text')
w=WorldMapView()
w.render({})
pygame.quit()
'''
        result=subprocess.run([sys.executable,'-c',code],cwd=ROOT,capture_output=True,timeout=15)
        self.assertEqual(result.returncode,0,result.stderr.decode('utf-8',errors='replace'))

    def test_high_zoom_texture_tracks_one_pixel_pan_across_crop_boundaries(self):
        import pygame
        # Deliberately sharp source detail makes a discontinuous crop visible.
        # The same geography should move by one screen pixel with the camera.
        tile=pygame.Surface((512,512))
        for x in range(512):
            color=255 if (x//8)%2 else 0
            pygame.draw.line(tile,(color,color,color),(x,0),(x,511))
        view=WorldMapView();view.resize((1024,512))
        view.camera.set_pose(256,Point(180.07,90.11))
        largest_difference=0
        with patch('sim_app.world_map._tile',return_value=tile):
            previous=view._base(True,False).copy()
            for _ in range(79):
                view.pan((1,0))
                current=view._base(True,False).copy()
                for x in range(20,1000):
                    difference=abs(previous.get_at((x-1,250)).r-current.get_at((x,250)).r)
                    largest_difference=max(largest_difference,difference)
                previous=current
        # Allow small bilinear/rounding residuals; the former crop stretching
        # produced a 165-level jump when its integer source edge changed.
        self.assertLessEqual(largest_difference,16)

    def test_observation_range_projects_local_circle_and_does_not_spill_outside_scene(self):
        import pygame
        world={'width':900,'height':600,'georeference':{
            'center_latitude':60,'center_longitude':20,'meters_per_unit':150}}
        row={'id':'range_test','type':'ground','team':'red','x':450,'y':300,'sensor_range':150}
        view=WorldMapView();view.resize((600,400));view.focus_scene(world)
        with patch('pygame.draw.ellipse',wraps=pygame.draw.ellipse) as ellipses:
            view.render(world,[row],row['id'],show_grid=False,show_terrain=False,show_labels=False)
        self.assertEqual(len(ellipses.call_args_list),1)
        bounds=pygame.Rect(ellipses.call_args.args[2])
        # Read the drawn cardinal extrema back into the simulation coordinate
        # system. Both axes must describe the same observation radius in u.
        for pixel in ((bounds.left,bounds.centery),(bounds.right,bounds.centery),
                      (bounds.centerx,bounds.top),(bounds.centerx,bounds.bottom)):
            x,y=view.screen_to_local(pixel,world)
            radius=((x-row['x'])**2+(y-row['y'])**2)**.5
            self.assertAlmostEqual(radius,row['sensor_range'],delta=3)
        self.assertAlmostEqual(bounds.height/bounds.width,.5,delta=.02)

        # A selected unit near the boundary must not paint a sensor overlay
        # over geographic land outside the actual simulation rectangle.
        row['x']=880
        without=view.render(world,[dict(row,sensor_range=0)],row['id'],
                            show_grid=False,show_terrain=False,show_labels=False).copy()
        with_range=view.render(world,[row],row['id'],
                               show_grid=False,show_terrain=False,show_labels=False)
        scene=view.scene_rect
        outside_changes=inside_changes=0
        for y in range(view.size[1]):
            for x in range(view.size[0]):
                if without.get_at((x,y)) != with_range.get_at((x,y)):
                    if scene.collidepoint(x,y):
                        inside_changes+=1
                    else:
                        outside_changes+=1
        self.assertGreater(inside_changes,0)
        self.assertEqual(outside_changes,0)

    def test_scale_bar_matches_distance_at_actual_high_latitude_and_hides_at_pole(self):
        import pygame
        view=WorldMapView();view.resize((600,400))
        for latitude in (0,60,85):
            with self.subTest(latitude=latitude):
                view.camera.set_pose(100,Point(200,90-latitude))
                surface=pygame.Surface(view.size)
                with patch.object(view,'_text') as text, patch('pygame.draw.line',wraps=pygame.draw.line) as lines:
                    view._scale_bar(surface)
                label=text.call_args.args[1]
                value,unit=label.split(' · ')[0].split()
                labeled_meters=float(value)*(1000 if unit=='km' else 1)
                horizontal=[call.args for call in lines.call_args_list
                            if call.args[2][1]==call.args[3][1] and call.args[2][0]!=call.args[3][0]]
                self.assertEqual(len(horizontal),1)
                length=abs(horizontal[0][3][0]-horizontal[0][2][0])
                # The bar refers to centre latitude. Verify its pixel length
                # with two inverse-projected points on that latitude.
                start=view.to_lonlat((300,200));end=view.to_lonlat((300+length,200))
                actual_meters=distance_m(*start,*end)
                one_pixel=view.to_lonlat((301,200))
                self.assertAlmostEqual(actual_meters,labeled_meters,
                                       delta=distance_m(*start,*one_pixel)+1)
        view.camera.set_pose(100,Point(200,0))
        with patch.object(view,'_text') as text, patch('pygame.draw.line') as lines:
            view._scale_bar(pygame.Surface(view.size))
        self.assertFalse(lines.called)
        self.assertIn('不可用',text.call_args.args[1])


if __name__=='__main__':unittest.main()
