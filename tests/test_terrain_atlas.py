"""Atlas source integrity, overlap coverage, streaming and scene migration."""
from copy import deepcopy
from io import BytesIO
import json
import os
from pathlib import Path
import tempfile
import tkinter as tk
import unittest
from unittest.mock import patch

os.environ.setdefault('SDL_VIDEODRIVER','dummy')
os.environ.setdefault('SDL_AUDIODRIVER','dummy')
os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT','1')
import pygame

from sim_app import detailed_terrain as tiles, terrain_atlas as atlas
from sim_app.models import Point
from sim_app.renderer import Renderer, WINDOW_SIZE
from sim_app.scene import scene_from_data
from sim_app.simulation import Simulation
from sim_app.scene_editor import SceneEditor
from sim_app.tk_runtime import create_root
from tools.build_terrain_atlas import build
from tools.enlarge_atlas_scene import enlarge

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'configs/mountain_scene.json'


class AtlasPackagingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory=tempfile.TemporaryDirectory()
        cls.sources=Path(cls.directory.name)/'sources'
        cls.runtime=Path(cls.directory.name)/'runtime'
        cls.sources.mkdir()
        for row in range(5):
            for col in range(5):
                source=pygame.Surface((192,128))
                source.fill((40+row*32,50+col*30,100+row*5+col))
                pygame.draw.line(source,(213,217,203),(0,64),(191,64),3)
                stream=BytesIO()
                pygame.image.save(source,stream,'source.png')
                (cls.sources/f'r{row}_c{col}.png').write_bytes(stream.getvalue())
        cls.before={p.name:p.read_bytes() for p in cls.sources.iterdir()}
        cls.data=build(cls.sources,cls.runtime,overlap=(16,12),
                       horizontal_offsets={'r1_c2':-8,'r3_c2':6})
        cls.reference=pygame.Surface(tuple(cls.data['raster_size']))
        for tile in cls.data['tiles']:
            path=Path(tile['name'])
            if path.parts[0]!='0':
                continue
            x,y=map(int,path.stem.split('_'))
            cls.reference.blit(pygame.image.load(BytesIO((cls.runtime/path).read_bytes()),'tile.png'),(x*512,y*512))

    @classmethod
    def tearDownClass(cls):
        tiles.clear_cache()
        cls.directory.cleanup()

    def setUp(self):
        atlas.clear_cache()
        self.addCleanup(atlas.clear_cache)
        patcher=patch.object(atlas,'ASSET_DIR',self.runtime)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_twenty_times_native_area_distinct_sources_and_original_bytes(self):
        self.assertEqual(self.data['world_size'],[240000,160000])
        self.assertEqual(self.data['raster_size'],[896,592])
        self.assertGreaterEqual(self.data['native_imagery_area_scale'],20)
        self.assertEqual(len(self.data['source_tiles']),25)
        self.assertEqual(len({t['sha256'] for t in self.data['source_tiles']}),25)
        for path in self.sources.iterdir():
            self.assertEqual(path.read_bytes(),self.before[path.name])
        for level in self.data['levels']:
            self.assertLessEqual(level['width'],896)
            self.assertLessEqual(level['height'],592)

    def test_overlap_translation_keeps_cores_and_outer_boundaries_covered(self):
        # Neither the renderer's background nor an unfilled pixel may appear.
        for y in range(0,592,11):
            for x in range(0,896,13):
                self.assertNotEqual(tuple(self.reference.get_at((x,y)))[:3],tiles.BACKGROUND)
        for row in range(5):
            for col in range(5):
                offset=self.data['horizontal_offsets'].get(f'r{row}_c{col}',0)
                x,y=col*176+offset+96,row*116+72
                self.assertEqual(tuple(self.reference.get_at((x,y)))[:3],
                                 (40+row*32,50+col*30,100+row*5+col))

    def test_pan_across_native_tile_edges_and_scaled_world_uses_same_pixels(self):
        with patch.object(tiles,'_draw_labels'):
            a=atlas.terrain_view_surface((240000,160000),(300,220),
                 (480*240000/896,330*160000/592,300*240000/896,220*160000/592))
            b=atlas.terrain_view_surface((480000,320000),(300,220),
                 (500*480000/896,330*320000/592,300*480000/896,220*320000/592))
        self.assertEqual(pygame.image.tobytes(a,'RGB'),
                         pygame.image.tobytes(self.reference.subsurface((480,330,300,220)),'RGB'))
        self.assertEqual(pygame.image.tobytes(a.subsurface((20,0,280,220)),'RGB'),
                         pygame.image.tobytes(b.subsurface((0,0,280,220)),'RGB'))

    def test_overview_labels_and_continuous_navigation_keep_caches_bounded(self):
        # All 25 labels can be visible together: the old 24-bitmap cap was unsafe.
        result=atlas.terrain_view_surface((240000,160000),(896,592),(0,0,240000,160000))
        self.assertIsNotNone(result)
        self.assertEqual(atlas.cache_info()['label_bitmaps'],25)
        for zoom in (1,4.666667,16,64,256):
            for region in atlas.regions():
                x,y=region['center']
                atlas.terrain_view_ppm(atlas.WORLD_SIZE,(240,160),
                    (x-120000/zoom,y-80000/zoom,240000/zoom,160000/zoom))
        cache=atlas.cache_info()
        self.assertLessEqual(cache['tiles'],48)
        self.assertEqual(cache['pose_surfaces'],1)
        self.assertLessEqual(cache['label_bitmaps'],64)
        self.assertLessEqual(cache['ppm_bytes'],240*160*3+40)

    def test_preflight_failure_does_not_publish_an_invalid_atlas(self):
        descriptor=(self.runtime/'terrain.json').read_bytes()
        target=self.sources/'r4_c4.png'
        saved=target.read_bytes()
        try:
            target.write_bytes((self.sources/'r4_c3.png').read_bytes())
            with self.assertRaisesRegex(ValueError,'distinct'):
                build(self.sources,self.runtime,overlap=(16,12))
            self.assertEqual((self.runtime/'terrain.json').read_bytes(),descriptor)
            target.unlink()
            with self.assertRaises(FileNotFoundError):
                build(self.sources,self.runtime,overlap=(16,12))
            self.assertEqual((self.runtime/'terrain.json').read_bytes(),descriptor)
        finally:
            target.write_bytes(saved)

    def test_bad_alignment_is_rejected_before_publishing(self):
        descriptor=(self.runtime/'terrain.json').read_bytes()
        for offsets in ({'r2_c2':1},{'r0_c0':1},{'r0_c2':400}):
            with self.subTest(offsets=offsets),self.assertRaises(ValueError):
                build(self.sources,self.runtime,overlap=(16,12),horizontal_offsets=offsets)
        self.assertEqual((self.runtime/'terrain.json').read_bytes(),descriptor)


class AtlasMigrationTests(unittest.TestCase):
    def setUp(self):
        self.original=json.loads(BASE.read_text(encoding='utf8'))
        self.enlarged=enlarge(self.original)

    def test_migration_preserves_source_and_moves_all_routes_into_central_image(self):
        self.assertEqual(self.original,json.loads(BASE.read_text(encoding='utf8')))
        self.assertEqual(self.enlarged['world']['terrain'],'astra_atlas')
        for axis in ('width','height'):
            self.assertEqual(self.enlarged['world'][axis],20*self.original['world'][axis])
        for original,new in zip(self.original['units'],self.enlarged['units']):
            self.assertEqual(original['equipment'],new['equipment'])
            self.assertEqual(original['id'],new['id'])
            self.assertEqual((new['x'],new['y']),atlas.source_to_world(original['x'],original['y']))
            for a,b in zip(original['waypoints'],new['waypoints']):
                self.assertEqual((b['x'],b['y']),atlas.source_to_world(a['x'],a['y']))
        with self.assertRaisesRegex(ValueError,'twice'):
            enlarge(self.enlarged)

    def test_camera_focus_and_overview_preserve_simulation_facts(self):
        pygame.font.init()
        sim=Simulation(scene_from_data(self.enlarged))
        renderer=Renderer(pygame.Surface(WINDOW_SIZE))
        renderer.draw(sim)
        self.assertEqual(renderer.camera.max_zoom,256)
        self.assertAlmostEqual(renderer.camera.zoom,atlas.DETAIL_ZOOM)
        self.assertEqual(set(renderer.unit_hit_rects),{u.id for u in sim.units})
        before=deepcopy((sim.units,sim.events,sim.snapshots,sim.step_count))
        selected=sim.units[-1]
        renderer.selected_unit_id=selected.id
        renderer.camera.set_pose(1,Point(120000,80000))
        renderer.display_action('focus_scene',sim)
        self.assertAlmostEqual(renderer.camera.zoom,atlas.DETAIL_ZOOM)
        pixel=renderer.world_to_screen(selected.position,sim)
        self.assertAlmostEqual(pixel[0],renderer.map_rect.centerx,delta=1)
        self.assertAlmostEqual(pixel[1],renderer.map_rect.centery,delta=1)
        self.assertEqual(before,(sim.units,sim.events,sim.snapshots,sim.step_count))


class InstalledAtlasFidelityTests(unittest.TestCase):
    def setUp(self):
        if not atlas.available():
            self.skipTest('The full generated atlas has not been packaged yet')
        atlas.clear_cache()
        self.addCleanup(atlas.clear_cache)

    def test_source_pixel_area_and_preserved_central_core(self):
        data=atlas.metadata()
        self.assertGreaterEqual(data['native_imagery_area_scale'],20)
        self.assertEqual(data['raster_size'],[7168,4776])
        self.assertEqual(len(data['tiles']),192)
        self.assertEqual(len({s['sha256'] for s in data['source_tiles']}),25)
        original=ROOT/'sim_app/assets/terrain_astra/terrain_astra.png'
        self.assertEqual((atlas.ASSET_DIR.parent/'sources/r2_c2.png').read_bytes(),original.read_bytes())
        source=pygame.image.load(BytesIO(original.read_bytes()),'source.png')
        # Outside the 128/86px source overlaps, every central pixel is retained.
        x,y,w,h=128,86,1280,852
        world_view=((2816+x)*240000/7168,(1876+y)*160000/4776,
                    w*240000/7168,h*160000/4776)
        with patch.object(tiles,'_draw_labels'):
            visible=atlas.terrain_view_surface(atlas.WORLD_SIZE,(w,h),world_view)
        self.assertEqual(pygame.image.tobytes(visible,'RGB'),
                         pygame.image.tobytes(source.subsurface((x,y,w,h)),'RGB'))


class AtlasEditorNavigationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.root=create_root()
            cls.root.withdraw()
        except tk.TclError as error:
            raise unittest.SkipTest(f'Tk display unavailable: {error}')

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls,'root'):
            cls.root.destroy()

    def setUp(self):
        self.editor=SceneEditor(self.root,lambda path:None)
        self.editor.pack(fill='both',expand=True)
        self.addCleanup(self.editor.destroy)
        for method,value in (('winfo_width',800),('winfo_height',520)):
            p=patch.object(self.editor.canvas,method,return_value=value)
            p.start()
            self.addCleanup(p.stop)
        self.editor.load_document(enlarge(json.loads(BASE.read_text(encoding='utf8'))))

    def test_region_navigation_and_overview_leave_unapplied_fields_untouched(self):
        before=self.editor.get_document().data
        self.assertAlmostEqual(self.editor._camera_zoom,atlas.DETAIL_ZOOM)
        self.assertEqual(self.editor.region_navigation.winfo_manager(),'grid')
        self.assertEqual(self.editor.scene_focus_button.cget('text'),'定位装备')
        self.editor.select(('units',0))
        self.editor._item_vars['speed'].set('123.45')
        dirty_before_navigation=self.editor.dirty
        for region in (atlas.regions()[0],atlas.regions()[-1],atlas.regions()[12]):
            self.editor.atlas_region.set(region['name'])
            self.editor._atlas_region_changed(animate=False)
            actual=self.editor._world_xy(400,260)
            for a,b in zip(actual,region['center']):
                self.assertAlmostEqual(a,b,places=5)
        self.editor.atlas_region.set('全域总览')
        self.editor._atlas_region_changed(animate=False)
        self.assertEqual(self.editor._camera_zoom,1)
        self.assertEqual(self.editor._camera_pan,(0.0,0.0))
        self.assertEqual(self.editor._item_vars['speed'].get(),'123.45')
        self.assertEqual(self.editor.get_document().data,before)
        self.assertEqual(self.editor.dirty,dirty_before_navigation)

    def test_equipment_focus_from_waypoint_selection_returns_to_its_unit(self):
        self.editor.select(('waypoints',0,0))
        self.editor.atlas_region.set('东北群山')
        self.editor._atlas_region_changed(animate=False)
        self.assertTrue(self.editor.focus_scene(animate=False))
        unit=self.editor.get_document().data['units'][0]
        actual=self.editor._world_xy(400,260)
        self.assertAlmostEqual(actual[0],unit['x'],places=5)
        self.assertAlmostEqual(actual[1],unit['y'],places=5)
        self.assertAlmostEqual(self.editor._camera_zoom,atlas.DETAIL_ZOOM)

    def test_legacy_loading_restores_controls_zoom_limit_and_saved_scene_identity(self):
        self.editor.load_document(json.loads(BASE.read_text(encoding='utf8')))
        self.assertEqual(self.editor._max_virtual_zoom(),64)
        self.assertEqual(self.editor.region_navigation.winfo_manager(),'')
        self.assertEqual(self.editor.scene_focus_button.cget('text'),'定位场景')
        self.assertEqual(self.editor.get_document().data['world']['terrain'],'astra_mountain')


if __name__=='__main__':
    unittest.main()
