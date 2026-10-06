"""Photographic source fidelity, tiled continuity and legacy isolation."""
from io import BytesIO
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault('SDL_VIDEODRIVER','dummy')
os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT','1')
import pygame
from sim_app import detailed_terrain as atlas
from sim_app import photographic_terrain as terrain
from tools.build_photographic_terrain import package

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'sim_app/assets/terrain_astra/terrain_astra.png'


class PhotographicTerrainTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(terrain.clear_cache)
        terrain.clear_cache()
        self.labels=patch.object(atlas,'_draw_labels')
        self.labels.start()
        self.addCleanup(self.labels.stop)

    def source(self):
        return pygame.image.load(BytesIO(SOURCE.read_bytes()),'source.png')

    def test_original_artwork_pixels_are_preserved_and_flat_layers_are_not_drawn(self):
        data=terrain.metadata()
        self.assertEqual(data['render_mode'],'photographic')
        self.assertEqual(data['native_source_size'],list(self.source().get_size()))
        self.assertEqual(data['raster_size'],data['native_source_size'])
        with patch.object(atlas,'_draw_features',side_effect=AssertionError('photograph overpainted')):
            result=terrain.terrain_view_surface((12000,8000),self.source().get_size(),(0,0,12000,8000))
        self.assertEqual(pygame.image.tobytes(result,'RGB'),pygame.image.tobytes(self.source(),'RGB'))

    def test_native_pixel_pan_across_tile_boundary_uses_the_same_landscape(self):
        source=self.source()
        # Native-scale crops cross both horizontal and vertical 512px seams.
        factor=12000/source.get_width()
        a=terrain.terrain_view_surface((12000,8000),(600,400),(500*factor,300*factor,600*factor,400*factor))
        b=terrain.terrain_view_surface((12000,8000),(600,400),(540*factor,300*factor,600*factor,400*factor))
        self.assertEqual(pygame.image.tobytes(a,'RGB'),pygame.image.tobytes(source.subsurface((500,300,600,400)),'RGB'))
        self.assertEqual(pygame.image.tobytes(b,'RGB'),pygame.image.tobytes(source.subsurface((540,300,600,400)),'RGB'))
        self.assertEqual(pygame.image.tobytes(a.subsurface((40,0,560,400)),'RGB'),
                         pygame.image.tobytes(b.subsurface((0,0,560,400)),'RGB'))

    def test_photo_ppm_and_cache_remain_bounded_through_zoom_and_pan(self):
        for zoom in (1,2,8,64):
            for x,y in ((0,0),(4000,3000),(9000,5000)):
                terrain.terrain_view_ppm((12000,8000),(480,320),(x,y,12000/zoom,8000/zoom))
        cache=terrain.cache_info()
        self.assertLessEqual(cache['tiles'],terrain.TILE_CACHE_LIMIT)
        self.assertEqual(cache['pose_surfaces'],1)
        self.assertLessEqual(cache['ppm_bytes'],480*320*3+40)
        view=((12000,8000),(480,320),(3000,2000,6000,4000))
        image=terrain.terrain_view_surface(*view)
        self.assertEqual(terrain.terrain_view_ppm(*view),b'P6\n480 320\n255\n'+pygame.image.tobytes(image,'RGB'))

    def test_new_landmarks_are_from_astra_artwork_not_the_old_schematic(self):
        landmarks=terrain.metadata()['landmarks']
        expected=json.loads((SOURCE.parent/'landmarks.json').read_text(encoding='utf8'))
        records={item['id']:item for item in expected['landmarks']}
        self.assertEqual(landmarks['airport']['position'],records['hegu_airport']['world'])
        self.assertEqual(landmarks['central_bridge']['position'],records['central_road_bridge']['world'])
        self.assertNotEqual(landmarks['airport']['position'],[10200,6470])
        self.assertEqual(terrain.metadata()['features'],[])

    def test_packager_keeps_native_resolution_and_does_not_modify_input(self):
        before=SOURCE.read_bytes()
        with tempfile.TemporaryDirectory() as directory:
            result=package(SOURCE,directory,SOURCE.parent/'landmarks.json')
            self.assertEqual(result['native_source_size'],list(self.source().get_size()))
            self.assertEqual(result['levels'][0]['width'],self.source().get_width())
            self.assertEqual(result['levels'][0]['height'],self.source().get_height())
            self.assertTrue((Path(directory)/'terrain.json').is_file())
            self.assertEqual(len(result['tiles']),9)
        self.assertEqual(SOURCE.read_bytes(),before)

    def test_switching_between_historical_and_astra_scenes_does_not_reuse_pose_pixels(self):
        view=((12000,8000),(480,320),(0,0,12000,8000))
        new=pygame.image.tobytes(terrain.terrain_view_surface(*view),'RGB')
        old=pygame.image.tobytes(atlas.terrain_view_surface(*view),'RGB')
        self.assertNotEqual(old,new)
        self.assertNotEqual(atlas.metadata().get('render_mode'),'photographic')
        self.assertEqual(new,pygame.image.tobytes(terrain.terrain_view_surface(*view),'RGB'))
        self.assertEqual(terrain.metadata()['render_mode'],'photographic')
