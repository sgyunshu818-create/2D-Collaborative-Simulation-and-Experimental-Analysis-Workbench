"""The detailed terrain keeps visual geography fixed while its view changes."""
import os
import unittest
from unittest.mock import patch

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
import pygame

from sim_app import detailed_terrain as terrain


class DetailedTerrainTests(unittest.TestCase):
    def setUp(self):
        # These cases retain coverage of the previous editable native layers;
        # photographic artwork is covered independently in test_photographic_terrain.
        directory = patch.object(terrain, 'ASSET_DIR', terrain.LEGACY_ASSET_DIR)
        directory.start()
        self.addCleanup(directory.stop)
        self.addCleanup(terrain.clear_cache)
        terrain.clear_cache()

    def test_canonical_geography_and_proportional_world_match(self):
        # A view of the same streets in a proportionally larger simulation uses
        # exactly the same geography; objects must not drift against the map.
        a = terrain.terrain_view_surface((12000,8000),(600,400),(7680,2550,400,266))
        b = terrain.terrain_view_surface((24000,16000),(600,400),(15360,5100,800,532))
        self.assertEqual(pygame.image.tobytes(a,"RGB"), pygame.image.tobytes(b,"RGB"))
        self.assertEqual(terrain.metadata()["landmarks"]["airport"]["position"], [10200,6470])

    def test_pan_preserves_vector_building_positions(self):
        a = terrain.terrain_view_surface((12000,8000),(800,500),(7680,2550,400,250))
        b = terrain.terrain_view_surface((12000,8000),(800,500),(7780,2550,400,250))
        # Both views have 2 pixels per world unit. The known roof's full interior
        # remains at the same world point even though the viewport moves 100 units.
        buildings=[f for f in terrain.metadata()["features"] if f["type"]=="building"]
        f=next(f for f in buildings if all(7800<p[0]<8000 and 2620<p[1]<2730 for p in f["polygon"]))
        x=sum(p[0] for p in f["polygon"])/4
        y=sum(p[1] for p in f["polygon"])/4
        px,py=round((x-7680)*2),round((y-2550)*2)
        self.assertEqual(a.get_at((px,py)),b.get_at((px-200,py)))
        self.assertIn(tuple(a.get_at((px,py)))[:3],
                      [tuple(f["color"]),tuple(min(255,v+22) for v in f["color"])])

    def test_native_detail_is_more_than_resizing_the_overview(self):
        overview=terrain.terrain_view_surface((12000,8000),(1200,800),(0,0,12000,8000))
        close=terrain.terrain_view_surface((12000,8000),(600,400),(7780,2600,187.5,125))
        coarse=overview.subsurface((778,260,19,13))
        enlarged=pygame.transform.smoothscale(coarse,close.get_size())
        # Roof geometry and lane markings create detail not present in the mip.
        self.assertNotEqual(pygame.image.tobytes(close,"RGB"),pygame.image.tobytes(enlarged,"RGB"))
        colors=set(tuple(close.get_at((x,y)))[:3] for x in range(0,600,3) for y in range(0,400,3))
        self.assertTrue(any(tuple(f["color"]) in colors for f in terrain.metadata()["features"]
                            if f["type"]=="building"))

    def test_world_edge_crop_and_outside_area(self):
        image=terrain.terrain_view_surface((12000,8000),(400,300),(-100,-75,200,150))
        self.assertEqual(tuple(image.get_at((20,20)))[:3],terrain.BACKGROUND)
        self.assertEqual(tuple(image.get_at((199,149)))[:3],terrain.BACKGROUND)
        self.assertNotEqual(tuple(image.get_at((350,250)))[:3],terrain.BACKGROUND)
        outside=terrain.terrain_view_surface((12000,8000),(130,90),(13000,9000,100,100))
        self.assertEqual(pygame.image.tobytes(outside,"RGB"),bytes(terrain.BACKGROUND)*(130*90))

    def test_streaming_cache_is_bounded_and_zoom_does_not_scale_whole_tiles(self):
        real_scale=pygame.transform.smoothscale
        allocations=[]
        def observed(source,size,*args):
            allocations.append(size)
            return real_scale(source,size,*args)
        with patch.object(pygame.transform,"smoothscale",side_effect=observed):
            for row in range(6):
                for col in range(10):
                    terrain.terrain_view_surface((12000,8000),(480,320),(col*1100+30,row*1200+30,187.5,125))
        cache=terrain.cache_info()
        self.assertLessEqual(cache["tiles"],48)
        self.assertEqual(cache["pose_surfaces"],1)
        self.assertLessEqual(cache["tile_bytes"],48*512*512*4)
        self.assertTrue(allocations)
        self.assertLessEqual(max(w for w,h in allocations),520)
        self.assertLessEqual(max(h for w,h in allocations),360)

    def test_ppm_is_same_pixels_and_reuses_one_view(self):
        args=((12000,8000),(240,160),(9600,6000,1200,800))
        image=terrain.terrain_view_surface(*args)
        ppm=terrain.terrain_view_ppm(*args)
        self.assertEqual(ppm,b"P6\n240 160\n255\n"+pygame.image.tobytes(image,"RGB"))
        self.assertIs(image,terrain.terrain_view_surface(*args))
        self.assertIs(ppm,terrain.terrain_view_ppm(*args))


if __name__ == "__main__":
    unittest.main()
