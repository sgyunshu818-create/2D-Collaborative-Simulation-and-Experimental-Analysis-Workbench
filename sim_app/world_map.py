"""Offline geographic atlas shared by the native editor and runtime renderer.

Natural Earth pixels are equirectangular. The camera uses degrees, while the
simulation uses its independent local coordinates via an explicit reference.
"""
from collections import OrderedDict
from functools import lru_cache
from io import BytesIO
from math import ceil, floor, log2
from pathlib import Path

from .camera import MapCamera
from .font_support import pygame_font
from .geography import GeoReference, distance_m, format_lonlat
from .models import Point
from .visual_assets import unit_glyph_surface

DATA = Path(__file__).resolve().parent / 'assets/world'
BACKGROUND = (8, 24, 37)
ACCENT = (45, 205, 232)
TEXT = (217, 234, 241)
MUTED = (139, 171, 188)
TEAM = {'red':(239, 110, 130), 'blue':(87, 200, 246)}


@lru_cache(maxsize=48)
def _tile(level, x, y):
    import pygame
    path = DATA / f'{level}_{x}_{y}.jpg'
    return pygame.image.load(BytesIO(path.read_bytes()), 'tile.jpg') if path.is_file() else None


def _font(size, bold=False):
    import pygame
    if not pygame.font.get_init():
        pygame.font.init()
    return pygame_font(size, bold=bold)


class WorldMapView:
    def __init__(self):
        self.camera = MapCamera(max_zoom=256)
        self.size = (640, 420)
        self.camera.bind((360, 180), (0, 0, *self.size))
        self.hit_rects = {}
        self.scene_rect = None
        self._background_key = None
        self._background = None
        self._last_surface = None
        self._text_cache = OrderedDict()

    def resize(self, size):
        self.size = tuple(max(1, int(v)) for v in size)
        self.camera.bind((360, 180), (0, 0, *self.size))

    def update(self):
        return self.camera.update()

    def to_screen(self, lon, lat):
        return self.camera.to_screen(Point(lon+180, 90-lat))

    def to_lonlat(self, screen):
        point = self.camera.to_world(screen)
        return point.x-180, 90-point.y

    def local_to_screen(self, x, y, world):
        ref = GeoReference.from_world(world)
        return self.to_screen(*ref.to_lonlat(x, y, world['width'], world['height'])) if ref else None

    def screen_to_local(self, screen, world):
        ref = GeoReference.from_world(world)
        return ref.to_local(*self.to_lonlat(screen), world['width'], world['height']) if ref else None

    def zoom_by(self, factor, anchor=None):
        return self.camera.zoom_by(factor, anchor)

    def pan(self, delta):
        self.camera.pan(delta)

    def focus_global(self, animate=True):
        self.camera.reset(animate=animate)

    def focus_scene(self, world, animate=False):
        ref = GeoReference.from_world(world)
        if ref is None:
            return False
        west, south, east, north = ref.bounds(world['width'], world['height'])
        zoom = min(self.camera.max_zoom, max(1, min(self.size[0]*.75/(east-west),
                         self.size[1]*.72/(north-south)) / self.camera.base_scale))
        # Scene focus is a deliberate camera jump; wheel changes remain eased.
        self.camera.set_pose(zoom, Point(ref.center_longitude+180, 90-ref.center_latitude))
        return True

    def pick_unit(self, pixel):
        return next((key for key, rect in reversed(list(self.hit_rects.items())) if rect.collidepoint(pixel)), None)

    def _text_bitmap(self, text, size=12, color=TEXT, bold=False):
        # Cache rendered pixels, never Font handles. SDL_ttf shutdown invalidates
        # Font objects even if Python references survive a later font.init().
        key = (str(text), size, color, bold)
        rendered = self._text_cache.get(key)
        if rendered is None:
            rendered = _font(size,bold).render(str(text),True,color)
            if len(self._text_cache) >= 192:
                self._text_cache.popitem(last=False)
            self._text_cache[key] = rendered
        else:
            self._text_cache.move_to_end(key)
        return rendered

    def _text(self, surface, text, pos, size=12, color=TEXT, *, anchor='topleft', plate=False, bold=False):
        import pygame
        rendered = self._text_bitmap(text, size, color, bold)
        rect = rendered.get_rect(**{anchor:tuple(round(v) for v in pos)})
        if plate:
            pygame.draw.rect(surface, (10,32,48), rect.inflate(10,6), border_radius=3)
        surface.blit(rendered, rect)
        return rect

    def _base(self, terrain, grid):
        import pygame
        scale = self.camera.base_scale * self.camera.zoom
        key = (self.size, self.camera.zoom, self.camera.actual_center, terrain, grid)
        if key == self._background_key:
            return self._background
        surface = pygame.Surface(self.size)
        surface.fill(BACKGROUND)
        rect = surface.get_rect()
        if terrain:
            level = max(0,min(3,ceil(log2(max(1,360*scale/1024)))))
            nx, ny = 2*2**level, 2**level
            view = self.camera.visible_world()
            tx0, tx1 = max(0, floor(view[0]/360*nx)), min(nx, ceil((view[0]+view[2])/360*nx))
            ty0, ty1 = max(0, floor(view[1]/180*ny)), min(ny, ceil((view[1]+view[3])/180*ny))
            for tx in range(tx0,tx1):
                for ty in range(ty0,ty1):
                    tile = _tile(level,tx,ty)
                    if tile is None:
                        continue
                    x0,y0 = self.camera.to_screen(Point(tx*360/nx,ty*180/ny))
                    x1,y1 = self.camera.to_screen(Point((tx+1)*360/nx,(ty+1)*180/ny))
                    dest = pygame.Rect(round(x0),round(y0),round(x1)-round(x0),round(y1)-round(y0))
                    visible = dest.clip(rect)
                    if visible.width < 1 or visible.height < 1:
                        continue
                    # Crop before resampling: a close-up never allocates a huge
                    # offscreen tile. Only viewport pixels are produced.
                    src = pygame.Rect(max(0,floor((visible.x-x0)/(x1-x0)*512)-1),
                                      max(0,floor((visible.y-y0)/(y1-y0)*512)-1),0,0)
                    src.width = max(1,ceil((visible.right-x0)/(x1-x0)*512)+1-src.x)
                    src.height = max(1,ceil((visible.bottom-y0)/(y1-y0)*512)+1-src.y)
                    src = src.clip(tile.get_rect())
                    # Keep the cropped source edges in their original screen
                    # positions. Stretching this crop to the viewport would
                    # snap geographic texture whenever its integer crop moves.
                    dx0,dy0=round(x0+src.x/512*(x1-x0)),round(y0+src.y/512*(y1-y0))
                    dx1,dy1=round(x0+src.right/512*(x1-x0)),round(y0+src.bottom/512*(y1-y0))
                    patch=pygame.transform.smoothscale(tile.subsurface(src),(max(1,dx1-dx0),max(1,dy1-dy0)))
                    surface.blit(patch,(dx0,dy0))
            wash = pygame.Surface(self.size,pygame.SRCALPHA)
            wash.fill((*BACKGROUND,75))
            surface.blit(wash,(0,0))
        west,north = self.to_lonlat((0,0))
        east,south = self.to_lonlat(self.size)
        if grid:
            step = next((v for v in (.1,.25,.5,1,2,5,10,15,30,60) if v*scale >= 100),60)
            for lon in _ticks(max(-180,west),min(180,east),step):
                x,_ = self.to_screen(lon,0)
                pygame.draw.line(surface,(75,108,118),(round(x),0),(round(x),self.size[1]),1)
                self._text(surface,f'{abs(lon):g}°{"E" if lon>=0 else "W"}',(x+5,37),11,MUTED,plate=True)
            for lat in _ticks(max(-90,south),min(90,north),step):
                _,y = self.to_screen(0,lat)
                pygame.draw.line(surface,(75,108,118),(0,round(y)),(self.size[0],round(y)),1)
                if 65<y<self.size[1]-65:
                    self._text(surface,f'{abs(lat):g}°{"N" if lat>=0 else "S"}',(7,y-14),11,MUTED,plate=True)
        if self.camera.zoom < 6:
            for name,lon,lat in (('北美洲',-105,44),('南美洲',-58,-18),('欧洲',15,51),
                                 ('非洲',20,7),('亚洲',88,42),('大洋洲',136,-26),('南极洲',15,-78),
                                 ('太平洋',-145,0),('大西洋',-32,12),('印度洋',75,-25)):
                p=self.to_screen(lon,lat)
                if rect.collidepoint(p):
                    self._text(surface,name,p,14,TEXT,anchor='center',plate=True)
        self._background_key,self._background=key,surface
        return surface

    def render(self, world_dict, units_rows=(), selected_id=None, show_grid=True,
               show_terrain=True, show_trails=True, show_labels=True, title=''):
        import pygame
        self.update()
        surface = self._base(show_terrain,show_grid).copy()
        viewport = surface.get_rect()
        self.hit_rects.clear()
        self.scene_rect=None
        try:
            ref=GeoReference.from_world(world_dict)
            if ref:
                ref.validate_extent(world_dict['width'],world_dict['height'])
        except (ValueError,TypeError,KeyError):
            ref=None
        rows=list(units_rows)
        if ref:
            west,south,east,north=ref.bounds(world_dict['width'],world_dict['height'])
            tl=self.to_screen(west,north); br=self.to_screen(east,south)
            scene_rect=pygame.Rect(round(tl[0]),round(tl[1]),max(2,round(br[0]-tl[0])),max(2,round(br[1]-tl[1])))
            self.scene_rect=scene_rect
            clipped=scene_rect.clip(viewport)
            if clipped.width and clipped.height:
                pygame.draw.rect(surface,ACCENT,scene_rect,1)
                if scene_rect.width < 130 or scene_rect.height < 85:
                    p=self.to_screen(ref.center_longitude,ref.center_latitude)
                    if viewport.collidepoint(p):
                        pygame.draw.circle(surface,ACCENT,(round(p[0]),round(p[1])),7,2)
                        self._text(surface,f'{title or "当前场景"} · {len(rows)} 单位',(p[0]+12,p[1]),12,TEXT,plate=True)
                else:
                    self._draw_sites(surface, world_dict)
                    self._draw_units(surface,world_dict,rows,selected_id,show_trails,show_labels)
        badge='全球地理总览' if self.camera.zoom<3 else '区域地理视图'
        self._text(surface,badge,(12,10),14,ACCENT,plate=True,bold=True)
        if ref is None:
            self._text(surface,'当前场景未设置地理参考 · 可浏览全球',(12,self.size[1]-52),12,TEXT,plate=True)
        self._text(surface,'N',(self.size[0]-25,17),14,TEXT,anchor='center',bold=True)
        pygame.draw.polygon(surface,ACCENT,[(self.size[0]-25,31),(self.size[0]-31,47),(self.size[0]-19,47)])
        self._scale_bar(surface)
        footer='Natural Earth · 离线地形' + (' · 小比例尺底图' if self.camera.zoom>12 else '')
        self._text(surface,footer,(12,self.size[1]-18),11,MUTED,plate=True)
        self._last_surface=surface
        return surface

    def _draw_units(self,surface,world,rows,selected,trails,labels):
        import pygame
        occupied=[]
        for row in rows:
            color=TEAM.get(str(getattr(row.get('team'),'value',row.get('team'))),TEAM['blue'])
            p=self.local_to_screen(row['x'],row['y'],world)
            path=[self.local_to_screen(q['x'],q['y'],world) for q in row.get('path',())]
            if path:
                pygame.draw.lines(surface,(72,115,132),False,[p,*path],1)
            if trails:
                points=[self.local_to_screen(q['x'],q['y'],world) for q in row.get('trail',())]
                if len(points)>1:
                    pygame.draw.lines(surface,color,False,points,2 if row['id']==selected else 1)
            waypoints=[self.local_to_screen(q['x'],q['y'],world) for q in row.get('waypoints',())]
            if waypoints:
                pygame.draw.lines(surface,color,False,[p,*waypoints],1)
                for q in waypoints:
                    if surface.get_rect().collidepoint(q):
                        pygame.draw.circle(surface,color,tuple(round(v) for v in q),4,1)
            if not surface.get_rect().inflate(50,50).collidepoint(p):
                continue
            icon=unit_glyph_surface(row,color,48,row['id']==selected)
            rect=icon.get_rect(center=tuple(round(v) for v in p))
            surface.blit(icon,rect)
            self.hit_rects[row['id']]=rect.inflate(6,6)
            if row['id']==selected:
                radius=row.get('sensor_range',0)
                if radius>0:
                    ex=self.local_to_screen(row['x']+radius,row['y'],world)
                    ey=self.local_to_screen(row['x'],row['y']+radius,world)
                    rx=min(2000,max(1,round(abs(ex[0]-p[0]))))
                    ry=min(2000,max(1,round(abs(ey[1]-p[1]))))
                    clip=surface.get_clip()
                    surface.set_clip(self.scene_rect.clip(surface.get_rect()))
                    pygame.draw.ellipse(surface,(45,115,136),(round(p[0])-rx,round(p[1])-ry,2*rx,2*ry),1)
                    surface.set_clip(clip)
            if labels:
                rendered=self._text_bitmap(row['id'],12,color)
                rect=rendered.get_rect(topleft=(round(p[0]+27),round(p[1]-12)))
                rect.clamp_ip(surface.get_rect().inflate(-12,-60))
                for offset in (0,24,-24,48,-48):
                    candidate=rect.move(0,offset)
                    if surface.get_rect().contains(candidate) and not any(candidate.colliderect(o) for o in occupied):
                        rect=candidate; break
                pygame.draw.rect(surface,(10,32,48),rect.inflate(8,4),border_radius=3)
                surface.blit(rendered,rect)
                occupied.append(rect.inflate(4,4))
                self.hit_rects[row['id']]=self.hit_rects[row['id']].union(rect)

    def _draw_sites(self, surface, world):
        import pygame
        for item in world.get('obstacles', ()):
            start=self.local_to_screen(item['x'],item['y'],world)
            end=self.local_to_screen(item['x']+item['width'],item['y']+item['height'],world)
            rect=pygame.Rect(round(start[0]),round(start[1]),max(1,round(end[0]-start[0])),max(1,round(end[1]-start[1])))
            pygame.draw.rect(surface,(32,58,73),rect)
            pygame.draw.rect(surface,(99,142,156),rect,1)
        for item in world.get('sites', ()):
            p=self.local_to_screen(item['x'],item['y'],world)
            if surface.get_rect().collidepoint(p):
                x,y=(round(v) for v in p)
                color=TEAM.get(item.get('team'),ACCENT)
                pygame.draw.polygon(surface,color,[(x,y-6),(x+6,y),(x,y+6),(x-6,y)],1)
                self._text(surface,item.get('label','参考点'),(x,y-26),11,color,anchor='center',plate=True)

    def _scale_bar(self,surface):
        import pygame
        x,y=self.size[0]-160,self.size[1]-30
        lon,lat=self.to_lonlat((self.size[0]/2,self.size[1]/2))
        if not -89.5 < lat < 89.5:
            self._text(surface,'极区 · 比例尺不可用',(x,y-23),11,MUTED,plate=True)
            return
        degrees=120/(self.camera.base_scale*self.camera.zoom)
        meters=distance_m(lon,lat,lon+degrees,lat)
        if meters<=0:
            return
        power=10**floor(log2(meters)/log2(10))
        distance=max(v*power for v in (1,2,5) if v*power<=meters)
        low,high=0,120
        for _ in range(20):
            mid=(low+high)/2
            span=mid/(self.camera.base_scale*self.camera.zoom)
            if distance_m(lon,lat,lon+span,lat)<distance:
                low=mid
            else:
                high=mid
        pixels=max(1,round((low+high)/2))
        label=(f'{distance/1000:g} km' if distance>=1000 else f'{distance:g} m')+' · 中心纬度'
        self._text(surface,label,(x,y-23),12,TEXT,plate=True)
        pygame.draw.line(surface,TEXT,(x,y),(x+pixels,y),2)
        pygame.draw.line(surface,TEXT,(x,y-4),(x,y+4),1)
        pygame.draw.line(surface,TEXT,(x+pixels,y-4),(x+pixels,y+4),1)

    def ppm(self):
        import pygame
        if self._last_surface is None:
            return None
        w,h=self._last_surface.get_size()
        return f'P6\n{w} {h}\n255\n'.encode('ascii')+pygame.image.tobytes(self._last_surface,'RGB')


def _ticks(low,high,step):
    start=ceil(low/step)*step
    return [start+i*step for i in range(max(0,floor((high-start)/step)+1))]
