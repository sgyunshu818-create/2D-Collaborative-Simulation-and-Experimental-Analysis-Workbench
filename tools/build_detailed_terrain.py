"""Author the fictional valley's relief and deterministic vector geography.

Run with the bundled development Python (NumPy and Pillow). These dependencies
are never required by the installed application. Assets are saved into 512 px
tiles so the application's maps have bounded memory at every zoom level.
"""
from collections import Counter
from pathlib import Path
import json
import math
import random
import sys

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "sim_app" / "assets" / "detailed"
ARTIFACT = ROOT / "artifacts" / "detailed_map_20261006"
WORLD = (12000, 8000)
RASTER = (6144, 4096)
SEED = 261006
SCALE = RASTER[0] / WORLD[0]


def noise(rng, dims, size=RASTER):
    raw = rng.random(dims, dtype=np.float32)
    im = Image.fromarray(raw, "F").resize(size, Image.Resampling.BICUBIC)
    return np.asarray(im, dtype=np.float32)


def perlin(rng, rows, cols):
    """Gradient noise produces continuous spurs instead of isolated round humps."""
    width,height=1536,1024
    xx=np.arange(width,dtype=np.float32)*cols/width
    yy=np.arange(height,dtype=np.float32)*rows/height
    ix,iy=xx.astype(int),yy.astype(int)
    fx=(xx-ix)[None,:].astype(np.float32);fy=(yy-iy)[:,None].astype(np.float32)
    angles=rng.random((rows+1,cols+1),dtype=np.float32)*(2*np.pi)
    gx,gy=np.cos(angles),np.sin(angles)
    def corner(dx,dy):
        return gx[iy[:,None]+dy,ix[None,:]+dx]*(fx-dx)+gy[iy[:,None]+dy,ix[None,:]+dx]*(fy-dy)
    u=fx**3*(fx*(fx*6-15)+10);v=fy**3*(fy*(fy*6-15)+10)
    a=corner(0,0)*(1-u)+corner(1,0)*u
    b=corner(0,1)*(1-u)+corner(1,1)*u
    raw=(a*(1-v)+b*v)*1.55
    return np.asarray(Image.fromarray(raw.astype(np.float32),"F").resize(RASTER,Image.Resampling.BICUBIC),dtype=np.float32)


def smooth(points, divisions=24):
    result = []
    for i in range(len(points)-1):
        a, b, c, d = points[max(0,i-1)], points[i], points[i+1], points[min(len(points)-1,i+2)]
        for j in range(divisions):
            t = j/divisions
            result.append([round(.5*((2*b[k])+(-a[k]+c[k])*t+(2*a[k]-5*b[k]+4*c[k]-d[k])*t*t+
                                    (-a[k]+3*b[k]-3*c[k]+d[k])*t*t*t),2) for k in (0,1)])
    return result + [list(points[-1])]


def rotated_rect(cx,cy,w,h,angle=0):
    co, si = math.cos(angle), math.sin(angle)
    return [[round(cx+x*co-y*si,2), round(cy+x*si+y*co,2)]
            for x,y in ((-w/2,-h/2),(w/2,-h/2),(w/2,h/2),(-w/2,h/2))]


def dist_to_path(x,y,points):
    answer = 1e9
    for a,b in zip(points,points[1:]):
        vx,vy = b[0]-a[0],b[1]-a[1]
        t = max(0,min(1,((x-a[0])*vx+(y-a[1])*vy)/(vx*vx+vy*vy or 1)))
        answer = min(answer,math.hypot(x-a[0]-t*vx,y-a[1]-t*vy))
    return answer


def geography(height):
    rng = random.Random(SEED)
    features = []
    rivers = [
        ([(6300,-100),(6560,600),(6200,1180),(6840,1990),(6560,2770),(5740,3510),
          (5890,4110),(5320,4840),(5500,5480),(4760,6350),(4250,7040),(4020,8100)],135),
        ([(-100,1950),(1250,2300),(2430,2150),(3580,2550),(4360,3110),(5740,3510)],56),
        ([(10800,-100),(10100,560),(9890,1380),(8890,2110),(7820,2520),(6560,2770)],46),
        ([(11900,7340),(10600,6850),(9580,7090),(8130,6830),(6740,6330),(4760,6350)],63),
        ([(840,7990),(1510,7270),(2030,6870),(2900,6610),(3680,6930),(4250,7040)],37),
    ]
    for points,width in rivers:
        features.append({"type":"river","points":smooth(points),"width":width})
    lake = [[6220,1160],[6420,1080],[6690,1110],[6920,1030],[7260,1100],[7490,1220],
            [7680,1430],[7590,1620],[7390,1720],[7310,1890],[7020,1950],[6760,1840],
            [6570,1730],[6490,1530],[6350,1410]]
    features.append({"type":"lake","polygon":smooth(lake+[lake[0]],4)[:-1]})
    small_lake = [[3200,900],[3310,790],[3510,760],[3670,850],[3730,1010],[3600,1140],[3390,1140],[3260,1060]]
    features.append({"type":"lake","polygon":smooth(small_lake+[small_lake[0]],4)[:-1]})
    features.append({"type":"apron","polygon":rotated_rect(9850,6110,720,235,-.14),
                     "color":[92,102,94]})
    fields=[]
    for cx,cy,nx,ny,angle in [(7950,4440,8,7,-.11),(2880,5960,6,7,.12),(9700,7420,6,3,-.04)]:
        for row in range(ny):
            for col in range(nx):
                x=cx+(col-nx/2)*174+rng.uniform(-12,12)
                y=cy+(row-ny/2)*141+rng.uniform(-10,10)
                if min(dist_to_path(x,y,r[0]) for r in rivers) < 200:
                    continue
                tone=rng.choice([(100,108,65),(112,113,73),(128,116,74),(87,104,65),(124,125,80),(87,111,77)])
                f={"type":"field","polygon":rotated_rect(x,y,158,126,angle+rng.uniform(-.035,.035)),
                   "color":tone,"row_color":[max(0,c-15) for c in tone],"rows":rng.randint(12,22)}
                features.append(f);fields.append(f)
    roads = [
        ([(760,4490),(2100,4560),(3160,4250),(4290,4450),(4930,4040),(5710,3540),
          (7040,3230),(7660,2740),(8510,2730),(9980,2910),(11370,3610)],30),
        ([(3070,0),(3080,1140),(3580,1740),(4030,3000),(4290,4450),(3860,5190),(4110,6030),(3630,8000)],24),
        ([(8520,0),(8450,980),(8240,1780),(8510,2730),(8790,3520),(8720,4630),
          (9190,5200),(9830,5900),(10960,5980),(11740,6800)],28),
        ([(230,7050),(1930,6450),(3370,5930),(4230,6060),(4830,6260),(6420,6540),
          (8000,6180),(9190,5200)],22),
        ([(7040,3230),(7350,3670),(7830,3960),(8720,4630),(10300,4530),(11900,4820)],22),
        ([(7040,3230),(6900,2500),(7400,2060),(8240,1780)],13),
        ([(3160,4250),(2940,3370),(2430,2800),(1820,2450)],11),
    ]
    road_features=[]
    for p,w in roads:
        f={"type":"road","points":smooth(p),"width":w}
        features.append(f);road_features.append(f)
    # Precisely located bridges cross the actual curved river, with approaches.
    for road_index,knot_index in [(0,5),(3,4)]:
        road=road_features[road_index]
        center=knot_index*24
        features.append({"type":"bridge","points":road["points"][center-4:center+5],"width":30})
    # Locate the actual western tributary crossing, rather than guessing a deck.
    western_river=next(f["points"] for f in features if f["type"]=="river" and f["width"]==37)
    road=road_features[1]["points"]
    center=min(range(len(road)),key=lambda i:dist_to_path(*road[i],western_river))
    features.append({"type":"bridge","points":road[max(0,center-3):center+4],"width":25})
    building_features=[]
    for cx,cy,nx,ny,step,angle in [(7840,2720,10,8,65,.03),(4240,4470,9,8,61,-.11),
                                (8860,4650,8,7,63,-.04),(10070,2900,7,7,64,.12)]:
        # Narrow internal streets and generous blocks remain visible at close zoom.
        for row in range(-ny//2,ny//2+1,2):
            p=rotated_rect(cx,cy+row*step,nx*step+60,1,angle)
            features.append({"type":"road","points":[p[0],p[1]],"width":10})
        for col in range(-nx//2,nx//2+1,2):
            p=rotated_rect(cx+col*step,cy,1,ny*step+60,angle)
            features.append({"type":"road","points":[p[0],p[3]],"width":10})
        for row in range(ny):
            for col in range(nx):
                x=cx+(col-(nx-1)/2)*step+rng.uniform(-4,4)
                y=cy+(row-(ny-1)/2)*step+rng.uniform(-4,4)
                if dist_to_path(x,y,roads[0][0])<35 or min(dist_to_path(x,y,p) for p,w in rivers)<w/2+28 or rng.random()<.12:
                    continue
                tone=rng.choice([(126,122,101),(117,104,88),(139,124,102),(114,127,120),(147,137,112)])
                f={"type":"building","kind":"house","polygon":rotated_rect(x,y,rng.uniform(24,43),rng.uniform(20,34),angle),"color":tone}
                features.append(f);building_features.append(f)
    # Depots, warehouses, fenced apron-like footprints, and a long joint runway.
    for cx,cy,nx,ny in [(8980,3480,4,3),(9040,4910,3,2),(9840,5960,4,2)]:
        for row in range(ny):
            for col in range(nx):
                f={"type":"building","kind":"warehouse","polygon":rotated_rect(cx+(col-nx/2)*125,cy+(row-ny/2)*100,96,63,-.08),
                   "color":rng.choice([(137,151,145),(147,154,148),(115,135,129)])}
                features.append(f);building_features.append(f)
    runway=rotated_rect(10200,6470,1660,83,-.14)
    centerline=[[sum(p[k] for p in (runway[0],runway[3]))/2 for k in (0,1)],
                [sum(p[k] for p in (runway[1],runway[2]))/2 for k in (0,1)]]
    features.append({"type":"runway","polygon":runway,"centerline":centerline})
    features.append({"type":"road","points":[[9410,6250],[9780,6180],[10620,6060]],"width":38})
    features.append({"type":"road","points":[[10070,6135],[10105,6260],[10185,6430]],"width":22})
    # Dense forest canopies are a shared fixed geometry in overview and native LOD.
    trees=[]
    forest_zones=[(1650,3600,1550,1750),(3900,1450,1550,1000),(9800,1150,2200,1300),
                  (10700,4480,1200,1300),(6200,6990,1450,1050),(1340,7350,1200,980),
                  (6750,4900,820,1000)]
    for _ in range(43000):
        x,y=rng.uniform(80,11920),rng.uniform(80,7920)
        density=max(math.exp(-2.2*(((x-cx)/rx)**2+((y-cy)/ry)**2)) for cx,cy,rx,ry in forest_zones)
        rough=float(height[min(RASTER[1]-1,int(y*SCALE)),min(RASTER[0]-1,int(x*SCALE))])
        if rng.random()>density*.78 or rough>.77:
            continue
        if min(dist_to_path(x,y,p) for p,w in rivers)<w/2+35:
            continue
        if min(dist_to_path(x,y,p) for p,w in roads)<w/2+24:
            continue
        if 6100<x<7800 and 1000<y<2000:
            continue
        # Keep the runway, taxiway, and apron clear, including their shoulders.
        airport_dx=(x-10200)*math.cos(-.14)+(y-6470)*math.sin(-.14)
        airport_dy=-(x-10200)*math.sin(-.14)+(y-6470)*math.cos(-.14)
        if abs(airport_dx)<950 and abs(airport_dy)<155:
            continue
        if 9460<x<10260 and 5910<y<6320:
            continue
        if 3300<x<3730 and 760<y<1150:
            continue
        if any(min(p[0] for p in f["polygon"])-15<x<max(p[0] for p in f["polygon"])+15 and
               min(p[1] for p in f["polygon"])-15<y<max(p[1] for p in f["polygon"])+15 for f in fields):
            continue
        if any(abs(x-cx)<420 and abs(y-cy)<350 for cx,cy in [(7840,2720),(4240,4470),(8860,4650),(10070,2900)]):
            continue
        f={"type":"tree","x":round(x,2),"y":round(y,2),"r":round(rng.uniform(9,19),2),
           "kind":"pine" if rough>.45 or rng.random()<.32 else "broadleaf",
           "color":rng.choice([(48,75,49),(54,83,55),(63,90,55),(57,84,58),(67,89,59),(61,79,49)])}
        trees.append(f)
    features.extend(trees)
    for _ in range(9000):
        x,y=rng.uniform(40,11960),rng.uniform(40,7960)
        rough=float(height[int(y*SCALE),int(x*SCALE)])
        if rough<.62 or rng.random()>.65:
            continue
        r=rng.uniform(5,18)
        features.append({"type":"rock","polygon":[[round(x-r,2),round(y,2)],[round(x-r*.3,2),round(y-r*.8,2)],
                         [round(x+r,2),round(y-r*.3,2)],[round(x+r*.5,2),round(y+r*.6,2)]],
                         "color":rng.choice([(116,121,106),(130,135,119),(139,141,124),(104,114,103)])})
    return features


def draw_geography(image,features):
    draw=ImageDraw.Draw(image)
    def pts(f):
        return [(round(p[0]*SCALE),round(p[1]*SCALE)) for p in f.get("points",f.get("polygon",[]))]
    def line(points,color,width):
        w=max(1,round(width*SCALE));draw.line(points,fill=color,width=w,joint="curve")
        for p in points:
            if w>3:draw.ellipse((p[0]-w/2,p[1]-w/2,p[0]+w/2,p[1]+w/2),fill=color)
    for f in features:
        kind=f["type"];p=pts(f)
        if kind=="river":
            line(p,(48,68,61),f["width"]+28);line(p,(58,105,111),f["width"]);line(p,(74,126,131),f["width"]*.57)
        elif kind=="apron":
            draw.polygon(p,fill=tuple(f["color"]));draw.line(p+[p[0]],fill=(133,141,124),width=2)
        elif kind=="lake":
            draw.polygon(p,fill=(49,89,97));draw.line(p+[p[0]],fill=(99,129,114),width=3,joint="curve")
            # Underwater shallows in the irregular littoral, rather than flat blue.
            cx=sum(x for x,y in p)/len(p);cy=sum(y for x,y in p)/len(p)
            inner=[(round(cx+(x-cx)*.9),round(cy+(y-cy)*.86)) for x,y in p]
            draw.polygon(inner,fill=(46,81,90))
        elif kind=="field":
            draw.polygon(p,fill=tuple(f["color"]));draw.line(p+[p[0]],fill=(65,75,52),width=2)
            a,b,c,d=p
            for t in range(1,f["rows"]):
                u=t/f["rows"]
                draw.line((a[0]+(d[0]-a[0])*u,a[1]+(d[1]-a[1])*u,b[0]+(c[0]-b[0])*u,b[1]+(c[1]-b[1])*u),fill=tuple(f["row_color"]),width=1)
        elif kind in ("road","bridge"):
            line(p,(45,53,46),f["width"]+9);line(p,(114,111,93),f["width"]);line(p,(147,141,116),max(2,f["width"]-6))
        elif kind=="building":
            draw.polygon([(x+3,y+3) for x,y in p],fill=(28,40,35));draw.polygon(p,fill=tuple(f["color"]));draw.line(p+[p[0]],fill=(72,81,72),width=1)
            a,b,c,d=p;draw.line(((a[0]+d[0])/2,(a[1]+d[1])/2,(b[0]+c[0])/2,(b[1]+c[1])/2),fill=tuple(min(255,c+20) for c in f["color"]),width=1)
        elif kind=="runway":
            draw.polygon(p,fill=(38,46,47));draw.line(p+[p[0]],fill=(138,148,135),width=2)
            a,b=f["centerline"];length=math.dist(a,b)
            for t in np.arange(0,length,75):
                u=t/length;v=min(1,(t+35)/length)
                draw.line(((a[0]+(b[0]-a[0])*u)*SCALE,(a[1]+(b[1]-a[1])*u)*SCALE,
                           (a[0]+(b[0]-a[0])*v)*SCALE,(a[1]+(b[1]-a[1])*v)*SCALE),fill=(219,217,185),width=2)
            a,b,c,d=f["polygon"]
            for u in (.035,.055,.075,.925,.945,.965):
                draw.line(((a[0]+(b[0]-a[0])*u)*SCALE,(a[1]+(b[1]-a[1])*u)*SCALE,
                           (d[0]+(c[0]-d[0])*u)*SCALE,(d[1]+(c[1]-d[1])*u)*SCALE),fill=(211,209,179),width=3)
        elif kind=="tree":
            x,y,r=f["x"]*SCALE,f["y"]*SCALE,f["r"]*SCALE
            draw.ellipse((x-r+r*.3,y-r+r*.3,x+r+r*.3,y+r+r*.3),fill=(25,43,34))
            if f["kind"]=="pine":
                draw.polygon([(x,y-r),(x+r,y+r*.5),(x-r,y+r*.5)],fill=tuple(f["color"]))
            else:
                draw.ellipse((x-r,y-r,x+r,y+r),fill=tuple(f["color"]))
                draw.ellipse((x-r*.75,y-r*.75,x+r*.45,y+r*.45),fill=tuple(min(255,c+12) for c in f["color"]))
        elif kind=="rock":
            draw.polygon(p,fill=tuple(f["color"]));draw.line(p[:2],fill=(155,156,138),width=1)
    return image


def build():
    DEST.mkdir(parents=True,exist_ok=True);ARTIFACT.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(SEED)
    # Aligned elongated ridges, crossed by a broad alluvial valley. This is an
    # authored topography, not a enlarged existing picture or periodic tile.
    x=np.linspace(0,WORLD[0],RASTER[0],dtype=np.float32)[None,:]
    y=np.linspace(0,WORLD[1],RASTER[1],dtype=np.float32)[:,None]
    broad=noise(rng,(22,32));medium=noise(rng,(70,104));fine=noise(rng,(220,330))
    ridge=np.square(np.clip(1-np.abs(perlin(rng,14,22)),0,1))
    ridge2=np.square(np.clip(1-np.abs(perlin(rng,32,48)),0,1))
    ridge3=np.square(np.clip(1-np.abs(perlin(rng,70,106)),0,1))
    ridge=ridge*.59+ridge2*.28+ridge3*.13
    del ridge2,ridge3
    mass=np.zeros((RASTER[1],RASTER[0]),np.float32)
    for cx,cy,rx,ry,amp,angle in [(1700,2100,1700,2700,.82,-.4),(3600,1050,1150,1700,.68,-.9),
                                (10800,1050,2050,1400,.74,.45),(10900,4140,1400,1450,.47,.2),
                                (960,7470,1500,950,.43,-.7),(6160,7460,1700,920,.46,.1)]:
        xx=(x-cx)*math.cos(angle)+(y-cy)*math.sin(angle)
        yy=-(x-cx)*math.sin(angle)+(y-cy)*math.cos(angle)
        mound=np.exp(-1.8*((xx/rx)**2+(yy/ry)**2))*amp
        mass=np.maximum(mass,mound)
    height=np.clip(.11+.14*broad+.05*medium+mass*(.47+.56*ridge)+.025*fine,0,1)
    del broad,medium,ridge
    # A directional raking light reveals both major slopes and fine erosion.
    gy,gx=np.gradient(height)
    shade=np.clip(.91+np.tanh((gx*.75+gy*.85)*170)*.32,.54,1.23)
    vegetation=noise(rng,(43,62))
    rock=np.clip((height-.45)/.4,0,1)
    rgb=np.empty((RASTER[1],RASTER[0],3),np.uint8)
    for i,(low,high) in enumerate(((68,145),(88,148),(65,132))):
        channel=(low+(vegetation-.5)*14)*(1-rock)+high*rock
        channel=channel*shade+(fine-.5)*12
        rgb[:,:,i]=np.clip(channel,0,255).astype(np.uint8)
    del rock,vegetation,shade,gx,gy,fine,mass
    image=Image.fromarray(rgb,"RGB");del rgb
    print("Relief generated",flush=True)
    features=geography(height);del height
    image=draw_geography(image,features)
    levels=[];current=image
    for level in range(5):
        folder=DEST/str(level);folder.mkdir(exist_ok=True)
        w,h=current.size
        for row in range(math.ceil(h/512)):
            for col in range(math.ceil(w/512)):
                current.crop((col*512,row*512,min(w,(col+1)*512),min(h,(row+1)*512))).save(folder/f"{col}_{row}.png",optimize=True)
        levels.append({"width":w,"height":h,"columns":math.ceil(w/512),"rows":math.ceil(h/512)})
        print(f"Level {level}: {w} x {h}",flush=True)
        current=current.resize((w//2,h//2),Image.Resampling.LANCZOS)
    overview=image.resize((1800,1200),Image.Resampling.LANCZOS)
    overview.save(ARTIFACT/"terrain_overview.png")
    for name,view in {"town":(7460,2390,840,600),"airport":(9200,5900,2100,1100),
                      "mountains":(1300,1300,1600,1100),"river_bridge":(5380,3250,1100,750)}.items():
        l,t,w,h=view
        image.crop((round(l*SCALE),round(t*SCALE),round((l+w)*SCALE),round((t+h)*SCALE))).resize((1200,800),Image.Resampling.LANCZOS).save(ARTIFACT/f"terrain_{name}.png")
    data={"name":"河谷联合作业试验区","seed":SEED,"world_size":WORLD,"raster_size":RASTER,"tile_size":512,
          "levels":levels,"feature_counts":dict(Counter(f["type"] for f in features)),
          "landmarks":{
              "riverbank_town":{"name":"东岸镇","position":[7840,2720]},
              "valley_town":{"name":"河谷镇","position":[4240,4470]},
              "joint_base":{"name":"联合作业营区","position":[8860,4650]},
              "industrial":{"name":"东岸仓储区","position":[8980,3480]},
              "airport":{"name":"河谷机场","position":[10200,6470]},
              "central_bridge":{"name":"河谷大桥","position":[5710,3530]},
              "lake":{"name":"青岚湖","position":[6960,1480]},
              "western_ridge":{"name":"西岭山地","position":[1900,2350]}},
          "features":features}
    (DEST/"terrain.json").write_text(json.dumps(data,ensure_ascii=False,separators=(",",":")),encoding="utf-8")
    (ARTIFACT/"terrain_manifest.json").write_text(json.dumps({k:v for k,v in data.items() if k!="features"},ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(data["feature_counts"],ensure_ascii=False),flush=True)
    print(f"Assets: {sum(1 for p in DEST.rglob('*.png'))} tiles",flush=True)


if __name__=="__main__":
    build()
