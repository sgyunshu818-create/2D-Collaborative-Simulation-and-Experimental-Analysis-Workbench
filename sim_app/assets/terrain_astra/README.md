# 小范围山区河谷地图

本目录提供 `astra_mountain` 地形的原图、地标和运行瓦片。示例场景为项目根目录下的 `configs/mountain_scene.json`。

- `terrain_astra.png`：1536 × 1024 原始地图图像。
- `landmarks.json`：地标和显示路径数据。
- `runtime/terrain.json`：运行瓦片描述。
- `runtime/0`、`runtime/1`、`runtime/2`：原生层及两层降采样 PNG。

虚拟世界范围为 12000 × 8000 仿真单位，最高交互缩放为 64 倍。缩放超过影像原始尺度后仅放大已有像素，不会增加地形细节。运行时按视口读取瓦片并叠加装备、路线和标签。

在项目根目录重新打包：

```powershell
.\.venv\Scripts\python.exe tools/build_photographic_terrain.py
```

重建使用核心运行环境中的 Pygame，无需安装图像实验的可选依赖。图像和标注属于二维视觉背景，不承担高程、道路通行或碰撞计算，也没有真实经纬度绑定。
