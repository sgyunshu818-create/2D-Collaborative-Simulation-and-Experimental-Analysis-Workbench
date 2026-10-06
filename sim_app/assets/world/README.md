# 离线世界地形

数据：Natural Earth I with Shaded Relief and Water，3.2.0，1:50,000,000。
官方来源：[数据页](https://www.naturalearthdata.com/downloads/50m-raster-data/50m-natural-earth-1/)。
使用 WGS84 等经纬度投影，范围西经180°至东经180°、南纬90°至北纬90°。

Natural Earth [许可](https://www.naturalearthdata.com/about/terms-of-use/)为公共领域。
完整来源、原始 TIFF 校验值及各瓦片指纹保存在 sources.json。

源图10800×5400像素，打包成1024/2048/4096/8192宽的四级 JPEG 瓦片，各块512×512。
只做采样、切片和格式转换，无生成或补绘地理细节。程序按视野加载，缓存上限48块。
应用绘制时叠加轻度深蓝色遮罩以保持标签对比，原始瓦片颜色未改。

这是全球小比例尺地形背景，无街道级、实时卫星或高程碰撞能力。等经纬投影高纬度有形变，
比例尺对应当前中心纬度附近的地面距离。放大超过底图细节范围会显示小比例尺提示。

`tools/build_world_tiles.py` 仅用于重建瓦片。须将原始 `NE1_50M_SR_W.tif`（仓库不提供）
放入项目根目录的 `artifacts/world_map_20261005/natural_earth_source/NE1_50M_SR_W/`。
读取 TIFF 需要 Pillow；也可在同目录提供同名 PNG 供工具读取，但仍须保留 TIFF 以计算来源校验值。
正常运行使用随附 JPEG 瓦片，不需要原始 TIFF 或 Pillow。
