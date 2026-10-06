# 思源宋体 / Source Han Serif

界面使用随项目分发的思源宋体简体中文静态字体：

- `SourceHanSerifCN-Regular.otf`：正文、表格、数值、地图标签。
- `SourceHanSerifCN-Bold.otf`：标题和强调文字。

来源：Adobe 官方 [Source Han Serif](https://github.com/adobe-fonts/source-han-serif/tree/release/SubsetOTF/CN)，下载日期2026-10-05。原始字体未修改，随附官方 [SIL Open Font License 1.1](LICENSE.txt)。完整下载地址、版本信息及文件校验值见 `sources.json`。

Windows 工作台只将字体注册到当前进程，不写系统字体目录或注册表。Pygame 直接读取本目录的字体文件。运行时无需联网；进程退出后临时注册失效。若素材文件缺失，程序会退回已安装的中文宋体字体。
