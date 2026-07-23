# RobotNav 项目协作约定

这是一个实验性点云、3D Gaussian 渲染和机器人导航数据工程项目。优先保持代码直接、可运行和容易修改，不为旧目录、旧命令或旧参数保留兼容层。

## 目录职责

- `src/robotnav/`：项目源码。
- `src/robotnav/rendering/`：渲染和点云检查。
- `src/robotnav/navigation/`：地图、规划和轨迹后处理。
- `configs/`：默认 TOML 参数。
- `data/input/`：本地输入数据，不提交大型点云文件。
- `outputs/`：运行生成的结果，不提交实验输出。

## 配置约定

- `configs/*.toml` 是运行参数和路径配置的唯一事实源。代码不得硬编码数据目录、输出目录、LAS/PLY 文件名或完整默认路径。
- 路径配置目前拆分为 `data_dir`、`output_dir`、`las_filename` 和可选的 `ply_filename`，禁止把目录和文件名混在单个路径字段中。
- `src/robotnav/config.py` 负责定位项目根目录、读取 TOML 和组合 `Path`；业务模块必须通过 `load_path_config()` 或其返回对象读取路径。
- 不得在渲染模块、导航模块或命令入口中重新定义第二套路径常量；路径工具也必须集中在 `config.py`。
- 新增命令时，应为其增加对应的 `configs/<command>.toml`，并让命令入口从该 TOML 获取默认路径。

## 开发约定

- 使用 `pathlib.Path`，不要写死机器相关绝对路径。
- 命令行入口负责参数处理，算法函数接收明确的配置或参数。
- 随机过程必须使用显式 seed。
- 新代码使用 Python 3.10+ 类型标注。
- 先运行环境检查，再运行 GPU 相关流程。

## 验证

```bash
uv run robotnav-env-check
uv run ruff check .
uv run ruff format --check .
uv run ty check
```
