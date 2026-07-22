# render_one_view.py 使用说明

`render_one_view.py` 是一个基于 `gsplat` 的 3DGS PLY 渲染工具，用来从给定虚拟相机位置生成 2D 图像。当前主要用途是：根据已有机器人/扫描轨迹中的空间坐标，在 3DGS 场景中生成对应视角的 2D 训练图像。

## 推荐命令

当前默认模式是 `single`，即根据一个相机位置和一个相机朝向只渲染一张 RGB 图像。下面这条命令可以从扫描起点 `(0, 0, 0)` 朝 `-X` 方向渲染一张图：

```powershell
python C:\task\xlk_work\tools\render_one_view.py `
  --ply C:\task\xlk_work\MindCloudXAI_output\test1_yup.ply `
  --eye 0 0 0 `
  --look-dir -1 0 0 `
  --fov 70 `
  --near-plane 0.001 `
  --out C:\task\xlk_work\tools\render_output2D\single_origin_negx.png
```

如果想一次从扫描起点 `(0, 0, 0)` 朝六个轴向都渲染图像，可以显式使用 `panorama` 模式：

```powershell
python C:\task\xlk_work\tools\render_one_view.py `
  --ply C:\task\xlk_work\MindCloudXAI_output\test1_yup.ply `
  --mode panorama `
  --eye 0 0 0 `
  --fov 70 `
  --near-plane 0.001 `
  --output-dir C:\task\xlk_work\tools\render_output2D\origin_panorama
```

`single` 模式会生成一张 PNG；`panorama` 模式会生成 6 张 PNG 和一个 `render_manifest.txt`。文件名中会包含相机位置 `eye` 和朝向目标点 `tgt`，便于回溯每张图对应的虚拟相机参数。

## 当前重要结论

基于 `test1_yup.ply` 的验证结果：

- `(0, 0, 0)` 基本可以认为是扫描任务起点，也就是扫描设备/操作者当时开始扫描的位置。
- 从 `(0, 0, 0)` 使用 `--mode panorama` 朝周围看，可以看到期望的教室室内场景。
- 当前场景坐标中，`-Y` 方向对应现实世界的竖直向上方向。因此脚本默认使用 `--up-axis -y`。
- `X` 和 `Z` 构成主要水平面。
- 与现实现场和 SuperSplat 对照后，之前相机的水平轴曾经发生左右镜像；现在已经在 `look_at_world_to_camera()` 中修正叉乘顺序。
- 当前看起来 `Z` 轴方向相对于现实直觉可能是反的。后续如果用真实轨迹坐标生成图像，需要用现场/SuperSplat 对照确认“朝前”到底对应 `+X/-X/+Z/-Z` 中哪一个方向。

一个已观察到的方向参考：

- `00_panorama_posx...png`：相机在 `(0,0,0)`，朝 `+X` 看。
- `01_panorama_negx...png`：相机在 `(0,0,0)`，朝 `-X` 看。
- `02_panorama_posy...png`：相机在 `(0,0,0)`，朝 `+Y` 看。
- `03_panorama_negy...png`：相机在 `(0,0,0)`，朝 `-Y` 看。
- `04_panorama_posz...png`：相机在 `(0,0,0)`，朝 `+Z` 看。
- `05_panorama_negz...png`：相机在 `(0,0,0)`，朝 `-Z` 看。

## 三种渲染模式

### single 模式，默认基础模式，期望用于后续机器人导航

`single` 模式只渲染一张 RGB 图像。后续根据机器人轨迹批量生成训练数据时，最推荐以这个模式作为基础：每个轨迹点和每个相机朝向调用一次，得到一张图。

使用 `--look-dir` 指定朝向：

```powershell
python C:\task\xlk_work\tools\render_one_view.py `
  --ply C:\task\xlk_work\MindCloudXAI_output\test1_yup.ply `
  --mode single `
  --eye 0 0 0 `
  --look-dir -1 0 0 `
  --fov 70 `
  --out C:\task\xlk_work\tools\render_output2D\single_origin_negx.png
```

使用 `--look-at` 指定目标点：

```powershell
python C:\task\xlk_work\tools\render_one_view.py `
  --ply C:\task\xlk_work\MindCloudXAI_output\test1_yup.ply `
  --mode single `
  --eye 0 0 0 `
  --look-at -1 0 0 `
  --out C:\task\xlk_work\tools\render_output2D\single_origin_lookat.png
```

如果已有完整外参矩阵，可以用 `--viewmat` 传入 row-major 的 world-to-camera 4x4 矩阵。此时 `--eye`、`--look-at`、`--look-dir` 会被忽略。

内参默认由 `--fov`、`--width`、`--height` 推出；也可以显式传入 `--fx --fy --cx --cy`。

### panorama 模式，推荐用于调试，可观察场景全貌

`panorama` 模式把相机固定在 `--eye X Y Z`，然后朝六个轴向看。它适合快速检查某个轨迹点周围是否能看到正确内容。

```powershell
python C:\task\xlk_work\tools\render_one_view.py `
  --ply C:\task\xlk_work\MindCloudXAI_output\test1_yup.ply `
  --mode panorama `
  --eye 0 0 0 `
  --fov 70 `
  --output-dir C:\task\xlk_work\tools\render_output2D\origin_panorama
```

后续正式生成训练数据时，可以先用 `panorama` 检查，再用 `single` 按需要的相机朝向逐张生成。

### orbit 模式，不推荐用于机器人视角，仅用于render_one_view.py代码生成过程调试

`orbit` 模式会先自动寻找高斯点最密集区域，然后把相机放在该点周围，从六个方向朝中心看。这个模式适合检查场景外观，但不适合模拟机器人第一视角。

之前尝试生成过的 orbit 图像中：

```text
target ~= (-0.0049, -0.0010, 0.0011)
distance = 0.1875
```

虽然 `0.1875` 看起来不大，但对于当前教室场景和原点附近的局部视角来说，这个偏移已经会把相机放到教室结构外侧或贴近外层高斯，导致看到的是教室上下层、横截面、外壳，而不是站在室内环视。因此后续轨迹生成不要把 `distance = 0.1875` 当作机器人步长参考；实际轨迹单位长度应该设置得更小，并结合真实坐标尺度逐步验证。

## 常用参数

- `--ply`：输入 3DGS PLY 文件路径。
- `--output-dir`：输出 PNG 和 `render_manifest.txt` 的目录。
- `--mode`：渲染模式，`single`、`panorama` 或 `orbit`。默认 `single`。
- `--eye X Y Z`：`single` 和 `panorama` 模式下的相机位置。
- `--look-dir DX DY DZ`：`single` 模式下的相机朝向，默认 `-1 0 0`。
- `--look-at X Y Z`：`single` 模式下的目标点，优先级高于 `--look-dir`。
- `--viewmat M ...`：`single` 模式下的 world-to-camera 4x4 外参矩阵，row-major 16 个数。
- `--views`：最多输出几个轴向视角，默认 6。
- `--fov`：视场角，默认 50。室内机器人视角可以尝试 60-90。
- `--fx --fy --cx --cy`：可选相机内参。如果不指定，则使用 `--fov` 自动计算。
- `--near-plane`：近裁剪面，默认 0.001。相机在场景内部时建议保持较小。
- `--up-axis`：图像上方向对应的世界轴，当前默认 `-y`。
- `--unit-scale`：PLY 坐标和高斯 scale 的缩放，默认 0.001。
- `--sh-degree`：颜色球谐阶数，默认 `auto`。一般不要改。
- `--background`：背景色，`black` 或 `white`。
- `--max-gaussians`：快速测试时可随机抽样部分高斯；正式渲染保持 0。

`orbit` 模式专用参数：

- `--distance-mult`：相机离自动密集中心的距离倍数。
- `--focus-grid`：寻找密集区域的 voxel 网格大小。
- `--focus-neighborhood`：密集 voxel 周围参与估计的邻域大小。
- `--min-local-radius`：局部半径下限。当前默认 0.15。

## 后续训练数据生成建议

1. 使用真实/估计轨迹点作为 `--eye`，采用默认的 `--mode single` 逐张生成 RGB。
2. 先用少量轨迹点验证坐标系，不要一开始批量生成。
3. 每个点至少输出水平面四向视角，例如 `+X/-X/+Z/-Z`；如果需要全景，再保留 `+Y/-Y`。
4. 因为现实竖直向上约等于 `-Y`，后续相机姿态应默认保留 `--up-axis -y`。
5. 如果发现前进方向或左右关系仍与现实/ SuperSplat 不一致，优先检查轨迹坐标系到 PLY 坐标系的轴映射，尤其是 `Z` 是否需要取反。
6. 轨迹步长要从很小的坐标增量开始试，例如 0.02、0.005、0.001 这类量级，而不要参考之前 orbit 的 `0.1875` 偏移。
