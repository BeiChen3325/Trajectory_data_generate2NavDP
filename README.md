# ZJUgive_DataEngine_For_RobotNav

## Environment Setup

Use conda to create an isolated Python environment.

The original development environment used Python `3.10.20`. The recommended environment name is `robotnav`.

### Windows + Python 3.10 + CUDA 12.1

Create and activate the conda environment:

```powershell
conda create -n robotnav python=3.10
conda activate robotnav
```

Install PyTorch, torchvision, and gsplat:

```powershell
python -m pip install torch==2.1.2+cu121 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
python -m pip install gsplat==1.5.2+pt21cu121 --index-url https://docs.gsplat.studio/whl
```

Install the remaining Python dependencies:

```powershell
python -m pip install -r requirements.txt
```

## Data Files

The default scripts expect LAS/PLY files exported from the LiXingYun platform to be placed next to this repository, under:

```python
DATA_DIR = PROJECT_ROOT.parent / "MindCloudXAI_output"
```

Recommended layout:

```text
xlk_work/
  ZJUgive_DataEngine_For_RobotNav/
  MindCloudXAI_output/
    test1-pointcloud-0704.las
    test1_yup.ply
```

With this layout, the default input paths resolve automatically. Render outputs are written under `render_output2D/`, and trajectory outputs are written under `trajectory_work/outputs/`.

This environment pins PyTorch `2.1.2+cu121` and gsplat `1.5.2+pt21cu121` for Windows, Python 3.10, and CUDA 12.1.

`requirements.txt` intentionally excludes `torch`, `torchvision`, and `gsplat` because those packages are tightly coupled to the operating system, Python version, PyTorch version, and CUDA version.

### If Online Install Fails

The gsplat wheel is downloaded from a GitHub-hosted release/index. On networks where GitHub access is unstable, download the wheel manually in a browser and install it from the local file before installing the remaining requirements.

Relevant sources:

- PyTorch CUDA 12.1 wheel index: https://download.pytorch.org/whl/cu121/torch/
- PyTorch wheel used by this environment: https://download.pytorch.org/whl/cu121/torch-2.1.2%2Bcu121-cp310-cp310-win_amd64.whl
- gsplat wheel index: https://docs.gsplat.studio/whl/gsplat/
- gsplat wheel used by this environment: https://github.com/nerfstudio-project/gsplat/releases/download/v1.5.2/gsplat-1.5.2%2Bpt21cu121-cp310-cp310-win_amd64.whl

Example local wheel install:

```powershell
python -m pip install .\wheels\gsplat-1.5.2+pt21cu121-cp310-cp310-win_amd64.whl
python -m pip install -r requirements.txt
```

The PyTorch wheel is about 2.5 GB, so it should not be committed to this repository. The gsplat wheel is much smaller and can be kept locally under a `wheels/` directory if offline deployment is needed.

### Other Platforms

If you are using a different operating system, Python version, PyTorch version, or CUDA version, install matching versions of `torch`, `torchvision`, and `gsplat` first, then install the remaining packages with `python -m pip install -r requirements.txt`.
