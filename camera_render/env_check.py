import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import gsplat
import cv2
import open3d
import scipy
import plyfile

print("gsplat版本:", gsplat.__version__)
print("torch版本:", torch.__version__)

import torchvision
print("torchvision版本:", torchvision.__version__)

try:
    import gsplat.csrc as csrc

    print("[OK] gsplat 预编译扩展 csrc 加载成功:", csrc.__file__)
except Exception as e:
    print("[FAIL] gsplat 预编译扩展 csrc 加载失败:", repr(e))

try:
    from gsplat.cuda._backend import _C

    print("[OK] gsplat CUDA 后端 _C 加载成功:", _C)
except ModuleNotFoundError as e:
    print("[FAIL] gsplat CUDA 后端加载失败:", repr(e))
    if e.name == "pkg_resources":
        print("   原因: 当前 setuptools 不再提供 pkg_resources，但 torch 2.1 的 cpp_extension 仍会导入它。")
        print("   建议: 降级 setuptools，例如 python -m pip install \"setuptools<81\"")
except Exception as e:
    print("[FAIL] gsplat CUDA 后端加载失败:", repr(e))

print("CUDA可用:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
