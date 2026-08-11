"""Expose the CUDA toolkit installed by the ``gpu`` dependency extra.

NVIDIA's pip-distributed CUDA 13 toolkit is installed below
``site-packages/nvidia/cu13`` rather than a system-wide ``/usr/local/cuda``.
gsplat uses PyTorch's standard CUDA discovery, which requires ``CUDA_HOME``
and ``nvcc`` on ``PATH`` for its first JIT build.
"""

from __future__ import annotations

import os
import sysconfig
from pathlib import Path


def configure_cuda_toolkit() -> Path | None:
    """Configure a bundled CUDA toolkit, without replacing a user toolkit."""
    configured_home = os.environ.get("CUDA_HOME")
    if configured_home and (Path(configured_home) / "bin" / "nvcc").is_file():
        return Path(configured_home)

    purelib = Path(sysconfig.get_paths()["purelib"])
    candidates = sorted((purelib / "nvidia").glob("cu*/bin/nvcc"), reverse=True)
    if not candidates:
        return None

    toolkit_home = candidates[0].parent.parent
    # The pip CUDA runtime package ships the SONAME file (``libcudart.so.13``)
    # but not the unversioned development linker name.  PyTorch extensions link
    # with ``-lcudart``, so provide that local, reversible alias when needed.
    cudart_soname = toolkit_home / "lib" / "libcudart.so.13"
    cudart_link = toolkit_home / "lib" / "libcudart.so"
    if cudart_soname.is_file() and not cudart_link.exists():
        cudart_link.symlink_to(cudart_soname.name)
    os.environ["CUDA_HOME"] = str(toolkit_home)
    bin_dir = str(toolkit_home / "bin")
    path_entries = os.environ.get("PATH", "").split(os.pathsep)
    if bin_dir not in path_entries:
        os.environ["PATH"] = os.pathsep.join([bin_dir, *filter(None, path_entries)])
    return toolkit_home
