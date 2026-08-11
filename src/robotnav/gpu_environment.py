"""GPU preflight checks shared by rendering entry points and diagnostics."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CheckResult:
    passed: bool
    detail: str = ""


@dataclass(frozen=True)
class GPUEnvironmentReport:
    driver: CheckResult
    cuda_runtime: CheckResult
    pytorch_cuda: CheckResult
    gpu: CheckResult
    gsplat_cuda: CheckResult

    @property
    def passed(self) -> bool:
        return all(
            result.passed
            for result in (
                self.driver,
                self.cuda_runtime,
                self.pytorch_cuda,
                self.gpu,
                self.gsplat_cuda,
            )
        )


PROJECT_ROOT = Path(__file__).resolve().parents[2]
GPU_ENVIRONMENT_REPORT_PATH = PROJECT_ROOT / "gpu_environment_report.txt"


def ensure_cuda_toolkit_environment() -> Path | None:
    """Expose the project's CUDA toolkit wheel to extension builders such as gsplat."""
    configured = os.environ.get("CUDA_HOME")
    if configured and (Path(configured) / "bin" / "nvcc").is_file():
        return Path(configured)
    candidates = sorted(Path(sys.prefix).glob("lib/python*/site-packages/nvidia/cu*/bin/nvcc"))
    if not candidates:
        return None
    cuda_home = candidates[-1].parents[1]
    os.environ["CUDA_HOME"] = str(cuda_home)
    bin_dir = str(cuda_home / "bin")
    path_items = os.environ.get("PATH", "").split(os.pathsep)
    if bin_dir not in path_items:
        os.environ["PATH"] = os.pathsep.join([bin_dir, *path_items])
    return cuda_home


def check_gpu_environment() -> GPUEnvironmentReport:
    """Check driver, PyTorch CUDA, a selected device, and gsplat's extension."""
    ensure_cuda_toolkit_environment()
    try:
        driver_process = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        driver_name = driver_process.stdout.strip().splitlines()
        driver = CheckResult(
            driver_process.returncode == 0 and bool(driver_name), ", ".join(driver_name)
        )
        if not driver.passed:
            detail = driver_process.stderr.strip() or "nvidia-smi returned no GPU"
            driver = CheckResult(False, detail)
    except (FileNotFoundError, subprocess.TimeoutExpired) as error:
        driver = CheckResult(False, str(error))

    try:
        import torch
    except ImportError as error:
        unavailable = CheckResult(False, f"PyTorch import failed: {error}")
        return GPUEnvironmentReport(driver, unavailable, unavailable, unavailable, unavailable)

    runtime_version = torch.version.cuda
    cuda_runtime = CheckResult(
        bool(runtime_version), str(runtime_version or "not compiled with CUDA")
    )
    pytorch_cuda = CheckResult(bool(torch.cuda.is_available()), "torch.cuda.is_available()")
    if pytorch_cuda.passed:
        try:
            gpu = CheckResult(True, torch.cuda.get_device_name(0))
        except RuntimeError as error:
            gpu = CheckResult(False, str(error))
    else:
        gpu = CheckResult(False, "no CUDA device available to PyTorch")

    if not (cuda_runtime.passed and pytorch_cuda.passed):
        gsplat_cuda = CheckResult(False, "skipped because PyTorch CUDA is unavailable")
    else:
        try:
            from gsplat.cuda._backend import _C

            gsplat_cuda = CheckResult(_C is not None, "gsplat CUDA extension loaded")
        except Exception as error:  # gsplat can raise a build/import-specific exception.
            gsplat_cuda = CheckResult(False, f"{type(error).__name__}: {error}")
    return GPUEnvironmentReport(driver, cuda_runtime, pytorch_cuda, gpu, gsplat_cuda)


def gpu_environment_fingerprint(report: GPUEnvironmentReport) -> str:
    """Return a stable fingerprint for the driver/device/runtime/extension combination."""
    payload = {
        "driver": report.driver.detail,
        "cuda_runtime": report.cuda_runtime.detail,
        "gpu": report.gpu.detail,
        "gsplat": report.gsplat_cuda.detail,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def require_gpu_environment_lock(report: GPUEnvironmentReport) -> None:
    """Require an existing environment snapshot to match the active GPU stack."""
    if not GPU_ENVIRONMENT_REPORT_PATH.is_file():
        return
    match = re.search(
        r"^Environment fingerprint:\s*([0-9a-f]{64})$",
        GPU_ENVIRONMENT_REPORT_PATH.read_text(encoding="utf-8"),
        re.MULTILINE,
    )
    if match is not None and match.group(1) != gpu_environment_fingerprint(report):
        raise RuntimeError(
            "GPU environment changed since gpu_environment_report.txt was generated; "
            "run scripts/export_gpu_environment.sh and review the new report before rendering."
        )


def format_gpu_environment_report(report: GPUEnvironmentReport) -> str:
    def line(label: str, result: CheckResult) -> str:
        suffix = f" ({result.detail})" if result.detail else ""
        return f"{label}: {'PASS' if result.passed else 'FAIL'}{suffix}"

    return "\n".join(
        [
            "GPU Environment Report",
            "",
            line("Driver", report.driver),
            line("CUDA runtime", report.cuda_runtime),
            line("PyTorch CUDA", report.pytorch_cuda),
            line("GPU", report.gpu),
            line("gsplat CUDA", report.gsplat_cuda),
            f"Environment fingerprint: {gpu_environment_fingerprint(report)}",
        ]
    )


def require_cuda_environment() -> None:
    report = check_gpu_environment()
    if not report.passed:
        raise RuntimeError(
            "CUDA environment unavailable, please fix before rendering.\n"
            + format_gpu_environment_report(report)
        )
    require_gpu_environment_lock(report)


def main() -> None:
    report = check_gpu_environment()
    print(format_gpu_environment_report(report))
    if not report.passed:
        raise SystemExit(1)
