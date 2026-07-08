"""
Background hardware monitor.

Samples CPU load, RAM usage, and (if an NVIDIA GPU is present) GPU
utilization, memory, and temperature at a fixed interval, on a separate
thread, for the duration of a training run. The point is to have real
measurements to plot afterwards -- "how hard was the hardware working
during training" -- rather than guessing from the outside.

GPU readings use NVML directly (via `pynvml`) rather than
`torch.cuda.memory_allocated`, because NVML reports the actual device
utilization and memory as the OS sees it, including anything outside of
torch's own allocator.
"""

from __future__ import annotations

import csv
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import psutil

try:
    import pynvml

    _NVML_IMPORTABLE = True
except ImportError:
    _NVML_IMPORTABLE = False


@dataclass
class HardwareSample:
    timestamp: float
    cpu_percent: float
    ram_percent: float
    gpu_util_percent: Optional[float]
    gpu_memory_used_mb: Optional[float]
    gpu_memory_total_mb: Optional[float]
    gpu_temperature_c: Optional[float]


class HardwareMonitor:
    """Use as a context manager around the code you want to profile:

        with HardwareMonitor(interval_seconds=1.0) as monitor:
            train(...)
        monitor.save_csv(Path("results/run_1/hardware_log.csv"))

    If no NVIDIA GPU / driver / `pynvml` is available, GPU columns are
    simply left empty and CPU/RAM are still logged -- it degrades
    gracefully instead of failing the whole training run.
    """

    def __init__(self, interval_seconds: float = 1.0, gpu_index: int = 0):
        self.interval_seconds = interval_seconds
        self.gpu_index = gpu_index
        self._samples: List[HardwareSample] = []
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._gpu_handle = None
        self.gpu_available = False

        if _NVML_IMPORTABLE:
            try:
                pynvml.nvmlInit()
                self._gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
                self.gpu_available = True
            except pynvml.NVMLError:
                self._gpu_handle = None

    def _read_gpu(self):
        if self._gpu_handle is None:
            return None, None, None, None
        util = pynvml.nvmlDeviceGetUtilizationRates(self._gpu_handle)
        mem = pynvml.nvmlDeviceGetMemoryInfo(self._gpu_handle)
        try:
            temp = pynvml.nvmlDeviceGetTemperature(self._gpu_handle, pynvml.NVML_TEMPERATURE_GPU)
        except pynvml.NVMLError:
            temp = None
        return float(util.gpu), mem.used / 1e6, mem.total / 1e6, temp

    def _run(self) -> None:
        psutil.cpu_percent(interval=None)  # first call is a meaningless baseline
        while not self._stop_event.is_set():
            gpu_util, gpu_used, gpu_total, gpu_temp = self._read_gpu()
            self._samples.append(
                HardwareSample(
                    timestamp=time.time(),
                    cpu_percent=psutil.cpu_percent(interval=None),
                    ram_percent=psutil.virtual_memory().percent,
                    gpu_util_percent=gpu_util,
                    gpu_memory_used_mb=gpu_used,
                    gpu_memory_total_mb=gpu_total,
                    gpu_temperature_c=gpu_temp,
                )
            )
            self._stop_event.wait(self.interval_seconds)

    def start(self) -> None:
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval_seconds * 2)
        if self._gpu_handle is not None:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass

    def save_csv(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "elapsed_seconds",
                    "cpu_percent",
                    "ram_percent",
                    "gpu_util_percent",
                    "gpu_memory_used_mb",
                    "gpu_memory_total_mb",
                    "gpu_temperature_c",
                ]
            )
            if not self._samples:
                return
            t0 = self._samples[0].timestamp
            for s in self._samples:
                writer.writerow(
                    [
                        round(s.timestamp - t0, 2),
                        s.cpu_percent,
                        s.ram_percent,
                        s.gpu_util_percent,
                        s.gpu_memory_used_mb,
                        s.gpu_memory_total_mb,
                        s.gpu_temperature_c,
                    ]
                )

    def __enter__(self) -> "HardwareMonitor":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()
