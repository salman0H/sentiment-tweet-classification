"""Quick sanity check: is torch actually seeing your GPU?

Run this right after installing torch, before running any real experiment.

    python scripts/verify_gpu.py
"""

import torch


def main() -> None:
    print(f"torch version        : {torch.__version__}")
    print(f"torch built with CUDA : {torch.version.cuda}")
    cuda_ok = torch.cuda.is_available()
    print(f"cuda.is_available()  : {cuda_ok}")

    if not cuda_ok:
        print(
            "\nNo GPU detected by torch. Common causes:\n"
            "  - You installed the CPU-only wheel (torch.version.cuda will print 'None' above)\n"
            "  - The NVIDIA driver is missing or older than what this torch build needs\n"
            "  - You're inside a container/VM without GPU passthrough\n"
            "Re-run the install command from https://pytorch.org/get-started/locally/ "
            "matching the CUDA version shown by `nvidia-smi`."
        )
        return

    device_name = torch.cuda.get_device_name(0)
    total_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"device                : {device_name}")
    print(f"total memory          : {total_memory_gb:.1f} GB")

    # A tiny real computation on the GPU, not just a flag check.
    x = torch.rand(2000, 2000, device="cuda")
    y = torch.rand(2000, 2000, device="cuda")
    torch.cuda.synchronize()
    result = x @ y
    torch.cuda.synchronize()
    print(f"test matmul result shape : {tuple(result.shape)} (ran on GPU without errors)")


if __name__ == "__main__":
    main()
