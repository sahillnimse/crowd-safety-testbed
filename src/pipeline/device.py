"""
Central device resolution for the whole testbed.
"""

import torch


def print_gpu_report() -> None:
    print("=" * 60)
    print("GPU / CUDA report")
    print("=" * 60)
    print(f"torch version:        {torch.__version__}")
    print(f"torch.cuda available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"CUDA version (torch): {torch.version.cuda}")
        print(f"device count:         {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            name = torch.cuda.get_device_name(i)
            total_gb = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)
            print(f"  [{i}] {name} ({total_gb:.1f} GB)")
    else:
        built_with_cuda = getattr(torch.version, "cuda", None) is not None
        print("No CUDA device visible to torch. Likely causes:")
        if not built_with_cuda:
            print("  - torch was installed WITHOUT CUDA support (CPU-only wheel).")
            print("    Fix: reinstall torch from the CUDA index, e.g.:")
            print("    pip uninstall torch torchvision")
            print("    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
        else:
            print("  - torch has CUDA support built in, but no GPU/driver was found.")
            print("    Check: `nvidia-smi` runs and shows your GPU.")
            print("    Check: the CUDA version torch was built for matches your driver.")
    print("=" * 60)


#: Values that all mean "pick for me".  Callers reach this function from the
#: web API, the route-session API and several CLI scripts, and each had its
#: own idea of how to spell "unset" -- ``/api/jobs`` normalised "" to None
#: before calling, ``/api/sessions`` did not, so the same Auto-detect choice
#: in the UI worked on one path and raised ValueError on the other.
#: Normalising here fixes every caller at once instead of one at a time.
_AUTO_DEVICE_ALIASES = frozenset({"", "auto", "default", "none"})


def resolve_device(requested: str | None = None) -> str:
    if isinstance(requested, str) and requested.strip().lower() in _AUTO_DEVICE_ALIASES:
        requested = None

    if requested is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cpu":
            print("[device] No GPU detected — running on CPU. "
                  "Run `python -m pipeline.device` for a diagnostic report.")
        return device

    requested = requested.strip().lower()

    if requested == "cpu":
        return "cpu"

    if requested.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"Requested device={requested!r} but torch.cuda.is_available() is False. "
                "Run `python -m pipeline.device` to diagnose, or pass --device cpu "
                "if you intend to run on CPU."
            )
        idx = 0
        if ":" in requested:
            idx = int(requested.split(":")[1])
        if idx >= torch.cuda.device_count():
            raise RuntimeError(
                f"Requested {requested!r} but only {torch.cuda.device_count()} "
                f"CUDA device(s) visible."
            )
        return requested

    raise ValueError(
        f"Unrecognized device: {requested!r}. Expected 'cpu', 'cuda', "
        f"'cuda:N', or one of {sorted(_AUTO_DEVICE_ALIASES)} to auto-detect."
    )


def require_gpu() -> None:
    if not torch.cuda.is_available():
        print_gpu_report()
        raise RuntimeError(
            "GPU required but not available (torch.cuda.is_available() == False). "
            "See report above. Aborting instead of silently running on CPU."
        )


if __name__ == "__main__":
    print_gpu_report()