import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def pytest_collection_modifyitems(config, items):
    try:
        import torch
        has_cuda = torch.cuda.is_available()
    except ImportError:
        has_cuda = False

    skip_gpu = __import__("pytest").mark.skip(reason="No CUDA available")
    skip_atari = __import__("pytest").mark.skip(reason="Atari ROM not available")

    for item in items:
        if "requires_gpu" in item.keywords and not has_cuda:
            item.add_marker(skip_gpu)
        if "requires_atari" in item.keywords:
            try:
                import ale_py  # noqa: F401
            except ImportError:
                item.add_marker(skip_atari)
