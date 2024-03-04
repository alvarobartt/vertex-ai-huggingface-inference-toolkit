from pathlib import Path

CACHE_PATH = Path.home() / ".cache" / "vertex_ai_huggingface_inference_toolkit"


def get_device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        pass

    try:
        import tensorflow as tf

        return "cuda" if tf.test.is_gpu_available(cuda_only=True) else "cpu"
    except ImportError:
        pass

    try:
        import jax

        return "cuda" if jax.devices("gpu") else "cpu"
    except ImportError:
        pass

    return "cpu"
