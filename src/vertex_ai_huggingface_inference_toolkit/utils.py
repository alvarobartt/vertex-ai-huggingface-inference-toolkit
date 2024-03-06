from pathlib import Path

CACHE_PATH = Path.home() / ".cache" / "vertex_ai_huggingface_inference_toolkit"
"""Is the path to the default cache directory created for `vertex-ai-huggingface-inference-toolkit`."""


def get_device() -> str:
    """Function to get the device type, either `cpu` or `cuda` (`tpu` not supported yet).

    The available devices will be check from the installed deep learning framework, as its
    the one supposed to run that device, so we need to ensure that's discoverable.

    Returns:
        The device type, either `cpu` or `cuda`.
    """

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
