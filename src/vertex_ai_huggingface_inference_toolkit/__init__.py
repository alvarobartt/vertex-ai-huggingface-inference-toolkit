"""`vertex_ai_huggingface_inference_toolkit `: 🤗 Hugging Face Inference Toolkit for Google Cloud Vertex AI (similar to SageMaker's Inference Toolkit, but unofficial)"""

__author__ = "Alvaro Bartolome <alvarobartt@gmail.com>"
__version__ = "0.0.2"

from vertex_ai_huggingface_inference_toolkit.diffusers import DiffusersModel
from vertex_ai_huggingface_inference_toolkit.transformers import TransformersModel

__all__ = ["DiffusersModel", "TransformersModel"]
