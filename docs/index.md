# 🤗 Hugging Face Inference Toolkit for Google Cloud Vertex AI

This is still very at a very early stage and subject to major changes.

## Examples

### From 🤗 Hub to Google Cloud Storage

```python
from vertex_ai_huggingface_inference_toolkit import TransformersModel

model = TransformersModel(
    model_name_or_path="facebook/bart-large-mnli",
    framework="torch",
    framework_version="2.2.0",
    transformers_version="4.38.2",
    python_version="3.10",
    cuda_version="12.3.0",
    environment_variables={
        "HF_TASK": "zero-shot-classification",
    },
)
model.deploy(
    machine_type="n1-standard-4",
    accelerator_type="NVIDIA_TESLA_T4",
    accelerator_count=1,
)
```

## Everything within Vertex AI

```python
from vertex_ai_huggingface_inference_toolkit import TransformersModel

model = TransformersModel(
    framework="torch",
    framework_version="2.2.0",
    transformers_version="4.38.2",
    python_version="3.10",
    cuda_version="12.3.0",
    environment_variables={
        "HF_HUB_ID": "facebook/bart-large-mnli",
        "HF_TASK": "zero-shot-classification",
    },
)
model.deploy(
    machine_type="n1-standard-4",
    accelerator_type="NVIDIA_TESLA_T4",
    accelerator_count=1,
)
```

## References / Acknowledgements

This work is heavily inspired by [`sagemaker-huggingface-inference-toolkit`](https://github.com/aws/sagemaker-huggingface-inference-toolkit) early work from Philipp Schmid, Hugging Face, and Amazon Web Services.

