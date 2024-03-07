# 🤗 Hugging Face Inference Toolkit for Google Cloud Vertex AI

> [!WARNING]
> This is still very at a very early stage and subject to major changes.

## Get started

[Install the `gcloud` CLI](https://cloud.google.com/sdk/docs/install) and authenticate with your Google Cloud account as:

```bash
gcloud init
gcloud auth login
```

Then install `vertex-ai-huggingface-inference-toolkit` via `pip install`:

```bash
pip install vertex-ai-huggingface-inference-toolkit>=0.1.0
```

Or via `uv pip install` for faster installations using [`uv`](https://astral.sh/blog/uv):

```bash
uv pip install vertex-ai-huggingface-inference-toolkit>=0.1.0
```

## Example

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
model.undeploy()
```

## References / Acknowledgements

This work is heavily inspired by [`sagemaker-huggingface-inference-toolkit`](https://github.com/aws/sagemaker-huggingface-inference-toolkit) early work from Philipp Schmid, Hugging Face, and Amazon Web Services.
