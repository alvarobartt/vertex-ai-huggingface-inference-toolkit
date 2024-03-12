# 🤗 Hugging Face Inference Toolkit for Google Cloud Vertex AI

> [!WARNING]
> This is still very at a very early stage and subject to major changes.

## Features

* 🤗 Straight forward way of deploying models from the Hugging Face Hub in Vertex AI
* 🐳 Automatically build Custom Prediction Routines (CPR) for Hugging Face Hub models using `transformers.pipeline`
* 📦 Everything is packaged within a single method, providing more flexibility and ease of usage than the former `google-cloud-aiplatform` SDK for custom models
* 🔌 Seamless integration for running inference on top of any model from the Hugging Face Hub in Vertex AI thanks to `transformers`
* 🔍 Includes custom `logging` messages for better monitoring and debugging via Google Cloud Logging

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
```

Once deployed we can send request to it via `cURL`:

```bash
curl -X POST -H "Content-Type: application/json" -d '{"sequences": "Messi is the GOAT", "candidate_labels": ["football", "basketball", "baseball"]}' <VERTEX_AI_ENDPOINT_URL>/predict
```

<details>
    <summary><b>Example on running on different versions (`torch`, CUDA, Ubuntu, etc.)</b></summary></br>

```python
from vertex_ai_huggingface_inference_toolkit import TransformersModel

model = TransformersModel(
    model_name_or_path="facebook/bart-large-mnli",
    framework="torch",
    framework_version="2.1.0",
    python_version="3.9",
    cuda_version="11.8.0",
    environment_variables={
        "HF_TASK": "zero-shot-classification",
    },
)
```
</details>

<details>
    <summary><b>Example on running on existing Docker image</b></summary></br>

To ensure the consistency of the following approach, the image should have been generated using `vertex_ai_huggingface_inference_toolkit` in advance.

```python
from vertex_ai_huggingface_inference_toolkit import TransformersModel

model = TransformersModel(
    model_name_or_path="facebook/bart-large-mnli",
    image_uri="us-east1-docker.pkg.dev/huggingface-cloud/vertex-ai-huggingface-inference-toolkit/py3.11-cu12.3.0-torch-2.2.0-transformers-4.38.2:latest",
    environment_variables={
        "HF_TASK": "zero-shot-classification",
    },
)
```
</details>

<details>
    <summary><b>Example on running TinyLlama for `text-generation`</b></summary></br>

```python
from vertex_ai_huggingface_inference_toolkit import TransformersModel

model = TransformersModel(
    project_id="my-project",
    location="us-east1",
    model_name_or_path="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    model_kwargs={"torch_dtype": "float16", "attn_implementation": "flash_attention_2"},
    extra_requirements=["flash-attn --no-build-isolation"],
    environment_variables={
        "HF_TASK": "text-generation",
    },
)
```
</details>

## References / Acknowledgements

This work is heavily inspired by [`sagemaker-huggingface-inference-toolkit`](https://github.com/aws/sagemaker-huggingface-inference-toolkit) early work from Philipp Schmid, Hugging Face, and Amazon Web Services.
