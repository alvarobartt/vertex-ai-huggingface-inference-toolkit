# Copyright 2021 The HuggingFace Team, Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# NOTE: this file defines a function named `download_files_from_hub` that
# has been ported from `sagemaker_huggingface_inference_toolkit/transformers_utils.py`
# as an adaptation of `_load_model_from_hub` function, in order to properly filter the
# files to download from the Hub based on the framework to use, with some minor changes.

import os

from huggingface_hub import HfFileSystem, snapshot_download

FRAMEWORK_MAPPING = {
    "torch": "pytorch*",
    "tensorflow": "tf*",
    "flax": "flax*",
    "rust": "rust*",
    "onnx": "*onnx*",
    "safetensors": "*safetensors",
    "coreml": "*mlmodel",
    "tflite": "*tflite",
    "savedmodel": "*tar.gz",
    "openvino": "*openvino*",
    "ckpt": "*ckpt",
}

FRAMEWORK_NAMING = {
    "pytorch": "torch",
    "pt": "torch",
    "jax": "flax",
    "tf": "tensorflow",
}


def download_files_from_hub(repo_id: str, framework: str) -> str:
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

    framework = FRAMEWORK_NAMING.get(framework.lower(), framework.lower())
    if framework not in ["torch", "tensorflow", "flax"]:
        raise NotImplementedError(
            f"Support for framework {framework} hasn't been tested yet, so deploying"
            " the model in Vertex AI may not work as expected while this package is still"
            " in beta."
        )

    if framework == "torch":
        fs = HfFileSystem()
        if any(file.endswith(".safetensors") for file in fs.ls(repo_id, detail=False)):  # type: ignore
            framework = "safetensors"

    pattern = FRAMEWORK_MAPPING.get(framework, None)
    ignore_patterns = list(set(FRAMEWORK_MAPPING.values()))
    if pattern in ignore_patterns:
        ignore_patterns.remove(pattern)

    return snapshot_download(repo_id, ignore_patterns=ignore_patterns or None)  # type: ignore
