import os
import tarfile
from io import BytesIO
from typing import Any, Dict, Optional
from uuid import uuid4

import torch
from google.cloud.aiplatform.prediction.predictor import Predictor
from google.cloud.aiplatform.utils import prediction_utils
from google.cloud.storage import Client
from PIL.Image import Image

from vertex_ai_huggingface_inference_toolkit.diffusers_utils import (
    PIPELINE_TASKS,
)
from vertex_ai_huggingface_inference_toolkit.utils import get_logger


class DiffusersPredictor(Predictor):
    """Custom `Predictor` for the Hugging Face `diffusers` library, that allows
    to load the model and run inference on top of it via the `pipeline` method. This
    class is also in charge of downloading the artifacts from Google Cloud Storage, when
    provided, and loading the model with a custom configuration and with automatic device
    placement, mostly via `accelerate`.
    """

    def __init__(self) -> None:
        """Initializes the `DiffusersPredictor` with a custom logger."""
        self._logger = get_logger("vertex-ai-huggingface-inference-toolkit")

    def load(self, artifacts_uri: Optional[str] = None) -> None:
        """Downloads the model from the given `artifacts_uri` or `HF_HUB_ID` environment
        variable, to load it via `pipeline` and placing it on the right device. So on, the
        outcome of `load` is the assignment of the value for the `_pipeline` attribute, that
        will be used to run the inference via `predict`.

        The `load` method is called within the CPR server during the initialization of the
        server, so it's the first method to be called before running the inference.

        Args:
            artifacts_uri: is the Google Cloud Storage URI to the artifact to serve, which
                will ideally be the directory where the model is stored in Google Cloud Storage.
                Also note it's optional not because the model can be loaded via `HF_HUB_ID`, because
                its mandatory to provide a Google Cloud Storage URI in Vertex AI, but because it
                can be used locally without the need of an artifact URI, as the model can be pulled
                from the Hugging Face Hub.

        Raises:
            RuntimeError: if neither the `artifacts_uri` nor the `HF_HUB_ID` environment variable
                are set, as the model needs to be loaded from somewhere.
        """

        # If the `artifacts_uri` is provided, then we download its contents into the current directory
        if artifacts_uri is not None:
            self._logger.info(
                f"Downloading artifacts from `artifacts_uri='{artifacts_uri}'`"
            )
            prediction_utils.download_model_artifacts(artifacts_uri)
            self._logger.info("Artifacts successfully downloaded!")

        # If the `artifacts_uri` was provided, but the `model.tar.gz` file was not downloaded from it,
        # and the `HF_HUB_ID` environment variable is not set, then we raise an error as the model needs
        # to be loaded from somewhere.
        hub_id = os.getenv("HF_HUB_ID", None)
        file_exists = os.path.exists("model.tar.gz") and os.path.isfile("model.tar.gz")
        if not file_exists and hub_id is None:
            error_msg = "Neither the environment variable `HF_HUB_ID` nor the file `model.tar.gz` exist!"
            self._logger.error(error_msg)
            raise RuntimeError(error_msg)

        # If the `artifacts_uri` was provided, and the `model.tar.gz` file was downloaded from it, then
        # we set the `model_path` to the `transformers-model` directory, otherwise we set it to the `HF_HUB_ID`.
        model_path = "./diffusers-model" if file_exists else hub_id
        if file_exists:
            if hub_id:
                self._logger.warn(
                    f"Since both the provided `artifacts_uri={artifacts_uri}` and the environment"
                    f" variable `HF_HUB_ID={hub_id}` are set, the `artifacts_uri` will be used as"
                    " it has priority over the `HF_HUB_ID` environment variable."
                )
            # Extract the `model.tar.gz` file into the `transformers-model` directory
            os.makedirs("./diffusers-model", exist_ok=True)
            with tarfile.open("model.tar.gz", "r:gz") as tar:
                tar.extractall(path="./diffusers-model")

        # If the `HF_MODEL_KWARGS` environment variable is set, then we parse its value into a dictionary
        model_kwargs = os.getenv("HF_MODEL_KWARGS", None)
        model_kwargs_dict: Dict[str, Any] = {}
        if model_kwargs is not None:
            try:
                model_kwargs_dict = eval(model_kwargs)
                self._logger.info(f"HF_MODEL_KWARGS value is {model_kwargs_dict}")
                # Since the device placement is in charge of the `TransformersPredictor`, we will pop those
                # keys from the `model_kwargs_dict` to avoid conflicts with the `pipeline` method.
                model_kwargs_dict.pop("device", None)
                model_kwargs_dict.pop("device_map", None)
            except Exception:
                self._logger.error(
                    f"Failed to parse `HF_MODEL_KWARGS` environment variable: {model_kwargs}"
                )

        # Set `torch_dtype` to `auto` is not set.
        if "torch_dtype" not in model_kwargs_dict:
            model_kwargs_dict["torch_dtype"] = "auto"
        else:
            model_kwargs_dict["torch_dtype"] = getattr(
                torch, model_kwargs_dict["torch_dtype"]
            )

        # If the `HF_TASK` environment variable is set, then we use it to load the `pipeline` with the
        # specified task, otherwise we load the `pipeline` with the default task, which is inferred from
        # the model's architecture.
        task = os.getenv("HF_TASK", "text-to-image")
        if task not in PIPELINE_TASKS:
            error_msg = (
                f"The `HF_TASK` environment variable value '{task}' is not supported! "
                f"Supported values are: {PIPELINE_TASKS}"
            )
            self._logger.error(error_msg)
            raise ValueError(error_msg)

        self._logger.info(f"HF_TASK value is {task}")

        self._logger.info("Loading `pipeline` using `device_map='auto'`")
        self._pipeline = PIPELINE_TASKS[task].from_pretrained(  # type: ignore
            pretrained_model_or_path=model_path,
            device_map="auto",
            **model_kwargs_dict,
        )
        self._logger.info(
            f"`pipeline` successfully loaded and running on device={self._pipeline.device}"
        )

    def _upload_image_to_gcs(self, image: Image) -> str:
        """Uploads the given `image` to Google Cloud Storage and returns the public URL.

        Args:
            image: is the image to be uploaded to Google Cloud Storage.

        Returns:
            The public URL to the uploaded image in Google Cloud Storage.
        """
        client = Client()
        bucket = client.get_bucket("vertex-ai-huggingface-inference-toolkit")

        with BytesIO() as output:
            image.save(output, format="JPEG")  # type: ignore
            contents = output.getvalue()

        blob = bucket.blob(f"diffusers/{uuid4()}.jpg")
        blob.upload_from_string(contents, content_type="image/jpeg")
        return blob.public_url  # type: ignore

    def predict(self, instances: Dict[str, Any]) -> Dict[str, Any]:
        """Runs the inference on top of the loaded `pipeline` with the given `instances`.

        Args:
            instances: is the dictionary containing the instances to be predicted, which can either
                be a dictionary with `instances` as the key and the value being a list of dicts, or
                directly a single instance with the expected keys by `pipeline`.

        Returns:
            The dictionary containing the predictions for the given `instances`.
        """

        # NOTE: the standard `predict` method assumes that the `instances` is a dictionary with the key
        # `instances` that contains the actual instances to be predicted, so we need to check whether
        # the `instances` is a dictionary or a list, and if it's a dictionary, then we need to extract
        # the `instances` from it.
        if "instances" in instances:
            instances = instances["instances"]

        image = self._pipeline(**instances).images[0]  # type: ignore
        return {"image_url": self._upload_image_to_gcs(image)}
