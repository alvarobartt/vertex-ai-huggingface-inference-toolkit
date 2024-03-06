import os
import re
import tarfile
from typing import Any, Dict, Optional

from google.cloud.aiplatform.prediction.predictor import Predictor
from google.cloud.aiplatform.utils import prediction_utils
from transformers import pipeline

from vertex_ai_huggingface_inference_toolkit.logging import get_logger
from vertex_ai_huggingface_inference_toolkit.utils import get_device


class TransformersPredictor(Predictor):
    def __init__(self) -> None:
        self._logger = get_logger("vertex-ai-huggingface-inference-toolkit")

    def load(self, artifacts_uri: Optional[str] = None) -> None:
        """Loads the preprocessor and model artifacts."""
        if artifacts_uri is not None:
            self._logger.info(
                f"Downloading artifacts from `artifacts_uri='{artifacts_uri}'`"
            )
            prediction_utils.download_model_artifacts(artifacts_uri)
            self._logger.info("Artifacts successfully downloaded!")

        hub_id = os.getenv("HF_HUB_ID", None)
        file_exists = os.path.exists("model.tar.gz") and os.path.isfile("model.tar.gz")
        if not file_exists and hub_id is None:
            error_msg = "Neither the environment variable `HF_HUB_ID` nor the file `model.tar.gz` exist!"
            self._logger.error(error_msg)
            raise RuntimeError(error_msg)

        model_path = "./transformers-model" if file_exists else hub_id
        if file_exists:
            if hub_id:
                self._logger.warn(
                    f"Since both the provided `artifacts_uri={artifacts_uri}` and the environment"
                    f" variable `HF_HUB_ID={hub_id}` are set, the `artifacts_uri` will be used as"
                    " it has priority over the `HF_HUB_ID` environment variable."
                )
            os.makedirs("./transformers-model", exist_ok=True)
            with tarfile.open("model.tar.gz", "r:gz") as tar:
                tar.extractall(path="./transformers-model")

        model_kwargs = os.getenv("HF_MODEL_KWARGS", None)
        model_kwargs_dict: Dict[str, Any] = {}
        if model_kwargs is not None:
            try:
                model_kwargs_dict = eval(model_kwargs)
                self._logger.info(f"HF_MODEL_KWARGS value is {model_kwargs_dict}")
                model_kwargs_dict.pop("device", None)
                model_kwargs_dict.pop("device_map", None)
            except Exception:
                self._logger.error(
                    f"Failed to parse `HF_MODEL_KWARGS` environment variable: {model_kwargs}"
                )

        task = os.getenv("HF_TASK", "")
        if task != "":
            self._logger.info(f"HF_TASK value is {task}")

        try:
            self._logger.info("Loading `pipeline` using `device_map='auto'`")
            self._pipeline = pipeline(
                task,
                model=model_path,
                device_map="auto",
                **model_kwargs_dict,
            )
        except ValueError as ve:
            self._logger.error(
                f"Failed to load `pipeline` using `device_map='auto'` failed with exception: {ve}"
            )
            # Some models like `DebertaV2ForSequenceClassification` do not support `device_map='auto'`
            pattern = re.compile(r"[a-zA-Z0-9]+ does not support `device_map='auto'`")
            if not pattern.search(str(ve)):
                raise ve

            device = get_device()
            self._logger.info(f"Loading `pipeline` using `device='{device}'` instead!")
            self._pipeline = pipeline(
                task,
                model=model_path,
                device=device,
                **model_kwargs_dict,
            )

        self._logger.info(
            f"`pipeline` successfully loaded and running on device={self._pipeline.device}"
        )

    def predict(self, instances: Dict[str, Any]) -> Dict[str, Any]:
        return self._pipeline(**instances)  # type: ignore
