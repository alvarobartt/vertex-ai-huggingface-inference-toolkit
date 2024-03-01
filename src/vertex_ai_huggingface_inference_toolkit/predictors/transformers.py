import os
import tarfile
from typing import Any, Dict

from google.cloud.aiplatform.prediction.predictor import Predictor
from google.cloud.aiplatform.utils import prediction_utils
from transformers import pipeline

from vertex_ai_huggingface_inference_toolkit.logging import get_logger


class TransformersPredictor(Predictor):
    def __init__(self) -> None:
        self._logger = get_logger(__name__)

    def load(self, artifacts_uri: str) -> None:
        """Loads the preprocessor and model artifacts."""
        self._logger.debug(f"Downloading artifacts from {artifacts_uri}")
        prediction_utils.download_model_artifacts(artifacts_uri)
        self._logger.debug("Artifacts successfully downloaded!")

        os.makedirs("./model", exist_ok=True)
        with tarfile.open("model.tar.gz", "r:gz") as tar:
            tar.extractall(path="./model")

        self._logger.debug(f"HF_TASK value is {os.getenv('HF_TASK')}")
        self._pipeline = pipeline(
            os.getenv("HF_TASK", ""), model="./model", device_map="auto"
        )
        self._logger.debug(
            f"`pipeline` successfully loaded using device={self._pipeline.device}"
        )

    def predict(self, instances: Dict[str, Any]) -> Dict[str, Any]:
        return self._pipeline(**instances)  # type: ignore
