import os
import re
import tarfile
from typing import Any, Dict

from google.cloud.aiplatform.prediction.predictor import Predictor
from google.cloud.aiplatform.utils import prediction_utils
from transformers import pipeline

from vertex_ai_huggingface_inference_toolkit.logging import get_logger
from vertex_ai_huggingface_inference_toolkit.utils import get_device


class TransformersPredictor(Predictor):
    def __init__(self) -> None:
        self._logger = get_logger("vertex-ai-huggingface-inference-toolkit")

    def load(self, artifacts_uri: str) -> None:
        """Loads the preprocessor and model artifacts."""
        print(f"Downloading artifacts from {artifacts_uri}")
        self._logger.info(f"Downloading artifacts from {artifacts_uri}")
        prediction_utils.download_model_artifacts(artifacts_uri)
        self._logger.info("Artifacts successfully downloaded!")

        os.makedirs("./transformers-model", exist_ok=True)
        with tarfile.open("model.tar.gz", "r:gz") as tar:
            tar.extractall(path="./transformers-model")

        self._logger.info(f"HF_TASK value is {os.getenv('HF_TASK')}")
        try:
            self._pipeline = pipeline(
                os.getenv("HF_TASK", ""),
                model="./transformers-model",
                device_map="auto",
            )
        except ValueError as ve:
            self._logger.error(f"Error while loading `pipeline`: {ve}")
            # Some models like `DebertaV2ForSequenceClassification` do not support `device_map='auto'`
            pattern = re.compile(r"[a-zA-Z0-9]+ does not support `device_map='auto'`")
            print(f"Pattern {pattern} search results are {pattern.search(str(ve))}")
            if not pattern.search(str(ve)):
                self._logger.info(f"Pattern {pattern} did not match the error message")

            self._pipeline = pipeline(
                os.getenv("HF_TASK", ""),
                model="./transformers-model",
                device=get_device(),
            )

        self._logger.info(
            f"`pipeline` successfully loaded using device={self._pipeline.device}"
        )

    def predict(self, instances: Dict[str, Any]) -> Dict[str, Any]:
        return self._pipeline(**instances)  # type: ignore
