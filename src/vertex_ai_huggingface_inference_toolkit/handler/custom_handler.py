from typing import Optional, Type

from google.cloud.aiplatform.prediction.handler import PredictionHandler

from vertex_ai_huggingface_inference_toolkit.predictors.transformers import (
    TransformersPredictor,
)


class CustomPredictionHandler(PredictionHandler):
    def __init__(
        self,
        artifacts_uri: Optional[str] = None,
        predictor: Optional[Type[TransformersPredictor]] = None,
    ) -> None:
        self._predictor = predictor()  # type: ignore
        self._predictor.load(artifacts_uri)
