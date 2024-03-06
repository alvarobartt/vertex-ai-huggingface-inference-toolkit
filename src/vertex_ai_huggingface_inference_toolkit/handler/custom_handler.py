from typing import Optional, Type

from google.cloud.aiplatform.prediction.handler import PredictionHandler

from vertex_ai_huggingface_inference_toolkit.predictors.transformers import (
    TransformersPredictor,
)


class CustomPredictionHandler(PredictionHandler):
    """Custom class that overrides the default `PredictionHandler` provided within
    `google-cloud-aiplatform` to be able to handle the cases where the `artifacts_uri`
    is None which could be translated into being able to run the server locally without
    the need of an artifact URI as the model is pulled from the Hugging Face Hub.
    """

    def __init__(
        self,
        artifacts_uri: Optional[str] = None,
        predictor: Optional[Type[TransformersPredictor]] = None,
    ) -> None:
        """Initializes the `TransformersPredictor` provided via the `predictor`
        arg, since the default `PredictionHandler` won't allow an empty `artifacts_uri`.

        Note:
            The `predictor` is mandatory, but it's been set as optional since the
            `CprModelServer` intializes the `PredictionHandler` without using `kwargs`
            so that the first arg is set to be `artifacts_uri` as it's provided without
            keyword.

        Args:
            artifacts_uri: is the Google Cloud Storage URI to the artifact to serve, which
                will ideally be the directory where the model is stored in Google Cloud Storage.
            predictor: is the `TransformersPredictor` class subclassing `PredictionHandler`,
                that implements the logic for the inference on top of the model.
        """

        self._predictor = predictor()  # type: ignore
        self._predictor.load(artifacts_uri)
