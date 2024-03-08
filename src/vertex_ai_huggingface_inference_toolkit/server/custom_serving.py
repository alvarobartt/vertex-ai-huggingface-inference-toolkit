import os

from google.cloud.aiplatform.prediction.model_server import CprModelServer


class CustomCprModelServer(CprModelServer):
    """Custom class that overrides the default `CprModelServer` provided within
    `google-cloud-aiplatform` to be able to run the inference server locally before
    going to Vertex AI, in order to better test and debug the potential issues within
    the `TransformersPredictor` used.
    """

    def __init__(self) -> None:
        """Sets the environment variables required by the `CprModelServer` so as to
        be able to run the server with minimal to no configuration, since only the `HF_HUB_ID`
        and, optionally, the `HF_TASK` need to be provided.
        """

        os.environ["HANDLER_MODULE"] = (
            "vertex_ai_huggingface_inference_toolkit.handler.custom_handler"
        )
        os.environ["HANDLER_CLASS"] = "CustomPredictionHandler"

        os.environ["PREDICTOR_MODULE"] = (
            "vertex_ai_huggingface_inference_toolkit.predictors.transformers"
        )
        os.environ["PREDICTOR_CLASS"] = "TransformersPredictor"

        os.environ["AIP_HTTP_PORT"] = "8080"
        os.environ["AIP_HEALTH_ROUTE"] = "/health"
        os.environ["AIP_PREDICT_ROUTE"] = "/predict"

        super().__init__()


if __name__ == "__main__":
    """
    Example:
        >>> export HF_HUB_ID="cardiffnlp/twitter-roberta-base-sentiment-latest"
        >>> export HF_TASK="text-classification"
        >>> python vertex_ai_huggingface_inference_toolkit/server/custom_serving.py
    """
    import uvicorn

    uvicorn.run(
        "vertex_ai_huggingface_inference_toolkit.server.custom_serving:CustomCprModelServer",
        host="0.0.0.0",
        port=8080,
        factory=True,
    )
