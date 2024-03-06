import os

from google.cloud.aiplatform.prediction.model_server import CprModelServer


class CustomCprModelServer(CprModelServer):
    def __init__(self) -> None:
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
    import uvicorn

    uvicorn.run(
        "vertex_ai_huggingface_inference_toolkit.server.custom_serving:CustomCprModelServer",
        host="0.0.0.0",
        port=8080,
        factory=True,
    )
