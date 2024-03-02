import warnings
from typing import Dict, List, Literal, Optional

from google.cloud import aiplatform


class TransformersModel:
    # Steps:
    # 1. Download the model from the Hugging Face Hub (only the required files)
    # 2a. Upload the model to Google Cloud Storage
    # 2b. Build the Docker image (could be built already)
    # 3. Register the model in Vertex AI
    # 4. Deploy the model in Vertex AI

    def __init__(
        self,
        model_name_or_path: Optional[str] = None,
        model_bucket_uri: Optional[str] = None,
        framework: Literal["torch", "jax", "tensorflow"] = "torch",
        framework_version: Optional[str] = None,
        transformers_version: Optional[str] = None,
        python_version: Optional[str] = None,
        cuda_version: Optional[str] = None,
        custom_image_uri: Optional[str] = None,
        extra_requirements: Optional[List[str]] = None,
        environment_variables: Optional[Dict[str, str]] = None,
    ) -> None:
        # aiplatform.init(project=project_id, location=region)
        if model_name_or_path is None and model_bucket_uri is None:
            raise ValueError(
                "You need to provide either `model_name_or_path` or `model_bucket_uri`"
            )

        if model_name_or_path is not None and model_bucket_uri is not None:
            raise ValueError(
                "You can't provide both `model_name_or_path` and `model_bucket_uri`"
            )

        if environment_variables is None:
            environment_variables = {}
        if "VERTEX_CPR_WEB_CONCURRENCY" not in environment_variables:
            warnings.warn(
                "Since the `VERTEX_CPR_WEB_CONCURRENCY` environment variable hasn't been set, it will default to 1, meaning that `uvicorn` will only run the model in one worker. If you prefer to run the model using more workers, feel free to provide a greater value for `VERTEX_CPR_WEB_CONCURRENCY`",
                stacklevel=1,
            )
            environment_variables["VERTEX_CPR_WEB_CONCURRENCY"] = "1"

        # https://github.com/googleapis/python-aiplatform/blob/63ad1bf9e365d2f10b91e2fd036e3b7d937336c0/google/cloud/aiplatform/models.py#L2974
        # self._model: aiplatform.Model = aiplatform.Model.upload(
        #     display_name=model_name_or_path.replace("/", "--"),  # type: ignore
        #     artifact_uri="gs://huggingface-cloud/bart-large-mnli",  # Should be created if not provided
        #     serving_container_image_uri="",  # framework, framework_version, cpu/gpu, cuda_version
        #     # if extra requirements then build an image installing those libraries on top of serving_container_image_uri
        #     serving_container_environment_variables=environment_variables,
        # )

    @classmethod
    def from_bucket(cls, bucket_name: str) -> "TransformersModel":  # type: ignore
        pass

    # from sagemaker.huggingface import HuggingFaceModel
    #
    # model = HuggingFaceModel(
    #     model_data=s3_uri,
    #     role=role,
    #     transformers_version="4.26",
    #     pytorch_version="1.13",
    #     py_version="py39",
    # )
    # def deploy(
    #     self,
    #     machine_type: Optional[str] = None,
    #     accelerator_type: Optional[str] = None,
    #     accelerator_count: int = 0,
    # ) -> None:
    #     return self._model.deploy(  # type: ignore
    #         machine_type=machine_type,
    #         accelerator_type=accelerator_type,
    #         accelerator_count=accelerator_count,
    #     )
