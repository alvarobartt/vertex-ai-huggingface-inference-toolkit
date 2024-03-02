import warnings
from typing import Dict, List, Literal, Optional

from google.cloud import aiplatform

from vertex_ai_huggingface_inference_toolkit.docker import build_docker_image
from vertex_ai_huggingface_inference_toolkit.google_cloud.storage import (
    upload_directory_to_gcs,
)
from vertex_ai_huggingface_inference_toolkit.huggingface import download_files_from_hub


class TransformersModel:
    # Steps:
    # 1. Download the model from the Hugging Face Hub (only the required files)
    # 2. Upload the model to Google Cloud Storage
    # 3. Build the Docker image (could be built already)
    # 4. Push the Docker image to Google Container Registry
    # 4. Register the model in Vertex AI

    def __init__(
        self,
        model_name_or_path: Optional[str] = None,
        model_bucket_uri: Optional[str] = None,
        framework: Literal["torch", "tensorflow", "flax"] = "torch",
        framework_version: Optional[str] = None,
        transformers_version: Optional[str] = None,
        python_version: Optional[str] = None,
        cuda_version: Optional[str] = None,
        custom_image_uri: Optional[str] = None,
        extra_requirements: Optional[List[str]] = None,
        environment_variables: Optional[Dict[str, str]] = None,
    ) -> None:
        if model_name_or_path is None and model_bucket_uri is None:
            raise ValueError(
                "You need to provide either `model_name_or_path` or `model_bucket_uri`"
            )

        if model_name_or_path is not None and model_bucket_uri is not None:
            raise ValueError(
                "You can't provide both `model_name_or_path` and `model_bucket_uri`"
            )

        if model_bucket_uri is not None:
            self.model_bucket_uri = model_bucket_uri

        if model_name_or_path is not None:
            _local_dir = download_files_from_hub(
                repo_id=model_name_or_path, framework=framework
            )
            self.model_bucket_uri = upload_directory_to_gcs(local_dir=_local_dir)

        # TODO: Push the local Docker image to the Artifact Registry
        self.image = custom_image_uri or build_docker_image(
            python_version=python_version or "3.10",
            framework=framework or "torch",
            framework_version=framework_version or "2.1.0",
            transformers_version=transformers_version or "4.38.2",
            cuda_version=cuda_version,
            extra_requirements=extra_requirements,
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
        # aiplatform.init(project=project_id, location=region)
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

    def deploy(self) -> None:
        pass
