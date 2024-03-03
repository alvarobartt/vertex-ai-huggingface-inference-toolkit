import warnings
from typing import Dict, List, Literal, Optional

from google.auth import default
from google.cloud import aiplatform

from vertex_ai_huggingface_inference_toolkit.docker import (
    build_docker_image,
    configure_docker_and_push_image,
)
from vertex_ai_huggingface_inference_toolkit.google_cloud.artifact_registry import (
    create_repository_in_artifact_registry,
)
from vertex_ai_huggingface_inference_toolkit.google_cloud.storage import (
    upload_directory_to_gcs,
)
from vertex_ai_huggingface_inference_toolkit.huggingface import download_files_from_hub


class TransformersModel:
    def __init__(
        self,
        project_id: Optional[str] = None,
        location: Optional[str] = None,
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

        if project_id is None:
            # https://google-auth.readthedocs.io/en/master/reference/google.auth.html
            _, project_id = default()
        self.project_id = project_id

        if location is None:
            warnings.warn(
                "`location` has not been provided, so `location=us-central1` will be used"
                " instead, as that's the Google Cloud default region.",
                stacklevel=1,
            )
            location = "us-central1"
        self.location = location

        if model_bucket_uri is not None:
            self.model_bucket_uri = model_bucket_uri

        if model_name_or_path is not None:
            _local_dir = download_files_from_hub(
                repo_id=model_name_or_path, framework=framework
            )
            self.model_bucket_uri = upload_directory_to_gcs(
                project_id=self.project_id,  # type: ignore
                location=self.location,
                local_dir=_local_dir,  # type: ignore
            )

        if custom_image_uri is None:
            _image = build_docker_image(
                python_version=python_version or "3.10",
                framework=framework or "torch",
                framework_version=framework_version or "2.1.0",
                transformers_version=transformers_version or "4.38.2",
                cuda_version=cuda_version,
                extra_requirements=extra_requirements,
            )
            create_repository_in_artifact_registry(
                project_id=self.project_id,  # type: ignore
                location=self.location,
                name="vertex-ai-huggingface-inference-toolkit",
                format="DOCKER",
            )
            custom_image_uri = configure_docker_and_push_image(
                project_id=self.project_id,  # type: ignore
                location=self.location,
                repository="vertex-ai-huggingface-inference-toolkit",
                image_with_tag=_image,
            )
        self.image_uri = custom_image_uri

        if environment_variables is None:
            environment_variables = {}
        if "VERTEX_CPR_WEB_CONCURRENCY" not in environment_variables:
            warnings.warn(
                "Since the `VERTEX_CPR_WEB_CONCURRENCY` environment variable hasn't been set, it will default to 1, meaning that `uvicorn` will only run the model in one worker. If you prefer to run the model using more workers, feel free to provide a greater value for `VERTEX_CPR_WEB_CONCURRENCY`",
                stacklevel=1,
            )
            environment_variables["VERTEX_CPR_WEB_CONCURRENCY"] = "1"

        display_name = (
            model_name_or_path.replace("/", "--")
            if model_name_or_path is not None
            else model_bucket_uri.split("/")[-1]  # type: ignore
        )

        # https://github.com/googleapis/python-aiplatform/blob/63ad1bf9e365d2f10b91e2fd036e3b7d937336c0/google/cloud/aiplatform/models.py#L2974
        self._model: aiplatform.Model = aiplatform.Model.upload(
            display_name=display_name,
            project=self.project_id,
            location=self.location,
            artifact_uri=self.model_bucket_uri,
            serving_container_image_uri=self.image_uri,
            serving_container_environment_variables=environment_variables,
        )

    @classmethod
    def from_bucket(cls, bucket_name: str) -> "TransformersModel":  # type: ignore
        pass

    def deploy(
        self,
        machine_type: Optional[str] = None,
        min_replica_count: int = 1,
        max_replica_count: int = 1,
        accelerator_type: Optional[str] = None,
        accelerator_count: Optional[int] = None,
    ) -> None:
        # https://github.com/googleapis/python-aiplatform/blob/63ad1bf9e365d2f10b91e2fd036e3b7d937336c0/google/cloud/aiplatform/models.py#L3431
        self._model.deploy(
            machine_type=machine_type,
            min_replica_count=min_replica_count,
            max_replica_count=max_replica_count,
            accelerator_type=accelerator_type,
            accelerator_count=accelerator_count,
        )
