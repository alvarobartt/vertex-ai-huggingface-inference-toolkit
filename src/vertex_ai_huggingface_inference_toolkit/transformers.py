import os
import sys
import tarfile
import warnings
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

if sys.version_info < (3, 9):
    import importlib_resources
else:
    import importlib.resources as importlib_resources

from google.auth import default
from google.cloud import aiplatform

from vertex_ai_huggingface_inference_toolkit.utils import (
    CACHE_PATH,
    build_docker_image,
    configure_docker_and_push_image,
    create_repository_in_artifact_registry,
    download_files_from_hub,
    upload_file_to_gcs,
)


class TransformersModel:
    def __init__(
        self,
        project_id: Optional[str] = None,
        location: Optional[str] = None,
        model_name_or_path: Optional[str] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        model_bucket_uri: Optional[str] = None,
        framework: Literal["torch", "tensorflow", "flax"] = "torch",
        framework_version: Optional[str] = None,
        transformers_version: Optional[str] = None,
        python_version: Optional[str] = None,
        cuda_version: Optional[str] = None,
        ubuntu_version: Optional[str] = None,
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

        if model_name_or_path is not None:
            if os.path.exists(model_name_or_path):
                _local_dir = model_name_or_path
                _tar_gz_path = Path(_local_dir) / "model.tar.gz"
            else:
                _local_dir = download_files_from_hub(
                    repo_id=model_name_or_path, framework=framework
                )

                _cache_path = CACHE_PATH / model_name_or_path.replace("/", "--")
                if not _cache_path.exists():
                    _cache_path.mkdir(parents=True, exist_ok=True)

                _tar_gz_path = _cache_path / "model.tar.gz"

            if _tar_gz_path.exists():
                _tar_gz_path.unlink()

            with tarfile.open(_tar_gz_path, "w:gz") as tf:
                for root, _, files in os.walk(_local_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        if os.path.islink(file_path):
                            file_path = os.path.realpath(file_path)
                        tf.add(file_path, arcname=file)

            model_bucket_uri = upload_file_to_gcs(
                project_id=self.project_id,  # type: ignore
                location=self.location,
                local_path=_tar_gz_path.as_posix(),
                remote_path=f"{model_name_or_path.replace('/', '--')}/model.tar.gz",
            )
        self.model_bucket_uri = model_bucket_uri.replace("/model.tar.gz", "")  # type: ignore

        if custom_image_uri is None:
            _image = build_docker_image(
                python_version=python_version or "3.10",
                framework=framework or "torch",
                framework_version=framework_version or "2.1.0",
                transformers_version=transformers_version or "4.38.2",
                cuda_version=cuda_version,
                ubuntu_version=ubuntu_version,
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
        if model_kwargs is not None and "HF_MODEL_KWARGS" not in environment_variables:
            environment_variables["HF_MODEL_KWARGS"] = str(model_kwargs)
        if isinstance(environment_variables["HF_MODEL_KWARGS"], dict):
            environment_variables["HF_MODEL_KWARGS"] = str(
                environment_variables["HF_MODEL_KWARGS"]
            )
        if "VERTEX_CPR_WEB_CONCURRENCY" not in environment_variables:
            warnings.warn(
                "Since the `VERTEX_CPR_WEB_CONCURRENCY` environment variable hasn't been set,"
                " it will default to 1, meaning that `uvicorn` will only run the model in one"
                " worker. If you prefer to run the model using more workers, feel free to provide"
                " a greater value for `VERTEX_CPR_WEB_CONCURRENCY`",
                stacklevel=1,
            )
            environment_variables["VERTEX_CPR_WEB_CONCURRENCY"] = "1"

        display_name = (
            model_name_or_path.replace("/", "--")
            if model_name_or_path is not None
            else model_bucket_uri.split("/")[-1]  # type: ignore
        )

        task = environment_variables.get("HF_TASK", "")
        if task == "" or task not in ["text-generation", "zero-shot-classification"]:
            warnings.warn(
                "`HF_TASK` hasn't been set within the `environment_variables` dict, so the"
                " `task` will default to an empty string which may not be ideal. Additionally,"
                " the `HF_TASK` needs to be defined so that the `instance_schema_uri` and"
                " `predictions_schema_uri` can be generated automatically based on the `pipeline`"
                " definition.",
                stacklevel=1,
            )
            instance_schema_uri, prediction_schema_uri = None, None
        else:
            _path = str(
                importlib_resources.files("vertex_ai_huggingface_inference_toolkit")
                / "_internal"
                / "schemas"
                / task
            )
            instance_schema_uri = upload_file_to_gcs(
                project_id=self.project_id,  # type: ignore
                location=self.location,
                local_path=f"{_path}/input.yaml",
                remote_path=f"{display_name}/{task}/input.yaml",
            )
            prediction_schema_uri = upload_file_to_gcs(
                project_id=self.project_id,  # type: ignore
                location=self.location,
                local_path=f"{_path}/output.yaml",
                remote_path=f"{display_name}/{task}/output.yaml",
            )

        # https://github.com/googleapis/python-aiplatform/blob/63ad1bf9e365d2f10b91e2fd036e3b7d937336c0/google/cloud/aiplatform/models.py#L2974
        self._model: aiplatform.Model = aiplatform.Model.upload(  # type: ignore
            display_name=display_name,
            project=self.project_id,
            location=self.location,
            artifact_uri=self.model_bucket_uri,
            serving_container_image_uri=self.image_uri,
            serving_container_environment_variables=environment_variables,
            instance_schema_uri=instance_schema_uri,
            prediction_schema_uri=prediction_schema_uri,
        )
        self._endpoints: List[
            Union[aiplatform.Endpoint, aiplatform.PrivateEndpoint]
        ] = []

    @property
    def endpoints(
        self,
    ) -> Optional[List[Union[aiplatform.Endpoint, aiplatform.PrivateEndpoint]]]:
        return self._endpoints

    def deploy(
        self,
        machine_type: Optional[str] = None,
        min_replica_count: int = 1,
        max_replica_count: int = 1,
        accelerator_type: Optional[str] = None,
        accelerator_count: Optional[int] = None,
    ) -> None:
        # https://github.com/googleapis/python-aiplatform/blob/63ad1bf9e365d2f10b91e2fd036e3b7d937336c0/google/cloud/aiplatform/models.py#L3431
        self._endpoints.append(
            self._model.deploy(
                machine_type=machine_type,
                min_replica_count=min_replica_count,
                max_replica_count=max_replica_count,
                accelerator_type=accelerator_type,
                accelerator_count=accelerator_count,
            )
        )

    def undeploy(self) -> None:
        for endpoint in self._endpoints:
            endpoint.delete(force=True, sync=False)
        self._endpoints = []
        self._model.delete(sync=False)
