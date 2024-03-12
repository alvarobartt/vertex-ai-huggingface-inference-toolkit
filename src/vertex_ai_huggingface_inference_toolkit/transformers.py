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
    """Class that manages the whole lifecycle of a Hugging Face model either from the Hub
    or from an existing Google Cloud Storage bucket to be deployed to Google Cloud Vertex AI
    as an endpoint, running a Custom Prediction Routine (CPR) on top of a Hugging Face optimized
    Docker image pushed to Google Cloud Artifact Registry.

    This class is responsible for:
    - Downloading the model from the Hub if `model_name_or_path` is provided.
    - Uploading the model to Google Cloud Storage if `model_name_or_path` is provided.
    - Building a Docker image with the prediction code, handler and the required dependencies if `image_uri` not provided.
    - Pushing the Docker image to Google Cloud Artifact Registry if `image_uri` not provided.
    - Registering the model in Google Cloud Vertex AI.
    - Deploying the model as an endpoint with the provided environment variables.

    Note:
        This class is intended to be a high-level abstraction to simplify the process of deploying
        models from the Hugging Face Hub to Google Cloud Vertex AI, and is built on top of `google-cloud-aiplatform`
        and the rest of the required Google Cloud Python SDKs.
    """

    def __init__(
        self,
        # Google Cloud
        project_id: Optional[str] = None,
        location: Optional[str] = None,
        # Google Cloud Storage
        model_name_or_path: Optional[str] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        model_target_bucket: str = "vertex-ai-huggingface-inference-toolkit",
        # Exclusive arg for Google Cloud Storage
        model_bucket_uri: Optional[str] = None,
        # Google Cloud Artifact Registry (Docker)
        framework: Literal["torch", "tensorflow", "flax"] = "torch",
        framework_version: Optional[str] = None,
        transformers_version: str = "4.38.2",
        python_version: str = "3.10",
        cuda_version: str = "12.3.0",
        ubuntu_version: str = "22.04",
        extra_requirements: Optional[List[str]] = None,
        image_target_repository: str = "vertex-ai-huggingface-inference-toolkit",
        # Exclusive arg for Google Cloud Artifact Registry
        image_uri: Optional[str] = None,
        # Google Cloud Vertex AI
        environment_variables: Optional[Dict[str, str]] = None,
    ) -> None:
        """Initializes the `TransformersModel` class, setting up the required attributes to
        deploy a model from the Hugging Face Hub to Google Cloud Vertex AI.

        Args:
            project_id: is either the name or the identifier of the project in Google Cloud.
            location: is the identifier of the region and zone where the resources will be created.
            model_name_or_path: is the name of the model to be downloaded from the Hugging Face Hub.
            model_kwargs: is the dictionary of keyword arguments to be passed to the model's `from_pretrained` method.
            model_target_bucket: is the name of the bucket in Google Cloud Storage where the model will be uploaded to.
            model_bucket_uri: is the URI to the model tar.gz file in Google Cloud Storage.
            framework: is the framework to be used to build the Docker image, e.g. `torch`, `tensorflow`, `flax`.
            framework_version: is the version of the framework to be used to build the Docker image.
            transformers_version: is the version of the `transformers` library to be used to build the Docker image.
            python_version: is the version of Python to be used to build the Docker image.
            cuda_version: is the version of CUDA to be used to build the Docker image.
            ubuntu_version: is the version of Ubuntu to be used to build the Docker image.
            extra_requirements: is the list of extra requirements to be installed in the Docker image.
            image_target_repository: is the name of the repository in Google Cloud Artifact Registry where the Docker image will be pushed to.
            image_uri: is the URI to the Docker image in Google Cloud Artifact Registry.
            environment_variables: is the dictionary of environment variables to be set in the Docker image.

        Raises:
            ValueError: if neither `model_name_or_path` nor `model_bucket_uri` is provided.
            ValueError: if both `model_name_or_path` and `model_bucket_uri` are provided.

        Examples:
            >>> from vertex_ai_huggingface_inference_toolkit import TransformersModel
            >>> model = TransformersModel(
            ...     model_name_or_path="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
            ...     project_id="my-gcp-project",
            ...     location="us-central1",
            ...     environment_variables={
            ...         "HF_TASK": "zero-shot-classification",
            ...     },
            ... )
            >>> model.deploy(
            ...     machine_type="n1-standard-8",
            ...     accelerator_type="NVIDIA_TESLA_T4",
            ...     accelerator_count=1,
            ... )
        """

        if model_name_or_path is None and model_bucket_uri is None:
            raise ValueError(
                "You need to provide either `model_name_or_path` or `model_bucket_uri`"
            )

        if model_name_or_path is not None and model_bucket_uri is not None:
            raise ValueError(
                "You can't provide both `model_name_or_path` and `model_bucket_uri`"
            )

        # If the `project_id` is not provided, then we retrieve the default one i.e. the one
        # set as default via `gcloud config set project <project-id>`
        if project_id is None:
            # https://google-auth.readthedocs.io/en/master/reference/google.auth.html
            _, project_id = default()
        self.project_id = project_id

        # If the `location` is not provided, then we default to `us-central1` as that's the
        # Google Cloud default region.
        if location is None:
            warnings.warn(
                "`location` has not been provided, so `location=us-central1` will be used"
                " instead, as that's the Google Cloud default region.",
                stacklevel=1,
            )
            location = "us-central1"
        self.location = location

        # If the `model_bucket_uri` has been provided, then we will need to check whether it's
        # the name to the model on the Hugging Face Hub, or the path to the model in the local
        # storage.
        if model_name_or_path is not None:
            if os.path.exists(model_name_or_path):
                # If the `model_name_or_path` is a local path, then we will need to compress everything
                # within the directory into `model.tar.gz` and upload it to Google Cloud Storage.
                _local_dir = model_name_or_path
                _tar_gz_path = Path(_local_dir) / "model.tar.gz"
            else:
                # If the `model_name_or_path` is a model from the Hugging Face Hub, then we will need
                # to download it and upload it to Google Cloud Storage.
                _local_dir = download_files_from_hub(
                    repo_id=model_name_or_path, framework=framework
                )

                _cache_path = CACHE_PATH / model_name_or_path.replace("/", "--")
                if not _cache_path.exists():
                    _cache_path.mkdir(parents=True, exist_ok=True)

                _tar_gz_path = _cache_path / "model.tar.gz"

            if _tar_gz_path.exists():
                _tar_gz_path.unlink()

            # Then, we compress the directory into a `model.tar.gz` file
            with tarfile.open(_tar_gz_path, "w:gz") as tf:
                for root, _, files in os.walk(_local_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        if os.path.islink(file_path):
                            file_path = os.path.realpath(file_path)
                        tf.add(file_path, arcname=file)

            # Finally, we upload the `model.tar.gz` file to Google Cloud Storage
            model_bucket_uri = upload_file_to_gcs(
                project_id=self.project_id,  # type: ignore
                location=self.location,
                local_path=_tar_gz_path.as_posix(),
                bucket_name=model_target_bucket,
                remote_path=f"{model_name_or_path.replace('/', '--')}/model.tar.gz",
            )
        self.model_bucket_uri = model_bucket_uri.replace("/model.tar.gz", "")  # type: ignore

        # If the `image_uri` has not been provided, then we will need to build the Docker image
        # and push it to Google Cloud Artifact Registry.
        if image_uri is None:
            # Depending on the `framework`, if `framework_version` has not been provided, then we
            # select the latest or stable version of the selected `framework`, but it's not ideal.
            if framework_version is None:
                if framework == "torch":
                    framework_version = "2.1.0"
                elif framework == "tensorflow":
                    framework_version = "2.15.0"
                elif framework == "flax":
                    framework_version = "0.8.0"

            # Then we build the Docker image with the provided args, that will be replaced within the
            # Dockerfile to build, as those are internally defined as `BUILD_ARGS`
            _image = build_docker_image(
                python_version=python_version,
                framework=framework,
                framework_version=framework_version,
                transformers_version=transformers_version,
                cuda_version=cuda_version,
                ubuntu_version=ubuntu_version,
                extra_requirements=extra_requirements,
            )
            # Once the Docker image has been built, then we push it to Google Cloud Artifact Registry, but first
            # we need to create a new repository if it doesn't exist.
            create_repository_in_artifact_registry(
                project_id=self.project_id,  # type: ignore
                location=self.location,
                name=image_target_repository,
                format="DOCKER",
            )
            image_uri = configure_docker_and_push_image(
                project_id=self.project_id,  # type: ignore
                location=self.location,
                repository=image_target_repository,
                image_with_tag=_image,
            )
        self.image_uri = image_uri

        # If the `environment_variables` are not set or any of the expected ones it not set, then we
        # will set them to their default values.
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

        # If the `model_name_or_path` has been provided, then we will use it as the `display_name` of the
        # model in Google Cloud Vertex AI Model Registry, otherwise we will use the last part of the
        # `model_bucket_uri` as the `display_name`.
        display_name = (
            model_name_or_path.replace("/", "--")
            if model_name_or_path is not None
            else model_bucket_uri.split("/")[-1]  # type: ignore
        )

        # If the `HF_TASK` environment variable has not been set in `environmnent_variables`, then we will
        # warn the user that it hasn't been set, and set the `instance_schema_uri` and `prediction_schema_uri`
        # to `None`.
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
            # Since only the `text-generation` and `zero-shot-classification` tasks have the
            # `instance_schema_uri` and `prediction_schema_uri` defined, then we will only
            # upload the schemas if the `task` is one of those.
            instance_schema_uri = upload_file_to_gcs(
                project_id=self.project_id,  # type: ignore
                location=self.location,
                local_path=f"{_path}/input.yaml",
                bucket_name=model_target_bucket,
                remote_path=f"{display_name}/{task}/input.yaml",
            )
            prediction_schema_uri = upload_file_to_gcs(
                project_id=self.project_id,  # type: ignore
                location=self.location,
                local_path=f"{_path}/output.yaml",
                bucket_name=model_target_bucket,
                remote_path=f"{display_name}/{task}/output.yaml",
            )

        # Finally, we upload the model to Google Cloud Vertex AI Model Registry, providing the
        # `model_bucket_uri` in Google Cloud Storage, the `image_uri` in Google Cloud Artifact Registry,
        # and the `environment_variables` to be set in the Docker image at runtime.
        # `aiplatform.Model.upload` reference:
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
        """Returns the list of deployed `Endpoint` resources, if any."""
        return self._endpoints

    def deploy(
        self,
        machine_type: Optional[str] = None,
        min_replica_count: int = 1,
        max_replica_count: int = 1,
        accelerator_type: Optional[str] = None,
        accelerator_count: Optional[int] = None,
    ) -> None:
        """Deploys the model to a `Endpoint` resource, with the given `machine_type` and
        `accelerator_type` and `accelerator_count` if provided. The `min_replica_count` and
        `max_replica_count` are set to 1 by default, but can be changed if needed.

        Args:
            machine_type: is the type of machine to use for the deployment, e.g. `n1-standard-8`.
            min_replica_count: is the minimum number of replicas to use for the deployment.
            max_replica_count: is the maximum number of replicas to use for the deployment.
            accelerator_type: is the type of accelerator to use for the deployment, e.g. `NVIDIA_TESLA_T4`.
            accelerator_count: is the number of accelerators to use for the deployment, e.g. `1`.

        References:
            - https://github.com/googleapis/python-aiplatform/blob/63ad1bf9e365d2f10b91e2fd036e3b7d937336c0/google/cloud/aiplatform/models.py#L3431
        """

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
        """Undeploys the model from the `Endpoint` resources, if any. Finally, deletes the model
        from Vertex AI Model Registry."""

        for endpoint in self._endpoints:
            endpoint.delete(force=True, sync=False)
        self._endpoints = []
        self._model.delete(sync=False)
