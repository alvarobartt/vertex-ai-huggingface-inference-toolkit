from typing import Any, Dict, List, Literal, Optional

from vertex_ai_huggingface_inference_toolkit.model import Model


class TransformersModel(Model):
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
            ...     project_id="my-gcp-project",
            ...     location="us-central1",
            ...     model_name_or_path="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
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

        super().__init__(
            project_id=project_id,
            location=location,
            model_name_or_path=model_name_or_path,
            model_kwargs=model_kwargs,
            model_target_bucket=model_target_bucket,
            model_bucket_uri=model_bucket_uri,
            framework=framework,
            framework_version=framework_version,
            huggingface_framework="transformers",  # type: ignore
            huggingface_framework_version=transformers_version,
            python_version=python_version,
            cuda_version=cuda_version,
            ubuntu_version=ubuntu_version,
            extra_requirements=extra_requirements,
            image_target_repository=image_target_repository,
            image_uri=image_uri,
            environment_variables=environment_variables,
        )
