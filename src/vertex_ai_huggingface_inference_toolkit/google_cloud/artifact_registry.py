import warnings

from google.api_core import exceptions
from google.cloud.artifactregistry_v1beta2 import (
    ArtifactRegistryClient,
    CreateRepositoryRequest,
    GetRepositoryRequest,
)
from google.cloud.artifactregistry_v1beta2.types.repository import Repository


def create_repository_in_artifact_registry(
    project_id: str, location: str, name: str, format: str = "DOCKER"
) -> str:
    client = ArtifactRegistryClient()
    parent = f"projects/{project_id}/locations/{location}"

    request = CreateRepositoryRequest(
        parent=parent,
        repository_id=name,
        repository=Repository(format_=format),
    )
    try:
        response = client.create_repository(request=request)
        return response.result().name  # type: ignore
    except exceptions.AlreadyExists as ae:
        repository_path = f"{parent}/repositories/{name}"

        request = GetRepositoryRequest(name=repository_path)

        response = client.get_repository(request=request)
        if response.format_ != format:
            raise RuntimeError(
                f"`repository={name}` already exists, but it's not a Docker repository"
                f" but a `{response.format_}` one, please make sure to specify another"
                " `name` instead."
            ) from ae

        warnings.warn(
            f"Skipping `repository={name}` creation since it already exists!",
            stacklevel=1,
        )
        return f"{parent}/repositories/{name}"
    except Exception as e:
        raise RuntimeError(
            f"`repository={name}` couldn't be created with exception `{e}`"
        ) from e
