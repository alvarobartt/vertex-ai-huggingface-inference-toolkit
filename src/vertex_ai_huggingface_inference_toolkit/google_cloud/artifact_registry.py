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
    """Creates a new repository in Google Cloud Artifact Registry.

    Args:
        project_id: is either the name or the identifier of the project in Google Cloud.
        location: is the identifier of the region and zone where the repository will be created.
        name: is the name of the repository to be created.
        format: is the format of the repository to be created, which by default is `DOCKER`.

    Returns:
        The path to the repository that has been created in Google Artifact Registry.

    Raises:
        RuntimeError: if the repository already exists but its `format` doesn't match
            the specifiec via `format` arg.
    """

    client = ArtifactRegistryClient()
    parent = f"projects/{project_id}/locations/{location}"

    # Creates the request payload to create a repository in Google Artifact Registry
    # with the specified `name` and `format`, under the specified `project_id` and `location`
    request = CreateRepositoryRequest(
        parent=parent,
        repository_id=name,
        repository=Repository(format_=format),
    )
    try:
        # Sends the request to create the repository in Google Artifact Registry
        response = client.create_repository(request=request)
        return response.result().name  # type: ignore
    except exceptions.AlreadyExists as ae:
        repository_path = f"{parent}/repositories/{name}"

        # Sends a request to get the information of the existing repository, if it exists
        # then checks if it's a repository matching `format`, if not, raises an exception.
        request = GetRepositoryRequest(name=repository_path)
        response = client.get_repository(request=request)

        # Checks the format of the existing repository, if it's not a `format` repository
        # then raises an exception.
        response_format = Repository.Format(response.format_).name
        if response_format != format:
            raise RuntimeError(
                f"`repository={name}` already exists, but it's not a Docker repository"
                f" but a `{response_format}` one, please make sure to specify another"
                " `name` instead."
            ) from ae

        warnings.warn(
            f"Skipping `repository={name}` creation since it already exists!",
            stacklevel=1,
        )
        return repository_path
    except Exception as e:
        raise RuntimeError(
            f"`repository={name}` couldn't be created with exception `{e}`"
        ) from e
