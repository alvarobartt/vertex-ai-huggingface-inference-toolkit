import re
import subprocess
import sys
from datetime import datetime
from typing import List, Optional

if sys.version_info < (3, 9):
    import importlib_resources
else:
    import importlib.resources as importlib_resources

import docker

from vertex_ai_huggingface_inference_toolkit.utils import CACHE_PATH


def build_docker_image(
    python_version: str,
    framework: str,
    framework_version: str,
    transformers_version: str,
    cuda_version: Optional[str] = None,
    ubuntu_version: Optional[str] = None,
    extra_requirements: Optional[List[str]] = None,
) -> str:
    """Builds the Docker image locally using `docker`, building it via `--build-args`
    on top of either `Dockerfile.cpu` or `Dockerfile.gpu` provided within the current
    package, since those images are already suited for inference in Vertex AI.

    Args:
        python_version: is the Python version to be installed via `apt-get` install, so
            it needs to be provided as a string, i.e. `3.9`, `3.10`, `3.11`, etc.
        framework: is the identifier of the deep learning framework to use. Available
            options for the moment are `torch`, `tensorflow` and `jax`.
        framework_version: is the version of the provided framework as shown in PyPI.
        transformers_version: is the version of `transformers` to install, since the
            inference code will be run via `transformers`.
        cuda_version: is the version of CUDA to use, if planning to deploy the model
            within an instance with GPU acceleration. The CUDA versions to be provided
            need to be in the format of X.Y.Z, and available at https://hub.docker.com/r/nvidia/cuda/tags?page=1&name=-base-ubuntu
            i.e. "12.3.0", "12.3.2", etc.
        ubuntu_version: is the version of Ubuntu which depends on the CUDA version specified
            above, since it's appended to the image name to pull from. The available Ubuntu versions
            per CUDA version are available at https://hub.docker.com/r/nvidia/cuda/tags?page=1&name=runtime-ubuntu too.
        extra_requirements: is an optional list of requirements to install within the
            image, following the `pip install` formatting i.e. `sentence-transformers >= 2.5.0`.

    Returns:
        The Docker image name, including the tag, that has been built locally.
    """

    # The tag is set in advance, generated from the replacements of the `--build-args`
    _device_string = f"cu{cuda_version}" if cuda_version is not None else "cpu"
    _tag = f"py{python_version}-{_device_string}-{framework}-{framework_version}-transformers-{transformers_version}"

    # The `_build_args` to be replaced in the `Dockerfile` when building it need to be
    # prepared in advance, to ensure the formatting and assignment is fine.
    _dockerfile = "Dockerfile.cpu"
    _build_args = {
        "PYTHON_VERSION": python_version,
        "FRAMEWORK": framework,
        "FRAMEWORK_VERSION": framework_version,
        "TRANSFORMERS_VERSION": transformers_version,
    }
    if cuda_version is not None:
        _build_args["CUDA_VERSION"] = cuda_version
        _dockerfile = "Dockerfile.gpu"
    if ubuntu_version is not None:
        _build_args["UBUNTU_VERSION"] = ubuntu_version
    if extra_requirements is not None:
        _build_args["EXTRA_REQUIREMENTS"] = " ".join(extra_requirements)

    _path = str(
        importlib_resources.files("vertex_ai_huggingface_inference_toolkit")
        / "_internal"
        / "dockerfiles"
    )

    cache_path = CACHE_PATH / _tag / datetime.now().strftime("%Y-%m-%d--%H:%M")
    cache_path.mkdir(parents=True, exist_ok=True)

    # (Optional) On top of the pre-defined `Dockerfile`, the replacements for the
    # `--build-args` are applied using the `_build_args` dictionary
    dockerfile_content = open(f"{_path}/{_dockerfile}", mode="r").read()
    for arg, value in _build_args.items():
        pattern = re.compile(rf"\$\{{\s*{arg}\s*}}")
        dockerfile_content = re.sub(pattern, value, dockerfile_content)

    # (Optional) The generated `Dockerfile` is stored within the cache for reproducibility
    dockerfile_path = cache_path / "Dockerfile"
    with dockerfile_path.open(mode="w") as dockerfile:
        dockerfile.write(dockerfile_content)

    # The `Dockerfile` is built using `platform=linux/amd64` as it will be deployed in
    # an instance running Linux
    client = docker.from_env()  # type: ignore
    image, _ = client.images.build(  # type: ignore
        path=_path,
        dockerfile=_dockerfile,
        platform="linux/amd64",
        buildargs=_build_args,
        tag=_tag,
        quiet=False,
        rm=True,
    )
    return image.tags[0]  # type: ignore


def configure_docker_and_push_image(
    project_id: str,
    location: str,
    repository: str,
    image_with_tag: str,
) -> str:
    """Configures Docker to use the Google Cloud Artifact Registry and pushes the image
    that has been built locally in advance and so on, available when listing the images
    with the command `docker images`.

    Args:
        project_id: is either the name or the identifier of the project in Google Cloud.
        location: is the identifier of the region and zone where the image will be pushed to.
        repository: is the name of the Docker repository in Google Artifact Registry.
        image_with_tag: is the Docker image built locally, including the tag.

    Returns:
        The repository path to the Docker image that has been pushed to Google Artifact Registry.
    """

    # If no tag has been provided, then assume the tag to use is `latest`
    if len(image_with_tag.split(":")) != 2:
        image_with_tag += ":latest"

    # Configures Docker to be authenticated within the Docker repository before pushing
    repository_url = f"{location}-docker.pkg.dev"
    # NOTE: running a `gcloud` command via `subprocess` from Python is not the most optimal
    # solution at all, but there's no support within Google Python SDKs for `gcloud auth configure-docker`
    subprocess.run(["gcloud", "auth", "configure-docker", repository_url, "--quiet"])

    repository_path = f"{repository_url}/{project_id}/{repository}/{image_with_tag}"

    # Sets the tag to the Docker image (matching the destination path in Google Artifact Registry)
    # and pushes the image using that tag.
    client = docker.from_env()  # type: ignore
    client.images.get(image_with_tag).tag(repository_path)  # type: ignore
    client.images.push(repository_path)
    return repository_path
