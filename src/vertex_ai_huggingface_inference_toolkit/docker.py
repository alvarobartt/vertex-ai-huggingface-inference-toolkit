import re
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
    extra_requirements: Optional[List[str]] = None,
) -> str:
    _cuda_string = f"cu{cuda_version}" if cuda_version is not None else "cpu"
    _tag = f"py{python_version}_{_cuda_string}_{framework}_{framework_version}_transformers_{transformers_version}"

    _dockerfile = "Dockerfile.cpu"
    _build_args = {
        "platform": "linux/amd64",
        "PYTHON_VERSION": python_version,
        "FRAMEWORK": framework,
        "FRAMEWORK_VERSION": framework_version,
        "TRANSFORMERS_VERSION": transformers_version,
    }
    if cuda_version is not None:
        _build_args["CUDA_VERSION"] = cuda_version
        _dockerfile = "Dockerfile.gpu"
    if extra_requirements is not None:
        _build_args["EXTRA_REQUIREMENTS"] = " ".join(extra_requirements)

    _path = str(
        importlib_resources.files("vertex_ai_huggingface_inference_toolkit")
        / "dockerfiles"
    )

    cache_path = CACHE_PATH / _tag / datetime.now().strftime("%Y-%m-%d--%H:%M")
    cache_path.mkdir(parents=True, exist_ok=True)

    dockerfile_content = open(f"{_path}/{_dockerfile}", mode="r").read()
    for arg, value in _build_args.items():
        pattern = re.compile(rf"\$\{{\s*{arg}\s*}}")
        dockerfile_content = re.sub(pattern, value, dockerfile_content)

    dockerfile_path = cache_path / "Dockerfile"
    with dockerfile_path.open(mode="w") as dockerfile:
        dockerfile.write(dockerfile_content)

    client = docker.from_env()
    image, _ = client.images.build(  # type: ignore
        path=_path,
        dockerfile=_dockerfile,
        buildargs=_build_args,
        tag=_tag,
        quiet=False,
        rm=True,
    )
    return image.tags[0]  # type: ignore
