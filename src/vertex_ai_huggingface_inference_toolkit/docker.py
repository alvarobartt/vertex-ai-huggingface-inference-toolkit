from datetime import datetime
from typing import List

import docker

from vertex_ai_huggingface_inference_toolkit.utils import CACHE_PATH


def build_docker_image(base_image: str, requirements: List[str]) -> str:
    cache_path = (
        CACHE_PATH
        / f"{base_image.replace(':', '_')}_{datetime.now().strftime('%Y%m%d%H%M')}"
    )
    cache_path.mkdir(parents=True, exist_ok=True)

    dockerfile_content = f"FROM {base_image}\n"
    dockerfile_content += "RUN python -m pip install --no-cache-dir --upgrade pip"
    for req in requirements:
        dockerfile_content += f" && \\\n    python -m pip install --no-cache-dir {req}"

    dockerfile_path = cache_path / "Dockerfile"
    with dockerfile_path.open(mode="w") as dockerfile:
        dockerfile.write(dockerfile_content)

    client = docker.from_env()
    image, _ = client.images.build(  # type: ignore
        path=cache_path.as_posix(),
        dockerfile="Dockerfile",
        buildargs={"platform": "linux/amd64"},
        tag="vertex-ai-huggingface-inference-toolkit:latest",
        quiet=False,
        rm=True,
    )
    return image.tags[0]  # type: ignore
