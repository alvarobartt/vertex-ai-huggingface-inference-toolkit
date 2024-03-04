import warnings
from typing import Optional

from google.cloud.storage import Client


def upload_file_to_gcs(
    project_id: str,
    location: str,
    local_path: str,
    remote_path: str,
    bucket_name: Optional[str] = None,
) -> str:
    client = Client(project=project_id)

    if bucket_name is None:
        bucket_name = "vertex-ai-huggingface-inference-toolkit"

    bucket = client.bucket(bucket_name)
    if not bucket.exists():
        warnings.warn(
            f"Bucket '{bucket_name}' does not exist. Creating it now.", stacklevel=1
        )
        client.create_bucket(bucket_name, location=location)
    bucket.iam_configuration.uniform_bucket_level_access_enabled = True
    bucket.patch()

    blob = bucket.blob(remote_path)
    blob.upload_from_filename(local_path)
    return f"gs://{bucket_name}/{remote_path.replace('/model.tar.gz', '')}"  # type: ignore
