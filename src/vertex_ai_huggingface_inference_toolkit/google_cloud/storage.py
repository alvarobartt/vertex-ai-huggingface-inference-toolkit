import os
import warnings
from pathlib import Path
from typing import Optional

from google.cloud.storage import Client


def upload_directory_to_gcs(
    project_id: str,
    location: str,
    local_dir: str,
    bucket_name: Optional[str] = None,
    remote_dir: Optional[str] = None,
) -> str:
    client = Client(project=project_id)

    bucket = client.bucket(bucket_name)
    if not bucket.exists():
        warnings.warn(
            f"Bucket '{bucket_name}' does not exist. Creating it now.", stacklevel=1
        )
        client.create_bucket(bucket_name, location=location)

    local_path = Path(local_dir)
    for local_file in local_path.glob("**/*"):
        if local_file.is_file():
            relative_path = local_file.relative_to(local_path)
            remote_file_path = os.path.join(remote_dir or "", str(relative_path))

            blob = bucket.blob(remote_file_path)
            blob.upload_from_filename(str(local_file))
    return bucket.path  # type: ignore
