import warnings

from google.cloud.storage import Client


def upload_file_to_gcs(
    project_id: str,
    location: str,
    local_path: str,
    bucket_name: str,
    remote_path: str,
) -> str:
    """Uploads a file from local storage to Google Cloud Storage.

    Args:
        project_id: is either the name or the identifier of the project in Google Cloud.
        location: is the identifier of the region and zone where the file will be uploaded to.
        local_path: is the path to the file in the local storage.
        bucket_name: is the name of the bucket in Google Cloud Storage where the file will
            be uploaded to.
        remote_path: is the destination path in Google Cloud Storage where the file will be
            uploaded to.

    Returns:
        The path in Google Cloud Storage to the uploaded file.
    """

    client = Client(project=project_id)

    # If the bucket doesn't exist, we create it, ensuring that the `uniform_bucket_level_access_enabled`
    # is enabled so that we don't run into permission issues when downloading files from
    # the bucket from the running container in Vertex AI.
    bucket = client.bucket(bucket_name)
    if not bucket.exists():
        warnings.warn(
            f"Bucket '{bucket_name}' does not exist. Creating it now.", stacklevel=1
        )
        client.create_bucket(bucket_name, location=location)

        bucket.iam_configuration.uniform_bucket_level_access_enabled = True
        bucket.patch()

    # Finally, the blob is created and the file is uploaded to that blob
    blob = bucket.blob(remote_path)
    blob.upload_from_filename(local_path)
    return f"gs://{bucket_name}/{remote_path}"  # type: ignore
