## 🔮 TODOs

- [ ] Add some working examples under `examples/`
- [ ] Add custom `logging` messages within the `__init__` and `deploy` methods, skipping the `google-cloud-aiplatform` and any other logs already there
- [X] Add support for models available locally (provided via `model_name_or_path` but without having to call `snapshot_download`)
- [ ] Handle logic within `TransformersModel` when `HF_HUB_ID` is within `environment_variables` and neither `model_name_or_path` nor `model_bucket_uri` have been provided
- [X] Add some more flags in case the user wants to specify the target bucket in Google Cloud Storage and the target repository in Google Artifact Registry
- [ ] Handle the possible / supported values for the different `--build-args` provided to the Docker image
- [ ] Add flag `experimental` to bypass the predefined checks / untested stuff in case users want to experiment further with the package without the limitations
- [ ] Should we allow custom Docker images or just provide the optimized and supported ones? On the same topic, should we upload some of those pre-built to the Docker Hub so that those can be pulled from there instead of being built every single time locally?
