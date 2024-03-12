from diffusers.pipelines.auto_pipeline import (
    AutoPipelineForImage2Image,
    AutoPipelineForInpainting,
    AutoPipelineForText2Image,
)

PIPELINE_TASKS = {
    "text-to-image": AutoPipelineForText2Image,
    "image-to-text": AutoPipelineForImage2Image,
    "inpainting": AutoPipelineForInpainting,
}
