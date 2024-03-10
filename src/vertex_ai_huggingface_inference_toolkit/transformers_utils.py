PIPELINE_TASKS = [
    "audio-classification",
    "automatic-speech-recognition",
    "conversational",
    "depth-estimation",
    "document-question-answering",
    "feature-extraction",
    "fill-mask",
    "image-classification",
    "image-feature-extraction",
    "image-segmentation",
    "image-to-image",
    "image-to-text",
    "mask-generation",
    "object-detection",
    "question-answering",
    "summarization",
    "table-question-answering",
    "text2text-generation",
    "text-classification",
    "sentiment-analysis",
    "text-generation",
    "text-to-audio",
    "text-to-speech",
    "token-classification",
    "ner",
    "translation",
    "translation_xx_to_yy",
    "video-classification",
    "visual-question-answering",
    "zero-shot-classification",
    "zero-shot-image-classification",
    "zero-shot-audio-classification",
    "zero-shot-object-detection",
]
"""List of the supported `task` values in `transformers.pipeline`, available at
https://huggingface.co/docs/transformers/v4.38.2/en/main_classes/pipelines#transformers.pipeline.task
"""

FEATURE_EXTRACTOR_TASKS = [
    "automatic-speech-recognition",
    "image-segmentation",
    "image-classification",
    "audio-classification",
    "object-detection",
    "zero-shot-image-classification",
]
"""List of the `task` values in `transformers.pipeline` that require a `feature_extractor` to
be loaded successfully, and ensured it's used in the `pipeline` constructor.
"""
