import torch
from open_clip import create_model_from_pretrained, get_tokenizer
import torch_tensorrt

__all__ = ['SigLipPreprocessor', 'SigLipTokenizer', 'get_compiled_siglip_vision', 'get_compiled_siglip_text']

MODEL_NAME = 'hf-hub:timm/ViT-B-16-SigLIP2-512'
_model, SigLipPreprocessor = create_model_from_pretrained(MODEL_NAME)
SigLipTokenizer = get_tokenizer(MODEL_NAME)

assert torch.cuda.is_available()
_model.cuda().eval()

_compiled_vision_model = None
_compiled_text_model = None

def get_compiled_siglip_vision():
    global _compiled_vision_model
    if _compiled_vision_model is None:
        _compiled_vision_model = torch_tensorrt.compile(
            _model.visual,
            inputs=[
                torch_tensorrt.Input(
                    min_shape=(1, 3, 512, 512),
                    opt_shape=(64, 3, 512, 512),
                    max_shape=(128, 3, 512, 512)
                )
            ],
            enabled_precisions={torch_tensorrt.dtype.f16},
            optimization_level=5
        )
    return _compiled_vision_model

def get_compiled_siglip_text():
    global _compiled_text_model
    if _compiled_text_model is None:
        _compiled_text_model = torch_tensorrt.compile(
            _model.text,
            inputs=[
                torch_tensorrt.Input(
                    min_shape=(1, 64),
                    opt_shape=(64, 64),
                    max_shape=(128, 64),
                    dtype=torch.long
                )
            ],
            enabled_precisions={torch.float16},
            optimization_level=5
        )
    return _compiled_text_model