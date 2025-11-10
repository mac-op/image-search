import sys
from pathlib import Path
from typing import Optional

import torch
from open_clip import create_model_from_pretrained, get_tokenizer, CustomTextCLIP
import torch_tensorrt
from open_clip.tokenizer import HFTokenizer
from open_clip.transform import _convert_to_rgb
from torch.fx import GraphModule
from torchvision.transforms import Compose, Resize, InterpolationMode, ToTensor, Normalize

MODEL_NAME = 'hf-hub:timm/ViT-B-16-SigLIP2-512'

_model: Optional[CustomTextCLIP] = None
SigLipTokenizer: Optional[HFTokenizer] = None

DEFAULT_SIGLIP_VIS_MOD = "data/siglip_vis_trt.pt2"
DEFAULT_SIGLIP_TEXT_MOD = "data/siglip_text_trt.pt2"
DEFAULT_SIGLIP_TOKENIZER_PATH = "data/"

def set_default_paths(v: str, t: str, tk: str):
    global DEFAULT_SIGLIP_VIS_MOD, DEFAULT_SIGLIP_TEXT_MOD, DEFAULT_SIGLIP_TOKENIZER_PATH
    DEFAULT_SIGLIP_VIS_MOD = v
    DEFAULT_SIGLIP_TEXT_MOD = t
    DEFAULT_SIGLIP_TOKENIZER_PATH = tk

def prepend_default_siglip_paths(prefix: str):
    prefix = prefix.rstrip("/")
    global DEFAULT_SIGLIP_VIS_MOD, DEFAULT_SIGLIP_TEXT_MOD, DEFAULT_SIGLIP_TOKENIZER_PATH
    DEFAULT_SIGLIP_VIS_MOD = prefix + "/" + DEFAULT_SIGLIP_VIS_MOD
    DEFAULT_SIGLIP_TEXT_MOD = prefix + "/" + DEFAULT_SIGLIP_TEXT_MOD
    DEFAULT_SIGLIP_TOKENIZER_PATH = prefix + "/" + DEFAULT_SIGLIP_TOKENIZER_PATH

assert torch.cuda.is_available()

_compiled_vision_model = None
_compiled_text_model = None

def _get_siglip():
    global _model
    if _model is None:
        _model, SigLipPreprocessor = create_model_from_pretrained(MODEL_NAME)
    _model.cuda().eval()
    return _model

def _compile_siglip_vision() -> GraphModule:
    return torch_tensorrt.compile(
        _get_siglip().visual.cuda().eval(),
        inputs=[
            torch_tensorrt.Input(
                min_shape=(1, 3, 512, 512),
                opt_shape=(32, 3, 512, 512),
                max_shape=(64, 3, 512, 512)
            )
        ],
        enabled_precisions={torch_tensorrt.dtype.f16},
        optimization_level=5
    )

def export_siglip_vision(path: Optional[str|Path]=None, overwrite: bool=False):
    if path is None:
        path = DEFAULT_SIGLIP_VIS_MOD
    if Path(path).exists() and not overwrite:
        print(f"Vision model already exists at {path}. Skipping export.", file=sys.stderr)
        return
    torch_tensorrt.save(_compile_siglip_vision(), path)

def get_compiled_siglip_vision(path: Optional[str|Path]=None, force_compile: bool=False) -> GraphModule:
    if path is None:
        if not force_compile:
            path = DEFAULT_SIGLIP_VIS_MOD
        else:
            return _compile_siglip_vision()
    if not Path(path).exists():
        print(f"Vision model not found at {path}. Downloading from HuggingFace...", file=sys.stderr)
        return _compile_siglip_vision()
    return torch_tensorrt.load(path).module()

def _compile_siglip_text() -> GraphModule:
    return torch_tensorrt.compile(
        _get_siglip().text.cuda().eval(),
        inputs=[
            torch_tensorrt.Input(
                min_shape=(1, 64),
                opt_shape=(32, 64),
                max_shape=(64, 64),
                dtype=torch.long
            ),
        ],
        enabled_precisions={torch_tensorrt.dtype.f16},
        optimization_level=5
    )

def export_siglip_text(path: Optional[str|Path]=None, overwrite: bool=False):
    if path is None:
        path = DEFAULT_SIGLIP_TEXT_MOD
    if Path(path).exists() and not overwrite:
        print(f"Text model already exists at {path}. Skipping export.", file=sys.stderr)
        return
    torch_tensorrt.save(_compile_siglip_text(), path)

def get_compiled_siglip_text(path: Optional[str|Path]=None, force_compile: bool=False) -> GraphModule:
    if path is None:
        if not force_compile:
            path = DEFAULT_SIGLIP_TEXT_MOD
        else:
            return _compile_siglip_text()
    if not Path(path).exists():
        print(f"Text model not found at {path}. Downloading from HuggingFace...", file=sys.stderr)
        return _compile_siglip_text()
    return torch_tensorrt.load(path).module()

def export_tokenizer(path: Optional[str|Path]=None, overwrite: bool=False):
    if path is None:
        path = DEFAULT_SIGLIP_TOKENIZER_PATH
    if Path(path).exists() and not overwrite:
        print(f"Tokenizer already exists at {path}. Skipping export.", file=sys.stderr)
        return
    global SigLipTokenizer
    if SigLipTokenizer is None:
        SigLipTokenizer = get_tokenizer(MODEL_NAME)
    SigLipTokenizer.save_pretrained(dest=path)

def get_siglip_tokenizer(path: Optional[str|Path]=None) -> HFTokenizer:
    if path is None:
        path = DEFAULT_SIGLIP_TOKENIZER_PATH
    if not Path(path).exists():
        print(f"Tokenizer not found at {path}. Downloading from HuggingFace...", file=sys.stderr)
        return get_tokenizer(MODEL_NAME)
    return get_tokenizer(path)

def get_siglip_preprocessor() -> Compose:
    return Compose([
        Resize(size=(512, 512), interpolation=InterpolationMode.BILINEAR, max_size=None, antialias=True),
        _convert_to_rgb,
        ToTensor(),
        Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])