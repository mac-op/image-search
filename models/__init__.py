from dataclasses import dataclass
from typing import Optional

from open_clip.tokenizer import HFTokenizer
from torch.fx import GraphModule
from torchvision.transforms import Compose
from transformers import BlipProcessor

from models.BLIP import get_compiled_blip, load_blip_preprocessor
from models.SigLIP import get_compiled_siglip_text, get_compiled_siglip_vision, get_siglip_preprocessor, \
    get_siglip_tokenizer


@dataclass
class ModelConfig:
    blip_model: Optional[GraphModule]
    siglip_text_model: Optional[GraphModule]
    siglip_vision_model: Optional[GraphModule]

    blip_preprocessor: Optional[BlipProcessor]
    siglip_preprocess: Optional[Compose]
    siglip_tokenizer: Optional[HFTokenizer]

    @staticmethod
    def default():
        return ModelConfig(
            blip_model=get_compiled_blip(),
            siglip_text_model=get_compiled_siglip_text(),
            siglip_vision_model=get_compiled_siglip_vision(),
            blip_preprocessor=load_blip_preprocessor(),
            siglip_preprocess=get_siglip_preprocessor(),
            siglip_tokenizer=get_siglip_tokenizer()
        )

    @staticmethod
    def for_ingest():
        return ModelConfig(
            blip_model=None,
            siglip_text_model=None,
            siglip_vision_model=get_compiled_siglip_vision(),
            blip_preprocessor=None,
            siglip_preprocess=get_siglip_preprocessor(),
            siglip_tokenizer=None
        )

    @staticmethod
    def for_inference():
        return ModelConfig(
            blip_model=get_compiled_blip(),
            siglip_text_model=get_compiled_siglip_text(),
            siglip_vision_model=None,
            blip_preprocessor=load_blip_preprocessor(),
            siglip_preprocess=None,
            siglip_tokenizer=get_siglip_tokenizer()
        )