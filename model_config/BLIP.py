import sys
from pathlib import Path
from typing import Optional, Tuple, Union

import torch
from torch.fx import GraphModule
from torch.nn import functional as F
from transformers import BlipProcessor, BlipForImageTextRetrieval
import torch_tensorrt

assert torch.cuda.is_available()

MODEL_NAME = "Salesforce/blip-itm-large-coco"
DEFAULT_BLIP_MOD = "data/blip_trt.pt2"
DEFAULT_BLIP_PREPROCESSOR_PATH = "data/blip_preprocessor/"


def set_default_paths(mod: str, pr: str):
    global DEFAULT_BLIP_MOD, DEFAULT_BLIP_PREPROCESSOR_PATH
    DEFAULT_BLIP_MOD = mod
    DEFAULT_BLIP_PREPROCESSOR_PATH = pr

def prepend_default_blip_paths(prefix: str):
    prefix = prefix.rstrip("/")
    global DEFAULT_BLIP_MOD, DEFAULT_BLIP_PREPROCESSOR_PATH
    DEFAULT_BLIP_MOD = prefix + "/" + DEFAULT_BLIP_MOD
    DEFAULT_BLIP_PREPROCESSOR_PATH = prefix + "/" + DEFAULT_BLIP_PREPROCESSOR_PATH

class BlipWrapper(torch.nn.Module):
    def __init__(self):
        super().__init__()
        _model: BlipForImageTextRetrieval = BlipForImageTextRetrieval.from_pretrained(MODEL_NAME)
        _model = _model.cuda().eval()
        self.model = _model

    def forward(
            self,
            pixel_values: torch.FloatTensor,
            input_ids: torch.LongTensor,
            attention_mask: Optional[torch.LongTensor] = None,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        vision_outputs = self.model.vision_model(
            pixel_values=pixel_values, interpolate_pos_encoding=False, return_dict=False)

        image_embeds = vision_outputs[0]
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long)

        # Compute ITM
        q_embeds_itm = self.model.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=False,
        )
        q_embeds_itm = q_embeds_itm[0]
        itm_score = self.model.itm_head(q_embeds_itm[:, 0, :])

        # Compute cosine similarity
        q_embeds = self.model.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False,
        )
        q_embeds = q_embeds[0]
        image_feat = F.normalize(self.model.vision_proj(image_embeds[:, 0, :]), dim=-1)
        text_feat = F.normalize(self.model.text_proj(q_embeds[:, 0, :]), dim=-1)
        cosine_similarity = image_feat @ text_feat.t()

        return itm_score.softmax(dim=1)[:, 1], cosine_similarity.diag()

def _compile_blip_cuda() -> GraphModule:
    wrapped_model = BlipWrapper().cuda().eval()
    _compiled_model =  torch_tensorrt.compile(
        wrapped_model,
        inputs=[
            torch_tensorrt.Input(
                min_shape=(1, 3, 384, 384),
                opt_shape=(32, 3, 384, 384),
                max_shape=(64, 3, 384, 384),
                dtype=torch.float16
            ),
            torch_tensorrt.Input(
                min_shape=(64, 2),
                opt_shape=(64, 64),
                max_shape=(64, 128),
                dtype=torch.long
            ),
            torch_tensorrt.Input(
                min_shape=(64, 2),
                opt_shape=(64, 64),
                max_shape=(64, 128),
                dtype=torch.long
            )
        ],
        enabled_precisions = {torch.float16},
        optimization_level = 5
    )
    return _compiled_model

def export_blip(path: Optional[Union[str, Path]]=None, overwrite: bool=False):
    if path is None:
        path = DEFAULT_BLIP_MOD
    if Path(path).exists() and not overwrite:
        print(f"BLIP model already exists at {path}. Skipping export.", file=sys.stderr)
        return
    torch_tensorrt.save(_compile_blip_cuda(), path)

def get_compiled_blip(path: Optional[Union[str, Path]]=None, force_compile: bool = False) -> GraphModule:
    if path is None and not force_compile:
        path = DEFAULT_BLIP_MOD
    if not Path(path).exists():
        print(f"BLIP model not found at {path}. Compiling...")
        return _compile_blip_cuda()
    print(f"Loading BLIP model from {path or DEFAULT_BLIP_MOD}")
    return torch_tensorrt.load(path).module()

def save_blip_preprocessor(path: Optional[Union[str, Path]]=None, overwrite: bool=False):
    if path is None:
        path = DEFAULT_BLIP_PREPROCESSOR_PATH
    if Path(path).exists() and not overwrite:
        print(f"Preprocessor already exists at {path}. Skipping export.", file=sys.stderr)
        return
    processor = BlipProcessor.from_pretrained(MODEL_NAME, use_fast=True)
    processor.save_pretrained(save_directory=path)

def load_blip_preprocessor(path: Optional[Union[str, Path]]=None) -> BlipProcessor:
    if path is None:
        path = DEFAULT_BLIP_PREPROCESSOR_PATH
    elif not Path(path).exists():
        print(f"Preprocessor not found at {path}. Downloading from HuggingFace...", file=sys.stderr)
        return BlipProcessor.from_pretrained(MODEL_NAME, use_fast=True)
    return BlipProcessor.from_pretrained(path, use_fast=True)