from typing import Optional, Tuple

import torch
from torch.nn import functional as F
from transformers import BlipProcessor, BlipForImageTextRetrieval
import torch_tensorrt

__all__ = ['BlipProcessorInstance', "get_compiled_blip"]

MODEL_NAME = "Salesforce/blip-itm-large-coco"
BlipProcessorInstance: BlipProcessor = BlipProcessor.from_pretrained(MODEL_NAME, use_fast=True)
_model: BlipForImageTextRetrieval = BlipForImageTextRetrieval.from_pretrained(MODEL_NAME)

assert torch.cuda.is_available()
_model = _model.cuda().eval()
_compiled_model = None

class BlipWrapper(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = _model

    def forward(
            self,
            input_ids: torch.LongTensor,
            pixel_values: torch.FloatTensor,
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

        return itm_score, cosine_similarity

def get_compiled_blip():
    global _compiled_model
    if _compiled_model is None:
        wrapped_model = BlipWrapper().cuda().eval()
        _compiled_model =  torch_tensorrt.compile(
            wrapped_model,
            inputs=[
                torch_tensorrt.Input(
                    min_shape=(1, 3, 384, 384),
                    opt_shape=(16, 3, 384, 384),
                    max_shape=(64, 3, 384, 384),
                    dtype=torch.float16
                ),
                torch_tensorrt.Input(
                    min_shape=(1, 64),
                    opt_shape=(16, 64),
                    max_shape=(64, 64),
                    dtype=torch.long
                ),
                torch_tensorrt.Input(
                    min_shape=(1, 64),
                    opt_shape=(16, 64),
                    max_shape=(64, 64),
                    dtype=torch.long
                )
            ],
            enabled_precisions = {torch.float16},
            optimization_level = 5
        )
    return _compiled_model