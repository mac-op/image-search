import os
import torch

from models.SigLIP import (
    export_siglip_text,
    export_siglip_vision,
    export_tokenizer,
    prepend_default_siglip_paths,
)
from models.BLIP import (
    export_blip,
    save_blip_preprocessor,
    prepend_default_blip_paths
)

def write_to_mount():
    if not os.environ.get("DEV"):
        prepend_default_siglip_paths("/mnt")
        prepend_default_blip_paths("/mnt")
    export_siglip_text()
    export_siglip_vision()
    torch.cuda.empty_cache()

    export_tokenizer()
    export_blip()
    save_blip_preprocessor()
    torch.cuda.empty_cache()

