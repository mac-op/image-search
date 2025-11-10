import gc
import sys
from pathlib import Path
import torch

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.SigLIP import export_siglip_text, export_siglip_vision, export_tokenizer
from models.BLIP import export_blip, save_blip_preprocessor

if __name__ == "__main__":
    export_siglip_text()
    export_siglip_vision()
    torch.cuda.empty_cache()

    export_tokenizer()
    export_blip()
