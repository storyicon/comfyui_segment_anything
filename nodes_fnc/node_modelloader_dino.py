import os
import torch

import folder_paths

from ..libs.local_groundingdino.util.utils import clean_state_dict as local_groundingdino_clean_state_dict
from ..libs.local_groundingdino.util.slconfig import SLConfig as local_groundingdino_SLConfig
from ..libs.local_groundingdino.models import build_model as local_groundingdino_build_model

from ..utils.collection  import get_local_filepath

groundingdino_model_dir = os.path.join(
    folder_paths.models_dir, "grounding-dino")
groundingdino_model_list = {
    "GroundingDINO_SwinT_OGC (694MB)": {
        "config_url": "https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/GroundingDINO_SwinT_OGC.cfg.py",
        "model_url": "https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swint_ogc.pth",
    },
    "GroundingDINO_SwinB (938MB)": {
        "config_url": "https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/GroundingDINO_SwinB.cfg.py",
        "model_url": "https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swinb_cogcoor.pth"
    },
}



def load_groundingdino_model(model_name):
    dino_model_args = local_groundingdino_SLConfig.fromfile(
        get_local_filepath(
            groundingdino_model_list[model_name]["config_url"],
            groundingdino_model_dir
        ),
    )
    dino = local_groundingdino_build_model(dino_model_args)
    checkpoint = torch.load(
        get_local_filepath(
            groundingdino_model_list[model_name]["model_url"],
            groundingdino_model_dir,
        ),
    )
            
    dino.load_state_dict(local_groundingdino_clean_state_dict(
        checkpoint['model']), strict=False)
    return dino


def list_groundingdino_model():
    return list(groundingdino_model_list.keys())
