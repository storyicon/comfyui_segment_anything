import os
import sys
sys.path.append(
    os.path.dirname(os.path.abspath(__file__))
)

import copy
import torch
import numpy as np
from PIL import Image
import logging
from torch.hub import download_url_to_file
from urllib.parse import urlparse
import folder_paths
import comfy.model_management
from sam_hq.predictor import SamPredictorHQ
from sam_hq.build_sam_hq import sam_model_registry
from local_groundingdino.datasets import transforms as T
from local_groundingdino.util.utils import clean_state_dict as local_groundingdino_clean_state_dict
from local_groundingdino.util.slconfig import SLConfig as local_groundingdino_SLConfig
from local_groundingdino.models import build_model as local_groundingdino_build_model
import glob
import folder_paths

logger = logging.getLogger('comfyui_segment_anything')

sam_model_dir_name = "sams"
sam_model_list = {
    "sam_vit_h (2.56GB)": {
        "model_url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
    },
    "sam_vit_l (1.25GB)": {
        "model_url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth"
    },
    "sam_vit_b (375MB)": {
        "model_url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
    },
    "sam_hq_vit_h (2.57GB)": {
        "model_url": "https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_h.pth"
    },
    "sam_hq_vit_l (1.25GB)": {
        "model_url": "https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_l.pth"
    },
    "sam_hq_vit_b (379MB)": {
        "model_url": "https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_b.pth"
    },
    "mobile_sam(39MB)": {
        "model_url": "https://github.com/ChaoningZhang/MobileSAM/blob/master/weights/mobile_sam.pt"
    }
}

groundingdino_model_dir_name = "grounding-dino"
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

def get_bert_base_uncased_model_path() -> str:
    """Returns the path for the BERT base uncased model if it exists, otherwise returns a default path."""
    comfy_bert_model_base = os.path.join(folder_paths.models_dir, 'bert-base-uncased')
    if glob.glob(os.path.join(comfy_bert_model_base, '**/model.safetensors'), recursive=True):
        print('grounding-dino is using models/bert-base-uncased')
        return comfy_bert_model_base
    return 'bert-base-uncased'

def list_files(dirpath, extensions=[]):
    return [f for f in os.listdir(dirpath) if os.path.isfile(os.path.join(dirpath, f)) and f.split('.')[-1] in extensions]


def list_sam_model() -> list[str]:
    """Returns a list of keys from the dictionary `sam_model_list` as strings."""
    return list(sam_model_list.keys())


def load_sam_model(model_name: str) -> Any:
    """Loads a SAM (Semi-supervised Anomaly Detection Model) model specified by the model name. It retrieves the local file path for the model using the provided URL or extracts it from the URL. The function then initializes the model based on the retrieved checkpoint path, sets the appropriate device, and returns the loaded model."""
    sam_checkpoint_path = get_local_filepath(
        sam_model_list[model_name]["model_url"], sam_model_dir_name)
    model_file_name = os.path.basename(sam_checkpoint_path)
    model_type = model_file_name.split('.')[0]
    if 'hq' not in model_type and 'mobile' not in model_type:
        model_type = '_'.join(model_type.split('_')[:-1])
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint_path)
    sam_device = comfy.model_management.get_torch_device()
    sam.to(device=sam_device)
    sam.eval()
    sam.model_name = model_file_name
    return sam


def get_local_filepath(url: str, dirname: str, local_file_name: str | None = None) -> str:
    """Retrieves the local file path for a given URL by either using the provided local file name or extracting it from the URL. If the file already exists locally, it returns the path; otherwise, it downloads the file from the URL to the specified destination and returns the path."""
    if not local_file_name:
        parsed_url = urlparse(url)
        local_file_name = os.path.basename(parsed_url.path)

    destination = folder_paths.get_full_path(dirname, local_file_name)
    if destination:
        logger.warn(f'using extra model: {destination}')
        return destination

    folder = os.path.join(folder_paths.models_dir, dirname)
    if not os.path.exists(folder):
        os.makedirs(folder)

    destination = os.path.join(folder, local_file_name)
    if not os.path.exists(destination):
        logger.warn(f'downloading {url} to {destination}')
        download_url_to_file(url, destination)
    return destination


def load_groundingdino_model(model_name: str) -> local_groundingdino_build_model:
    """Loads a groundingdino model based on the provided model name, configuration settings, and model checkpoint. It adjusts the text encoder type if necessary, loads the model checkpoint, cleans the state dictionary keys, moves the model to the appropriate device, sets the model to evaluation mode, and returns the loaded model."""
    dino_model_args = local_groundingdino_SLConfig.fromfile(
        get_local_filepath(
            groundingdino_model_list[model_name]["config_url"],
            groundingdino_model_dir_name
        ),
    )

    if dino_model_args.text_encoder_type == 'bert-base-uncased':
        dino_model_args.text_encoder_type = get_bert_base_uncased_model_path()
    
    dino = local_groundingdino_build_model(dino_model_args)
    checkpoint = torch.load(
        get_local_filepath(
            groundingdino_model_list[model_name]["model_url"],
            groundingdino_model_dir_name,
        ),
    )
    dino.load_state_dict(local_groundingdino_clean_state_dict(
        checkpoint['model']), strict=False)
    device = comfy.model_management.get_torch_device()
    dino.to(device=device)
    dino.eval()
    return dino


def list_groundingdino_model() -> list[str]:
    """Returns a list of keys from the groundingdino_model_list dictionary."""
    return list(groundingdino_model_list.keys())


def groundingdino_predict(
        dino_model: torch.nn.Module, 
        image: Image.Image, 
        prompt: str, 
        threshold: float
    ) -> torch.Tensor:
    """Processes an image using a DINO model to predict bounding boxes based on a prompt, returning the filtered bounding boxes as torch.Tensor."""
    def load_dino_image(image_pil: Image.Image) -> torch.Tensor:
        """Loads a dinosaur image by applying a series of transformations including resizing, converting to a PyTorch tensor, and normalizing pixel values. Returns the processed image as a torch.Tensor."""
        transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        image, _ = transform(image_pil, None)  # 3, h, w
        return image

    def get_grounding_output(model, image, caption, box_threshold):
        caption = caption.lower()
        caption = caption.strip()
        if not caption.endswith("."):
            caption = caption + "."
        device = comfy.model_management.get_torch_device()
        image = image.to(device)
        with torch.no_grad():
            outputs = model(image[None], captions=[caption])
        logits = outputs["pred_logits"].sigmoid()[0]  # (nq, 256)
        boxes = outputs["pred_boxes"][0]  # (nq, 4)
        # filter output
        logits_filt = logits.clone()
        boxes_filt = boxes.clone()
        filt_mask = logits_filt.max(dim=1)[0] > box_threshold
        logits_filt = logits_filt[filt_mask]  # num_filt, 256
        boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
        return boxes_filt.cpu()

    dino_image = load_dino_image(image.convert("RGB"))
    boxes_filt = get_grounding_output(
        dino_model, dino_image, prompt, threshold
    )
    H, W = image.size[1], image.size[0]
    for i in range(boxes_filt.size(0)):
        boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
        boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
        boxes_filt[i][2:] += boxes_filt[i][:2]
    return boxes_filt


def create_pil_output(image_np, masks, boxes_filt):
    output_masks, output_images = [], []
    boxes_filt = boxes_filt.numpy().astype(int) if boxes_filt is not None else None
    for mask in masks:
        output_masks.append(Image.fromarray(np.any(mask, axis=0)))
        image_np_copy = copy.deepcopy(image_np)
        image_np_copy[~np.any(mask, axis=0)] = np.array([0, 0, 0, 0])
        output_images.append(Image.fromarray(image_np_copy))
    return output_images, output_masks


def create_tensor_output(image_np: np.ndarray, masks: list[np.ndarray], boxes_filt: torch.Tensor | None) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Processes a NumPy array representing an image, a list of masks as NumPy arrays, and optionally a tensor of bounding boxes. Modifies the image array based on the masks, converts it to a PIL Image, and splits it into RGB channels. Returns a tuple of lists containing the RGB image tensors and mask tensors."""
    output_masks, output_images = [], []
    boxes_filt = boxes_filt.numpy().astype(int) if boxes_filt is not None else None
    for mask in masks:
        image_np_copy = copy.deepcopy(image_np)
        image_np_copy[~np.any(mask, axis=0)] = np.array([0, 0, 0, 0])
        output_image, output_mask = split_image_mask(
            Image.fromarray(image_np_copy))
        output_masks.append(output_mask)
        output_images.append(output_image)
    return (output_images, output_masks)


def split_image_mask(image: Image.Image) -> tuple[torch.Tensor, torch.Tensor]:
    """Converts an input PIL Image into RGB format, normalizes it, and creates a tensor. If an alpha channel exists, extracts and normalizes it into a tensor; otherwise, returns a tensor of zeros. Returns a tuple of the RGB image tensor and the mask tensor."""
    image_rgb = image.convert("RGB")
    image_rgb = np.array(image_rgb).astype(np.float32) / 255.0
    image_rgb = torch.from_numpy(image_rgb)[None,]
    if 'A' in image.getbands():
        mask = np.array(image.getchannel('A')).astype(np.float32) / 255.0
        mask = torch.from_numpy(mask)[None,]
    else:
        mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")
    return (image_rgb, mask)


def sam_segment(sam_model: Any, image: Any, boxes: torch.Tensor) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Processes a given image, a tensor of bounding boxes, and a SAM model to segment the image using the SAM model. Converts the image to a NumPy array, extracts the RGB channels, applies the bounding boxes transformation, predicts masks using the SAM model, and returns a tuple of lists containing the RGB image tensors and mask tensors. Handles the case where no bounding boxes are provided."""
    if boxes.shape[0] == 0:
        return None
    sam_is_hq = False
    # TODO: more elegant
    if hasattr(sam_model, 'model_name') and 'hq' in sam_model.model_name:
        sam_is_hq = True
    predictor = SamPredictorHQ(sam_model, sam_is_hq)
    image_np = np.array(image)
    image_np_rgb = image_np[..., :3]
    predictor.set_image(image_np_rgb)
    transformed_boxes = predictor.transform.apply_boxes_torch(
        boxes, image_np.shape[:2])
    sam_device = comfy.model_management.get_torch_device()
    masks, _, _ = predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes.to(sam_device),
        multimask_output=False)
    masks = masks.permute(1, 0, 2, 3).cpu().numpy()
    return create_tensor_output(image_np, masks, boxes)


class SAMModelLoader:
    @classmethod
    def INPUT_TYPES(cls) -> dict[str, tuple[list[str]]]:
        """Returns a dictionary with a key "model_name" mapped to a tuple containing a list of strings obtained from the function list_sam_model()"""
        return {
            "required": {
                "model_name": (list_sam_model(), ),
            }
        }
    CATEGORY = "segment_anything"
    FUNCTION = "main"
    RETURN_TYPES = ("SAM_MODEL", )

    def main(self, model_name: str) -> tuple[Any, ...]:
        """Loads a SAM (Semi-supervised Anomaly Detection Model) model specified by the model name, initializes it based on the retrieved checkpoint path, sets the appropriate device, and returns the loaded model."""
        sam_model = load_sam_model(model_name)
        return (sam_model, )


class GroundingDinoModelLoader:
    @classmethod
    def list_groundingdino_model() -> list[str]:
        """Returns a dictionary with a key "model_name" containing a list of keys from the groundingdino_model_list dictionary."""
        return {
            "required": {
                "model_name": (list_groundingdino_model(), ),
            }
        }
    CATEGORY = "segment_anything"
    FUNCTION = "main"
    RETURN_TYPES = ("GROUNDING_DINO_MODEL", )

    def main(self, model_name: str) -> tuple[local_groundingdino_build_model, ...]:
        """Loads a groundingdino model based on the provided model name, configuration settings, and model checkpoint. It adjusts the text encoder type if necessary, loads the model checkpoint, cleans the state dictionary keys, moves the model to the appropriate device, sets the model to evaluation mode, and returns the loaded model."""
        dino_model = load_groundingdino_model(model_name)
        return (dino_model, )


class GroundingDinoSAMSegment:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sam_model": ('SAM_MODEL', {}),
                "grounding_dino_model": ('GROUNDING_DINO_MODEL', {}),
                "image": ('IMAGE', {}),
                "prompt": ("STRING", {}),
                "threshold": ("FLOAT", {
                    "default": 0.3,
                    "min": 0,
                    "max": 1.0,
                    "step": 0.01
                }),
            }
        }
    CATEGORY = "segment_anything"
    FUNCTION = "main"
    RETURN_TYPES = ("IMAGE", "MASK")

    def main(self, grounding_dino_model: torch.nn.Module, sam_model: Any, image: list[np.ndarray], prompt: str, threshold: float) -> tuple[torch.Tensor, torch.Tensor]:
        """Processes a list of NumPy arrays representing images using a DINO model to predict bounding boxes based on a prompt, and then segments the images using a SAM model. Returns a tuple of concatenated RGB image tensors and mask tensors. Handles cases where no bounding boxes are provided or no images are processed."""
        res_images = []
        res_masks = []
        for item in image:
            item = Image.fromarray(
                np.clip(255. * item.cpu().numpy(), 0, 255).astype(np.uint8)).convert('RGBA')
            boxes = groundingdino_predict(
                grounding_dino_model,
                item,
                prompt,
                threshold
            )
            if boxes.shape[0] == 0:
                break
            (images, masks) = sam_segment(
                sam_model,
                item,
                boxes
            )
            res_images.extend(images)
            res_masks.extend(masks)
        if len(res_images) == 0:
            _, height, width, _ = image.size()
            empty_mask = torch.zeros((1, height, width), dtype=torch.uint8, device="cpu")
            return (empty_mask, empty_mask)
        return (torch.cat(res_images, dim=0), torch.cat(res_masks, dim=0))


class InvertMask:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
            }
        }
    CATEGORY = "segment_anything"
    FUNCTION = "main"
    RETURN_TYPES = ("MASK",)

    def main(self, mask):
        out = 1.0 - mask
        return (out,)

class IsMaskEmptyNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mask": ("MASK",),
            },
        }
    RETURN_TYPES = ["NUMBER"]
    RETURN_NAMES = ["boolean_number"]

    FUNCTION = "main"
    CATEGORY = "segment_anything"

    def main(self, mask):
        return (torch.all(mask == 0).int().item(), )