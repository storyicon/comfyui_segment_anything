import os
import sys
sys.path.append(
    os.path.dirname(os.path.abspath(__file__))
)

import copy
import torch
import numpy as np
import cv2
import re
import json

from PIL import Image, ImageDraw, ImageFont
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
from local_groundingdino.util.utils import get_phrases_from_posmap

import glob

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

def get_bert_base_uncased_model_path():
    comfy_bert_model_base = os.path.join(folder_paths.models_dir, 'bert-base-uncased')
    if glob.glob(os.path.join(comfy_bert_model_base, '**/model.safetensors'), recursive=True):
        print('grounding-dino is using models/bert-base-uncased')
        return comfy_bert_model_base
    return 'bert-base-uncased'

def list_files(dirpath, extensions=[]):
    return [f for f in os.listdir(dirpath) if os.path.isfile(os.path.join(dirpath, f)) and f.split('.')[-1] in extensions]


def list_sam_model():
    return list(sam_model_list.keys())


def load_sam_model(model_name):
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


def get_local_filepath(url, dirname, local_file_name=None):
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


def load_groundingdino_model(model_name):
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


def list_groundingdino_model():
    return list(groundingdino_model_list.keys())


def groundingdino_predict(
    dino_model,
    image,
    prompt,
    threshold
):
    def load_dino_image(image_pil):
        transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        image, _ = transform(image_pil, None)  # 3, h, w
        return image

    def replace_commas_with_dots(input_string):
        if ',' in input_string or '，' in input_string:
            modified_string = input_string.replace(',', '.').replace('，', '.')
            return modified_string
        else:
            return input_string

    def get_grounding_output(model, image, caption, box_threshold):
        caption = caption.lower()
        caption = caption.strip()
        caption = replace_commas_with_dots(caption)
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

        # get phrase
        tokenlizer = model.tokenizer
        tokenized = tokenlizer(caption)
        # print("grounding",caption)
        # build pred
        pred_phrases = []
        for logit, box in zip(logits_filt, boxes_filt):
            pred_phrase = get_phrases_from_posmap(
                logit > box_threshold, tokenized, tokenlizer
            )
            pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")

        return boxes_filt.cpu(), pred_phrases

    dino_image = load_dino_image(image.convert("RGB"))
    boxes_filt, pred_phrases = get_grounding_output(
        dino_model, dino_image, prompt, threshold
    )
    H, W = image.size[1], image.size[0]
    for i in range(boxes_filt.size(0)):
        boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
        boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
        boxes_filt[i][2:] += boxes_filt[i][:2]
    return boxes_filt, pred_phrases


def create_pil_output(image_np, masks, boxes_filt):
    output_masks, output_images = [], []
    boxes_filt = boxes_filt.numpy().astype(int) if boxes_filt is not None else None
    for mask in masks:
        output_masks.append(Image.fromarray(np.any(mask, axis=0)))
        image_np_copy = copy.deepcopy(image_np)
        image_np_copy[~np.any(mask, axis=0)] = np.array([0, 0, 0, 0])
        output_images.append(Image.fromarray(image_np_copy))
    return output_images, output_masks


def create_tensor_output(image_np, masks, boxes_filt):
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


def split_image_mask(image):
    image_rgb = image.convert("RGB")
    image_rgb = np.array(image_rgb).astype(np.float32) / 255.0
    image_rgb = torch.from_numpy(image_rgb)[None,]
    if 'A' in image.getbands():
        mask = np.array(image.getchannel('A')).astype(np.float32) / 255.0
        mask = torch.from_numpy(mask)[None,]
    else:
        mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")
    return (image_rgb, mask)

def sam_segment(
    sam_model,
    image,
    boxes
):
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
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (list_sam_model(), ),
            }
        }
    CATEGORY = "segment_anything"
    FUNCTION = "main"
    RETURN_TYPES = ("SAM_MODEL", )

    def main(self, model_name):
        sam_model = load_sam_model(model_name)
        return (sam_model, )


class GroundingDinoModelLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (list_groundingdino_model(), ),
            }
        }
    CATEGORY = "segment_anything"
    FUNCTION = "main"
    RETURN_TYPES = ("GROUNDING_DINO_MODEL", )

    def main(self, model_name):
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

    def main(self, grounding_dino_model, sam_model, image, prompt, threshold):
        res_images = []
        res_masks = []
        for item in image:
            item = Image.fromarray(
                np.clip(255. * item.cpu().numpy(), 0, 255).astype(np.uint8)).convert('RGBA')
            boxes, pred_phrases = groundingdino_predict(
                grounding_dino_model, item, prompt, threshold
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


def plot_boxes_to_image(image_pil, tgt):

    image_np = np.array(image_pil)

    H, W = tgt["size"]
    boxes = tgt["boxes"]
    labels = tgt["labels"]

    res_mask = []
    res_image = []

    font_scale = 1
    box_color = (255, 0, 0)
    text_color = (255, 255, 255)
    image_np = image_np[..., :3]

    # Make a copy of the image to avoid modifying the original image
    image_with_boxes = np.copy(image_np)

    labelme_data = {
        "version": "4.5.6",
        "flags": {},
        "shapes": [],
        "imagePath": None,
        "imageData": None,
        "imageHeight": H,
        "imageWidth": W,
    }
    for box, label in zip(boxes, labels):
        x1, y1, x2, y2 = box
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        # save labelme json
        points = [[x1, y1], [x2, y2]]
        shape = {
            "label": label.lower().split("(")[0],
            "points": points,
            "group_id": None,
            "shape_type": "rectangle",
            "flags": {},
        }
        labelme_data["shapes"].append(shape)

        # change lable
        label_name = label.lower().split("(")[0]
        threshold = label.lower().split("(")[1].split(")")[0]
        label = label_name+":"+threshold
        shape["threshold"] = threshold

        # Draw rectangle on the copied image
        cv2.rectangle(
            image_with_boxes, (int(x1), int(y1)), (int(x2), int(y2)), box_color, 3
        )

        # Draw label on the copied image
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)[0]
        label_ymin = max(y1, label_size[1] + 10)
        cv2.rectangle(
            image_with_boxes,
            (x1, y1 - label_size[1] - 10),
            (x1 + label_size[0], y1),
            box_color,
            -1,
        )
        cv2.putText(
            image_with_boxes,
            label,
            (x1, y1 - 7),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            text_color,
            2,
            cv2.LINE_AA,
            bottomLeftOrigin=False,
        )

        # Draw mask
        mask = np.zeros((H, W, 1), dtype=np.uint8)
        cv2.rectangle(mask, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 255), -1)
        mask_tensor = torch.from_numpy(mask).permute(2, 0, 1).float() / 255.0
        res_mask.append(mask_tensor)

    if len(res_mask) == 0:
        mask = np.zeros((H, W, 1), dtype=np.uint8)
        mask_tensor = torch.from_numpy(mask).permute(2, 0, 1).float() / 255.0
        res_mask.append(mask_tensor)

    # Convert the modified image to a torch tensor
    image_with_boxes_tensor = torch.from_numpy(image_with_boxes.astype(np.float32) / 255.0)
    image_with_boxes_tensor = torch.unsqueeze(image_with_boxes_tensor, 0)
    res_image.append(image_with_boxes_tensor)

    return res_image, res_mask, labelme_data


class GroundingDinoDetect:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "grounding_dino_model": ("GROUNDING_DINO_MODEL", {}),
                "image": ("IMAGE", {}),
                "prompt": ("STRING", {}),
                "threshold": (
                    "FLOAT",
                    {"default": 0.3, "min": 0, "max": 1.0, "step": 0.01},
                ),
                "only_output_result": (["enable", "disable"],),
            }
        }

    CATEGORY = "segment_anything"
    FUNCTION = "main"
    RETURN_TYPES = ("IMAGE", "MASK","JSON",)

    def main(
        self,
        grounding_dino_model,
        image,
        prompt,
        threshold,
        only_output_result,
    ):
        res_images = []
        res_masks = []
        res_labels = []

        count =1
        for item in image:
            count+=1
            image_pil = Image.fromarray(
                np.clip(255.0 * item.cpu().numpy(), 0, 255).astype(np.uint8)
            ).convert("RGBA")
            boxes, pred_phrases = groundingdino_predict(
                grounding_dino_model, image_pil, prompt, threshold
            )
            if boxes.shape[0] == 0 and only_output_result == "enable":
                break

            size = image_pil.size
            pred_dict = {
                "boxes": boxes,
                "size": [size[1], size[0]],
                "labels": pred_phrases,
            }

            image_tensor, mask_tensor, labelme_data = plot_boxes_to_image(
                image_pil, pred_dict
            )

            res_images.extend(image_tensor)
            res_masks.extend(mask_tensor)
            res_labels.append(labelme_data)

            if len(res_images) == 0:
                res_images.extend(item)
            if len(res_masks) == 0:
                mask = np.zeros((height, width, 1), dtype=np.uint8)
                empty_mask = torch.from_numpy(mask).permute(2, 0, 1).float() / 255.0
                res_masks.extend(empty_mask)

        return (
            torch.cat(res_images, dim=0),
            torch.cat(res_masks, dim=0),
            res_labels,
        )
