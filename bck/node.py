import torch
import numpy as np
from PIL import Image
import folder_paths
import comfy.model_management
from .node_functions import *

"""
class SAMModelLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (list_sam_model(), ),
            },
        }
    CATEGORY = "segment_anything"
    FUNCTION = "main"
    RETURN_TYPES = ("SAM_MODEL", )

    def main(self, model_name, use_cpu=False):
        sam_model = load_sam_model(model_name)
        return (sam_model, )
"""

class GroundingDinoModelLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (list_groundingdino_model(), ),
            },
        }
    CATEGORY = "segment_anything"
    FUNCTION = "main"
    RETURN_TYPES = ("GROUNDING_DINO_MODEL", )

    def main(self, model_name):
        dino_model = load_groundingdino_model(model_name)
        return (dino_model, )

"""
class GroundingDinoSAMSegment:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sam_model": ('SAM_MODEL', {}),
                "grounding_dino_model": ('GROUNDING_DINO_MODEL', {}),
                "image": ('IMAGE', {}),
                "prompt": ("STRING", {"default": "arms, legs, eyes, hair, head","multiline": True}),
                "box_threshold": ("FLOAT", {
                    "default": 0.3,
                    "min": 0,
                    "max": 1.0,
                    "step": 0.01
                }),
                "multimask": ('BOOLEAN', {"default":False}),
                "dedicated_device": (["Auto", "CPU", "GPU"], ),
                "optimize_prompt_for_dino":('BOOLEAN', {"default":False})
            },
        }
    CATEGORY = "segment_anything"
    FUNCTION = "main"
    RETURN_TYPES = ("IMAGE", "MASK")

    def main(self, grounding_dino_model, sam_model, image, prompt, box_threshold,optimize_prompt_for_dino=False,multimask=False, dedicated_device="Auto"):
        #
        device_mapping = {
            "Auto": comfy.model_management.get_torch_device(),
            "CPU": torch.device("cpu"),
            "GPU": torch.device("cuda")
        }
        device = device_mapping.get(dedicated_device)
        #
        # send model to selected device 
        grounding_dino_model.to(device)
        grounding_dino_model.eval()
        sam_model.to(device)
        sam_model.eval()
        #
        # in case sam or dino dont find anything, return blank mask and original image
        img_batch, img_height, img_width, img_channel = image.shape        # get original image dimensions 
        empty_mask = torch.zeros((1, 1, img_height, img_width), dtype=torch.float32) # [B,C,H,W]
        empty_mask = empty_mask / 255.0
        #
        # empty output
        res_images = []
        res_masks = []
        #
        detection_errors = False
        #
        for item in image:
            item = Image.fromarray(np.clip(255. * item.cpu().numpy(), 0, 255).astype(np.uint8)).convert('RGBA')
            # run dino for prompt guides segmentation
            boxes = groundingdino_predict(
                grounding_dino_model,
                item,
                prompt,
                box_threshold,
                optimize_prompt_for_dino,
                device
            )

            # abboart if nothing is detected and set detection_errors
            if boxes.numel() == 0:
                detection_errors = True
                break
            # create detailed masks with SAM depending on boxes found by dino
            (images, masks) = sam_segment(
                sam_model,
                item,
                boxes,
                multimask,
                device
            )
            # add results to output
            res_images.extend(images)
            res_masks.extend(masks)
        
        # if nothing was detected just send simple input image and empty mask
        if detection_errors is not False:
            print("\033[1;32m(segment-anything)\033[0m The tensor 'boxes' is empty. No elements were found in the image search.")
            res_images.append(image)
            res_masks.append(empty_mask)

        # generate output
        res_images = torch.cat(res_images, dim=0)
        res_masks = torch.cat(res_masks, dim=0)
        return (res_images, res_masks, )

        """

"""


class SAMAutoSegment:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sam_model": ('SAM_MODEL', {}),
                "image": ('IMAGE', {}),
                "points_per_side":("INT", {
                    "default": 32,
                    "min": 1,
                    "max": 5120,
                    "step": 1
                }),
                "min_mask_region_area":("INT", {
                    "default": 16,
                    "min": 0,
                    "max": 5120,
                    "step": 1
                }),
                "dedicated_device": (["Auto", "CPU", "GPU"], ),
            },
        }
    CATEGORY = "segment_anything"
    FUNCTION = "main"
    RETURN_TYPES = ("IMAGE", "MASK")

    def main(self, sam_model, image, points_per_side, min_mask_region_area, dedicated_device="Auto"):
        #
        device_mapping = {
            "Auto": comfy.model_management.get_torch_device(),
            "CPU": torch.device("cpu"),
            "GPU": torch.device("cuda")
        }
        device = device_mapping.get(dedicated_device)
        #
        # send model to selected device 
        sam_model.to(device)
        sam_model.eval()
        #
        # in case sam or dino dont find anything, return blank mask and original image
        img_batch, img_height, img_width, img_channel = image.shape        # get original image dimensions 
        empty_mask = torch.zeros((1, 1, img_height, img_width), dtype=torch.float32) # [B,C,H,W]
        empty_mask = empty_mask / 255.0
        
        # prepare Output
        images, masks = [], []
        
        
        for item in image:
            item = Image.fromarray(np.clip(255. * item.cpu().numpy(), 0, 255).astype(np.uint8)).convert('RGBA')
            # create detailed masks with SAM
            images, masks = sam_auto_segmentationHQ( sam_model, item, points_per_side, min_mask_region_area, device)

        # if nothing was detected just send simple input image and empty mask
        if not images:
            print("\033[1;32m(segment-anything)\033[0m The tensor 'boxes' is empty. No elements were found in the image search.")
            images.append(image)
            masks.append(empty_mask)

        # generate output
        res_images = torch.cat(images, dim=0)
        res_masks = torch.cat(masks, dim=0)
        return (res_images, res_masks, )





class BatchSelector:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ('IMAGE', {}),
                "mask": ('MASK', {}),
                "batch_select": ("FLOAT", {
                    "default": 1,
                    "min": 1,
                    "max": 1000000,
                    "step": 1
                }),
            }
        }
    CATEGORY = "segment_anything"
    FUNCTION = "main"
    RETURN_TYPES = ("IMAGE","MASK",)

    def main(self, image, mask, batch_select):
        selector = round(batch_select-1)
        selected_image = image[selector]
        selected_image = selected_image.unsqueeze(0)
        selected_mask = mask[selector]
        return (selected_image, selected_mask, )

"""