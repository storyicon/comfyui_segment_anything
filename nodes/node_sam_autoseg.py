import torch
import numpy as np
from PIL import Image

import comfy.model_management
from ..nodes_fnc.node_sam_autoseg import sam_auto_segmentationHQ

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
                "stability_score_thresh":("FLOAT", {
                    "default": 0.95,
                    "min": 0.01,
                    "max": 0.98,
                    "step": 0.01
                }),
                "box_nms_thresh":("FLOAT", {
                    "default": 0.7,
                    "min": 0.01,

                    "step": 0.01
                }),
                "crop_n_layers":("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 2,
                    "step": 1
                }),
                "crop_nms_thresh":("FLOAT", {
                    "default": 0,
                    "min": 0,
                    
                    "step": 0.01
                }),
                "crop_overlap_ratio":("FLOAT", {
                    "default": 0.34,
                    "min": -0.99,
                    "max": 1,
                    "step": 0.01
                }),

                "dedicated_device": (["Auto", "CPU", "GPU"], ),
            },
        }
    CATEGORY = "segment_anything"
    FUNCTION = "main"
    RETURN_TYPES = ("IMAGE", "MASK")

    def main(self, sam_model, image, points_per_side, stability_score_thresh ,min_mask_region_area, box_nms_thresh, crop_n_layers, crop_nms_thresh , crop_overlap_ratio, dedicated_device="Auto"):
        """
        SamAutomaticMaskGeneratorHQ possible options:

        (+) model (Sam): The SAM model to use for mask prediction.
        (+) points_per_side (int or None): The number of points to be sampled along one side of the image. The total number of points is points_per_side**2. If None, 'point_grids' must provide explicit point sampling.
        (-) points_per_batch (int): Sets the number of points run simultaneously by the model. Higher numbers may be faster but use more GPU memory.
        (-) pred_iou_thresh (float): A filtering threshold in [0,1], using the model's predicted mask quality.
        (+) stability_score_thresh (float): A filtering threshold in [0,1], using the stability of the mask under changes to the cutoff used to binarize the model's mask predictions.
        (-) stability_score_offset (float): The amount to shift the cutoff when calculated the stability score.
        (+) box_nms_thresh (float): The box IoU cutoff used by non-maximal suppression to filter duplicate masks.
        (discussion needed)(+) crop_n_layers (int): If >0, mask prediction will be run again on crops of the image. Sets the number of layers to run, where each layer has 2**i_layer number of image crops.  
        (discussion needed)(+) crop_nms_thresh (float): The box IoU cutoff used by non-maximal suppression to filter duplicate masks between different crops.     
        (discussion needed)(+) crop_overlap_ratio (float): Sets the degree to which crops overlap. In the first crop layer, crops will overlap by this fraction of the image length. Later layers with more crops scale down this overlap.
        (discussion needed)(-) crop_n_points_downscale_factor (int): The number of points-per-side sampled in layer n is scaled down by crop_n_points_downscale_factor**n.       
        (discussion needed)(-) point_grids (list(np.ndarray) or None): A list over explicit grids  of points used for sampling, normalized to [0,1]. The nth grid in the  list is used in the nth crop layer. Exclusive with points_per_side.
        (+) min_mask_region_area (int): If >0, postprocessing will be applied to remove disconnected regions and holes in masks with area smaller than min_mask_region_area. Requires opencv.
        """
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
            images, masks = sam_auto_segmentationHQ(
                sam_model=sam_model, 
                image=item, 
                points_per_side=points_per_side, 
                min_mask_region_area=min_mask_region_area, 
                stability_score_thresh=stability_score_thresh,
                box_nms_thresh=box_nms_thresh,
                crop_n_layers=crop_n_layers,
                crop_nms_thresh=crop_nms_thresh,
                crop_overlap_ratio=crop_overlap_ratio,
                device=device
            )

        # if nothing was detected just send simple input image and empty mask
        if not images:
            print("\033[1;32m(segment-anything)\033[0m The tensor 'boxes' is empty. No elements were found in the image search.")
            images.append(image)
            masks.append(empty_mask)

        # generate output
        res_images = torch.cat(images, dim=0)
        res_masks = torch.cat(masks, dim=0)
        return (res_images, res_masks, )

