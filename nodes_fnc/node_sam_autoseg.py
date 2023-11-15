import torch
import numpy as np
from PIL import Image

from ..libs.sam_hq.automatic import SamAutomaticMaskGeneratorHQ
from ..utils.collection  import to_tensor
from segment_anything import SamAutomaticMaskGenerator


def sam_auto_segmentationHQ(sam_model, image, points_per_side, min_mask_region_area, stability_score_thresh, box_nms_thresh,crop_n_layers,crop_nms_thresh, crop_overlap_ratio,device):

    """
    SamAutomaticMaskGeneratorHQ possible options:

    model (Sam): The SAM model to use for mask prediction.
    points_per_side (int or None): The number of points to be sampled along one side of the image. The total number of points is points_per_side**2. If None, 'point_grids' must provide explicit point sampling.
    points_per_batch (int): Sets the number of points run simultaneously by the model. Higher numbers may be faster but use more GPU memory.
    pred_iou_thresh (float): A filtering threshold in [0,1], using the model's predicted mask quality.
    stability_score_thresh (float): A filtering threshold in [0,1], using the stability of the mask under changes to the cutoff used to binarize the model's mask predictions.
    stability_score_offset (float): The amount to shift the cutoff when calculated the stability score.
    box_nms_thresh (float): The box IoU cutoff used by non-maximal suppression to filter duplicate masks.
    crop_n_layers (int): If >0, mask prediction will be run again on crops of the image. Sets the number of layers to run, where each layer has 2**i_layer number of image crops.
    crop_nms_thresh (float): The box IoU cutoff used by non-maximal suppression to filter duplicate masks between different crops.
    crop_overlap_ratio (float): Sets the degree to which crops overlap. In the first crop layer, crops will overlap by this fraction of the image length. Later layers with more crops scale down this overlap.
    crop_n_points_downscale_factor (int): The number of points-per-side sampled in layer n is scaled down by crop_n_points_downscale_factor**n.
    point_grids (list(np.ndarray) or None): A list over explicit grids  of points used for sampling, normalized to [0,1]. The nth grid in the  list is used in the nth crop layer. Exclusive with points_per_side.
    min_mask_region_area (int): If >0, postprocessing will be applied to remove disconnected regions and holes in masks with area smaller than min_mask_region_area. Requires opencv.
    output_mode (str): The form masks are returned in. Can be 'binary_mask', 'uncompressed_rle', or 'coco_rle'. 'coco_rle' requires pycocotools. For large resolutions, 'binary_mask' may consume large amounts of memory.
    """

    image_rgb = image.convert("RGB") # to RGB in any case 4 dimension are not supported
    image_np = np.array(image_rgb) # image (np.ndarray): The image to generate masks for, in HWC uint8 format.

    sam_is_hq = hasattr(sam_model, 'model_name') and 'hq' in getattr(sam_model, 'model_name', '').lower()

    # use other proccess for HQ
    if sam_is_hq is True:

        sam = SamAutomaticMaskGeneratorHQ(
            sam_model,
            stability_score_thresh=stability_score_thresh, 
            points_per_side=points_per_side, 
            min_mask_region_area=min_mask_region_area,
            box_nms_thresh=box_nms_thresh,
            crop_n_layers=crop_n_layers, 
            crop_nms_thresh=crop_nms_thresh,
            crop_overlap_ratio=crop_overlap_ratio,

        ) # build detector

        binary_masks_list = SamAutomaticMaskGeneratorHQ.generate(sam, image_np, multimask_output = False) # generate binary_masks
    else:
        sam = SamAutomaticMaskGenerator(sam_model, points_per_side=points_per_side , crop_n_layers=1, min_mask_region_area=min_mask_region_area) # build detector
        binary_masks_list = SamAutomaticMaskGenerator.generate(sam, image_np) # generate binary_masks
    #
    response_images, response_masks = [],[] # prepare output
    #
    for binary_mask in binary_masks_list:

        # possible binary_masks_list responses: 
        """
        area = binary_mask['area']
        bbox = binary_mask['bbox']
        predicted_iou = binary_mask['predicted_iou']
        point_coords = binary_mask['point_coords']
        stability_score = binary_mask['stability_score']
        crop_box = binary_mask['crop_box']
        """
        #
        segmentation = binary_mask['segmentation']
        # Mask
        binary_mask_np = np.array(segmentation)
        binary_mask_uint8 = binary_mask_np.astype(np.uint8) * 255
        binary_mask_tensor = torch.from_numpy(binary_mask_uint8).float().unsqueeze(0).unsqueeze(0) # do tensor [B,C,H,W]
        response_masks.append(binary_mask_tensor)
        # Images
        image_rgb_copy = image_rgb.copy()
        #
        background = Image.new("RGBA", image_rgb_copy.size, (0, 0, 0, 255))
        binary_mask_pil = Image.fromarray(binary_mask_uint8) # to PIL 
        image_rgb_copy.putalpha(binary_mask_pil) # apply Mask
        background.paste(image_rgb_copy, (0, 0), binary_mask_pil)
        image_rgb_copy = background
        #
        rgb_tensor = to_tensor(image_rgb_copy).unsqueeze(0)  # add batch_size 1
        rgb_tensor = rgb_tensor.permute(0, 2, 3, 1) # [B,C,H,W] to [B,H,W,C]
        response_images.append(rgb_tensor)

    return response_images, response_masks
