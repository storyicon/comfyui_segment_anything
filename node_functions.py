import os
import copy
import torch
import json
from torchvision.transforms import ToTensor, transforms
import numpy as np
from PIL import Image
import logging
from torch.hub import download_url_to_file
from urllib.parse import urlparse
import folder_paths
from .sam_hq.predictor import SamPredictorHQ
from .sam_hq.automatic import SamAutomaticMaskGeneratorHQ
from .sam_hq.build_sam_hq import sam_model_registry
from .local_groundingdino.datasets import transforms as T
from .local_groundingdino.util.utils import clean_state_dict as local_groundingdino_clean_state_dict
from .local_groundingdino.util.slconfig import SLConfig as local_groundingdino_SLConfig
from .local_groundingdino.models import build_model as local_groundingdino_build_model
from segment_anything import SamAutomaticMaskGenerator


logger = logging.getLogger('comfyui_segment_anything')
to_tensor = ToTensor()

sam_model_dir = os.path.join(folder_paths.models_dir, "sams")
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
    }
}

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


def list_files(dirpath, extensions=[]):
    return [f for f in os.listdir(dirpath) if os.path.isfile(os.path.join(dirpath, f)) and f.split('.')[-1] in extensions]

def list_sam_model():
    return list(sam_model_list.keys())

def load_sam_model(model_name ):
    sam_checkpoint_path = get_local_filepath(
        sam_model_list[model_name]["model_url"], sam_model_dir)
    model_file_name = os.path.basename(sam_checkpoint_path)
    model_type = model_file_name.split('.')[0]
    if 'hq' not in model_type and 'mobile' not in model_type:
        model_type = '_'.join(model_type.split('_')[:-1])
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint_path)
    sam.model_name = model_file_name
    return sam

def get_local_filepath(url, dirname, local_file_name=None):
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    if not local_file_name:
        parsed_url = urlparse(url)
        local_file_name = os.path.basename(parsed_url.path)
    destination = os.path.join(dirname, local_file_name)
    if not os.path.exists(destination):
        logging.warn(f'downloading {url} to {destination}')
        download_url_to_file(url, destination)
    return destination

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

def split_captions(input_string):
        # Split the input string by the pipe character and strip whitespace
        captions = input_string.split('|')
        processed_captions = []
        for caption in captions:
            # Stripping whitespace and converting to lower case
            cleaned_caption = caption.strip().lower()
            # Appending a terminal '.' if it doesn't already end with one
            if not cleaned_caption.endswith('.'):
                cleaned_caption += '.'
            processed_captions.append(cleaned_caption)

        return processed_captions

def get_grounding_output(model, image, caption, box_threshold, optimize_prompt_for_dino, device):
    # send image to device where to proccess 
    image = image.to(device)
    
    #check if we want prompt optmiziation = replace sd ","" with dino "." 
    if optimize_prompt_for_dino is not False:
        if caption.endswith(","):
          caption = caption[:-1]
        caption = caption.replace(",", ".")

    # idea @ttulttul https://github.com/storyicon/comfyui_segment_anything/pull/16
    if "|" in caption:
        #
        captions = split_captions(caption)
        all_boxes = []
        for caption in captions:
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
            all_boxes.append(boxes_filt.to(device))

        # Concatenate all the boxes along the 0 dimension and return.
        boxes_filt_concat = torch.cat(all_boxes, dim=0)
        return boxes_filt_concat.to(device)

    else:
        caption = caption.lower()
        caption = caption.strip()
        if not caption.endswith("."):
            caption = caption + "."
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
        return boxes_filt.to(device)

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

def groundingdino_predict(
    dino_model,
    image,
    prompt,
    box_threshold,
    optimize_prompt_for_dino,
    device
):
    dino_image = load_dino_image(image.convert("RGB"))
    boxes_filt = get_grounding_output(dino_model, dino_image, prompt, box_threshold,optimize_prompt_for_dino, device)
    H, W = image.size[1], image.size[0]
    for i in range(boxes_filt.size(0)):
        boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H]).to(device)
        boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
        boxes_filt[i][2:] += boxes_filt[i][:2]
    return boxes_filt.to(device)

def create_pil_output(image_np, masks, boxes_filt):
    output_masks, output_images = [], []
    boxes_filt = boxes_filt.numpy().astype(int) if boxes_filt is not None else None
    for mask in masks:
        output_masks.append(Image.fromarray(np.any(mask, axis=0)))
        image_np_copy = copy.deepcopy(image_np)
        image_np_copy[~np.any(mask, axis=0)] = np.array([0, 0, 0, 0])
        output_images.append(Image.fromarray(image_np_copy))
    return output_images, output_masks

def create_tensor_output(image_np, masks, boxes_filt,device):
    output_masks, output_images = [], []
    boxes_filt = boxes_filt.cpu().numpy().astype(int) if boxes_filt is not None else None
    for mask in masks:
        image_np_copy = copy.deepcopy(image_np)
        image_np_copy[~np.any(mask, axis=0)] = np.array([0, 0, 0, 0])
        output_image, output_mask = split_image_mask(Image.fromarray(image_np_copy),device)
        output_masks.append(output_mask)
        output_images.append(output_image)
    return (output_images, output_masks)

def split_image_mask(image, device):
    image_rgb = image.convert("RGB")
    image_rgb = np.array(image_rgb).astype(np.float32) / 255.0
    image_rgb = torch.from_numpy(image_rgb)[None,]
    if 'A' in image.getbands():
        mask = np.array(image.getchannel('A')).astype(np.float32) / 255.0
        mask = torch.from_numpy(mask)[None,]
    else:
        mask = torch.zeros((64, 64), dtype=torch.float32, device=device)
    return (image_rgb, mask)

def sam_segment(
    sam_model,
    image,
    boxes,
    multimask,
    device
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
    masks, _, _ = predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes.to(device),
        multimask_output=False
        )
    
    if multimask is not False:
        output_images, output_masks = [], []
        for batch_index in range(masks.size(0)):
            mask_np =  masks[batch_index].permute( 1, 2, 0).cpu().numpy()# H.W.C
            image_with_alpha = Image.fromarray(np.concatenate((image_np_rgb, mask_np * 255), axis=2).astype(np.uint8), 'RGBA')
            _, msk = split_image_mask(image_with_alpha,device)
            r, g, b, a = image_with_alpha.split()

            black_image = Image.new("RGB", image.size, (0, 0, 0))
            black_image.paste(image_with_alpha, mask=image_with_alpha.split()[3])

            rgb_ts = to_tensor(black_image)
            rgb_ts = rgb_ts.unsqueeze(0)
            rgb_ts = rgb_ts.permute(0, 2, 3, 1)

            output_images.append(rgb_ts)
            output_masks.append(msk)
                        
        return (output_images, output_masks)
    else:
        masks = masks.permute(1, 0, 2, 3).cpu().numpy()
        return create_tensor_output(image_np, masks, boxes, device)



def sam_auto_segmentationHQ(sam_model, image, points_per_side, min_mask_region_area, device):

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

    sam_is_hq = False
    # TODO: more elegant
    if hasattr(sam_model, 'model_name') and 'hq' in sam_model.model_name:
        sam_is_hq = True

    # use other proccess for HQ
    if sam_is_hq is True:
        sam = SamAutomaticMaskGeneratorHQ(sam_model, points_per_side=points_per_side, crop_n_layers=1, min_mask_region_area=min_mask_region_area) # build detector
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

        background = Image.new("RGBA", image_rgb_copy.size, (0, 0, 0, 255))
        binary_mask_pil = Image.fromarray(binary_mask_uint8) # to PIL 
        image_rgb_copy.putalpha(binary_mask_pil) # apply Mask
        background.paste(image_rgb_copy, (0, 0), binary_mask_pil)
        image_rgb_copy = background
        #
        compose_tensor = transforms.Compose([
            transforms.ToTensor(),  # convert to tensor
            #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Optional: Normalisierung
        ])
        #
        rgb_tensor = compose_tensor(image_rgb_copy).unsqueeze(0)  # add batch_size 1
        rgb_tensor = rgb_tensor.permute(0, 2, 3, 1) # [B,C,H,W] to [B,H,W,C]
        response_images.append(rgb_tensor)

    return response_images, response_masks