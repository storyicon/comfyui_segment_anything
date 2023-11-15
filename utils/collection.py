import os
import copy
import torch
import numpy as np
from PIL import Image
from urllib.parse import urlparse
import logging
logger = logging.getLogger('comfyui_segment_anything')

import folder_paths

from torchvision.transforms import ToTensor
from torch.hub import download_url_to_file


to_tensor = ToTensor()


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




# by @ttulttul
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
