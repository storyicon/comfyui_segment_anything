import torch
import numpy as np
from PIL import Image

from ..utils.collection import to_tensor , split_image_mask , create_tensor_output, split_captions
from ..libs.sam_hq.predictor import SamPredictorHQ
from ..libs.local_groundingdino.datasets import transforms as T

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

def sam_segment(
    sam_model,
    image,
    boxes,
    multimask,
    device
):  
    if boxes.shape[0] == 0:
        return None
    
    sam_is_hq = hasattr(sam_model, 'model_name') and 'hq' in getattr(sam_model, 'model_name', '').lower()

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
