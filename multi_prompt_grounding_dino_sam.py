import torch
import numpy as np
from PIL import Image
from typing import List, Tuple, Union

# Import the necessary classes from comfyui_segment_anything
from .node import GroundingDinoSAMSegment, load_groundingdino_model, load_sam_model

class MultiPromptGroundingDinoSAMSegment(GroundingDinoSAMSegment):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "prompts": ("STRING", {"forceInput": True}),  # This will be a list from BatchPromptSplitter
                "box_threshold": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01}),
                "text_threshold": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01}),
                "iou_threshold": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "segment_multi_prompt"
    CATEGORY = "image/segmentation"

    def __init__(self):
        super().__init__()

    def segment_multi_prompt(self, image: torch.Tensor, prompts: List[str], box_threshold: float, text_threshold: float, iou_threshold: float) -> Tuple[torch.Tensor]:
        # Convert the image tensor to a PIL Image
        pil_image = Image.fromarray((image.squeeze().cpu().numpy() * 255).astype(np.uint8))

        # Initialize models if not already done
        if self.groundingdino_model is None:
            self.groundingdino_model = load_groundingdino_model()
        if self.sam_model is None:
            self.sam_model = load_sam_model()

        combined_mask = torch.zeros_like(image[0, 0], dtype=torch.float32)

        for prompt in prompts:
            # Use the existing segment method for each prompt
            mask = super().segment(image, prompt, box_threshold, text_threshold, iou_threshold)[0]
            combined_mask = torch.max(combined_mask, mask)

        return (combined_mask,)

NODE_CLASS_MAPPINGS = {
    "MultiPromptGroundingDinoSAMSegment": MultiPromptGroundingDinoSAMSegment
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MultiPromptGroundingDinoSAMSegment": "Multi-Prompt GroundingDINO SAM Segmentation"
}