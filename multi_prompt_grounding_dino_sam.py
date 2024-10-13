from .node import *
import torch

class MultiPromptGroundingDinoSAMSegment(GroundingDinoSAMSegment):
    @classmethod
    def INPUT_TYPES(cls):
        base_input_types = GroundingDinoSAMSegment.INPUT_TYPES()
        base_input_types["required"].pop("prompt", None)  # Remove single prompt
        base_input_types["required"]["prompts"] = ("STRING", {"forceInput": True})  # Add multi-prompt input
        return base_input_types

    CATEGORY = "segment_anything"
    FUNCTION = "segment_multi_prompt"
    RETURN_TYPES = ("IMAGE", "MASK")

    def segment_multi_prompt(self, grounding_dino_model, sam_model, image, prompts, threshold):
        prompts = prompts if isinstance(prompts, list) else [p.strip() for p in prompts.split(',')]

        combined_image = None
        combined_mask = None

        for prompt in prompts:
            result_image, result_mask = super().main(grounding_dino_model, sam_model, image, prompt, threshold)
            
            if combined_image is None:
                combined_image = result_image
                combined_mask = result_mask
            else:
                combined_mask = torch.logical_or(combined_mask, result_mask)
                combined_image = torch.min(combined_image, result_image)

        if combined_image is None:
            return image, torch.zeros((image.shape[0], image.shape[2], image.shape[3]), dtype=torch.float32, device=image.device)

        return combined_image, combined_mask.float()

NODE_CLASS_MAPPINGS = {
    "MultiPromptGroundingDinoSAMSegment": MultiPromptGroundingDinoSAMSegment
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MultiPromptGroundingDinoSAMSegment": "Multi-Prompt GroundingDINO SAM Segmentation"
}
