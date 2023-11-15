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