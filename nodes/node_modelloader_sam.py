from ..nodes_fnc.node_modelloader_sam import list_sam_model, load_sam_model

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
