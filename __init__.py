from .node import *
from .install import *
from .batch_prompt_splitter import *
from .multi_prompt_grounding_dino_sam import *

NODE_CLASS_MAPPINGS = {
    'SAMModelLoader (segment anything)': SAMModelLoader,
    'GroundingDinoModelLoader (segment anything)': GroundingDinoModelLoader,
    'GroundingDinoSAMSegment (segment anything)': GroundingDinoSAMSegment,
    'InvertMask (segment anything)': InvertMask,
    "IsMaskEmpty": IsMaskEmptyNode,
    "MultiPromptGroundingDinoSAMSegment (segment anything)": MultiPromptGroundingDinoSAMSegment,    
    "BatchPromptSplitter (segment anything)": BatchPromptSplitter,
}

__all__ = ['NODE_CLASS_MAPPINGS']


