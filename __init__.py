from .node import *
from .install import *

NODE_CLASS_MAPPINGS = {
    'SAMModelLoader (segment anything)': SAMModelLoader,
    'GroundingDinoModelLoader (segment anything)': GroundingDinoModelLoader,
    'GroundingDinoSAMSegment (segment anything)': GroundingDinoSAMSegment,
    'GenerateVITMatte (segment anything)': GenerateVITMatte,
    'VITMatteTransformersModelLoader (segment anything)': VITMatteTransformersModelLoader,
    'MaskToTrimap (segment anything)': MaskToTrimap,
    'TrimapToMask (segment anything)': TrimapToMask,
    'InvertMask (segment anything)': InvertMask,
    "IsMaskEmpty": IsMaskEmptyNode,
    "BoundingBox (segment anything)":BoundingBox,
    "MaskToBoundingBox (segment anything)":MaskToBoundingBox,
    "BoundingBoxSAMSegment (segment anything)":BoundingBoxSAMSegment,
    "GroundingDinoBoundingBoxes (segment anything)":GroundingDinoBoundingBoxes
}

__all__ = ['NODE_CLASS_MAPPINGS']


