from .node import *

NODE_CLASS_MAPPINGS = {
    'SAMModelLoader (segment anything)': SAMModelLoader,
    'GroundingDinoModelLoader (segment anything)': GroundingDinoModelLoader,
    'GroundingDinoSAMSegment (segment anything)': GroundingDinoSAMSegment,
    'InvertMask (segment anything)': InvertMask
}

__all__ = ['NODE_CLASS_MAPPINGS']
