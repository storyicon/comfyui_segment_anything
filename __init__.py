import os
import sys
import folder_paths
sys.path.append(
    os.path.dirname(os.path.abspath(__file__))
)
from .node import *

NODE_CLASS_MAPPINGS = {
    'SAMModelLoader (segment anything)': SAMModelLoader,
    'GroundingDinoModelLoader (segment anything)': GroundingDinoModelLoader,
    'GroundingDinoSAMSegment (segment anything)': GroundingDinoSAMSegment,
    'BatchSelector (segment anything)': BatchSelector,
}

__all__ = ['NODE_CLASS_MAPPINGS']
