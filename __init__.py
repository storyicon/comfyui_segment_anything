import os
import sys
import folder_paths
sys.path.append(
    os.path.dirname(os.path.abspath(__file__))
)

from .nodes.node_batch_selector import BatchSelector
from .nodes.node_sam_autoseg import SAMAutoSegment
from .nodes.node_dinosam_prompt import GroundingDinoSAMSegment
from .nodes.node_modelloader_dino import GroundingDinoModelLoader
from .nodes.node_modelloader_sam import SAMModelLoader

NODE_CLASS_MAPPINGS = {
    'SAMModelLoader (segment anything)': SAMModelLoader,
    'GroundingDinoModelLoader (segment anything)': GroundingDinoModelLoader,
    'GroundingDinoSAMSegment (segment anything)': GroundingDinoSAMSegment,
    'SAM Auto Segmentation (segment anything)': SAMAutoSegment,
    'BatchSelector (segment anything)': BatchSelector,
}

__all__ = ['NODE_CLASS_MAPPINGS']
