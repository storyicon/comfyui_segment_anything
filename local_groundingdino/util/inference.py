from typing import Tuple, List

import cv2
import numpy as np
import supervision as sv
import torch
from PIL import Image
from torchvision.ops import box_convert

import local_groundingdino.datasets.transforms as T
from local_groundingdino.models import build_model
from local_groundingdino.util.misc import clean_state_dict
from local_groundingdino.util.slconfig import SLConfig
from local_groundingdino.util.utils import get_phrases_from_posmap

# ----------------------------------------------------------------------------------------------------------------------
# OLD API
# ----------------------------------------------------------------------------------------------------------------------


def preprocess_caption(caption: str) -> str:
    """Processes a caption by converting it to lowercase, stripping leading and trailing whitespaces, and ensuring it ends with a period."""
    result = caption.lower().strip()
    if result.endswith("."):
        return result
    return result + "."


def load_model(model_config_path: str, model_checkpoint_path: str, device: str = "cuda") -> torch.nn.Module:
    """Loads a model using configuration settings from a file, applies the specified device, builds the model based on the settings, loads the model checkpoint, cleans the state dictionary keys, sets the model to evaluation mode, and returns the loaded model."""
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    model.eval()
    return model


def load_image(image_path: str) -> tuple[numpy.ndarray, torch.Tensor]:
    """Loads an image from the specified path, applies a series of transformations including resizing, converting to a tensor, and normalization, returning the original image as a NumPy array and the transformed image as a PyTorch tensor."""
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image_source = Image.open(image_path).convert("RGB")
    image = np.asarray(image_source)
    image_transformed, _ = transform(image_source, None)
    return image, image_transformed


def predict(
        model: torch.nn.Module,
        image: torch.Tensor,
        caption: str,
        box_threshold: float,
        text_threshold: float,
        device: str = "cuda"
    ) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """Processes an image and a caption using a provided model to predict bounding boxes and associated phrases. It preprocesses the caption, moves the model and image to the specified device, generates predictions, filters based on thresholds, tokenizes the caption, and retrieves phrases based on logits exceeding a text threshold."""
    caption = preprocess_caption(caption=caption)

    model = model.to(device)
    image = image.to(device)

    with torch.no_grad():
        outputs = model(image[None], captions=[caption])

    prediction_logits = outputs["pred_logits"].cpu().sigmoid()[0]  # prediction_logits.shape = (nq, 256)
    prediction_boxes = outputs["pred_boxes"].cpu()[0]  # prediction_boxes.shape = (nq, 4)

    mask = prediction_logits.max(dim=1)[0] > box_threshold
    logits = prediction_logits[mask]  # logits.shape = (n, 256)
    boxes = prediction_boxes[mask]  # boxes.shape = (n, 4)

    tokenizer = model.tokenizer
    tokenized = tokenizer(caption)

    phrases = [
        get_phrases_from_posmap(logit > text_threshold, tokenized, tokenizer).replace('.', '')
        for logit
        in logits
    ]

    return boxes, logits.max(dim=1)[0], phrases


def annotate(image_source: np.ndarray, boxes: torch.Tensor, logits: torch.Tensor, phrases: List[str]) -> np.ndarray:
    h, w, _ = image_source.shape
    boxes = boxes * torch.Tensor([w, h, w, h])
    xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
    detections = sv.Detections(xyxy=xyxy)

    labels = [
        f"{phrase} {logit:.2f}"
        for phrase, logit
        in zip(phrases, logits)
    ]

    box_annotator = sv.BoxAnnotator()
    annotated_frame = cv2.cvtColor(image_source, cv2.COLOR_RGB2BGR)
    annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
    return annotated_frame


# ----------------------------------------------------------------------------------------------------------------------
# NEW API
# ----------------------------------------------------------------------------------------------------------------------


class Model:

    def __init__(
            self,
            model_config_path: str,
            model_checkpoint_path: str,
            device: str = "cuda"
        ) -> None:
        """Loads a model using configuration settings from a file, applies the specified device, builds the model based on the settings, loads the model checkpoint, cleans the state dictionary keys, sets the model to evaluation mode, and returns the loaded model."""
        self.model = load_model(
            model_config_path=model_config_path,
            model_checkpoint_path=model_checkpoint_path,
            device=device
        ).to(device)
        self.device = device

    def predict_with_caption(
            self,
            image: np.ndarray,
            caption: str,
            box_threshold: float,
            text_threshold: float
        ) -> tuple[sv.Detections, list[str]]:
        """
        import cv2

        image = cv2.imread(IMAGE_PATH)

        model = Model(model_config_path=CONFIG_PATH, model_checkpoint_path=WEIGHTS_PATH)
        detections, labels = model.predict_with_caption(
            image=image,
            caption=caption,
            box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD
        )

        import supervision as sv

        box_annotator = sv.BoxAnnotator()
        annotated_image = box_annotator.annotate(scene=image, detections=detections, labels=labels)
        """
        processed_image = Model.preprocess_image(image_bgr=image).to(self.device)
        boxes, logits, phrases = predict(
            model=self.model,
            image=processed_image,
            caption=caption,
            box_threshold=box_threshold,
            text_threshold=text_threshold, 
            device=self.device)
        source_h, source_w, _ = image.shape
        detections = Model.post_process_result(
            source_h=source_h,
            source_w=source_w,
            boxes=boxes,
            logits=logits)
        return detections, phrases

    def predict_with_classes(
            self,
            image: np.ndarray,
            classes: list[str],
            box_threshold: float,
            text_threshold: float
        ) -> sv.Detections:
        """
        import cv2

        image = cv2.imread(IMAGE_PATH)

        model = Model(model_config_path=CONFIG_PATH, model_checkpoint_path=WEIGHTS_PATH)
        detections = model.predict_with_classes(
            image=image,
            classes=CLASSES,
            box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD
        )


        import supervision as sv

        box_annotator = sv.BoxAnnotator()
        annotated_image = box_annotator.annotate(scene=image, detections=detections)
        """
        caption = ". ".join(classes)
        processed_image = Model.preprocess_image(image_bgr=image).to(self.device)
        boxes, logits, phrases = predict(
            model=self.model,
            image=processed_image,
            caption=caption,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            device=self.device)
        source_h, source_w, _ = image.shape
        detections = Model.post_process_result(
            source_h=source_h,
            source_w=source_w,
            boxes=boxes,
            logits=logits)
        class_id = Model.phrases2classes(phrases=phrases, classes=classes)
        detections.class_id = class_id
        return detections

    @staticmethod
    def preprocess_image(image_bgr: np.ndarray) -> torch.Tensor:
        """Preprocesses a BGR image represented as a NumPy array by converting it to a PyTorch tensor after applying a series of transformations including resizing, converting color space, normalization, and tensor conversion."""
        transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        image_pillow = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        image_transformed, _ = transform(image_pillow, None)
        return image_transformed

    @staticmethod
    def post_process_result(source_h: int, source_w: int, boxes: torch.Tensor, logits: torch.Tensor) -> sv.Detections:
        """Processes bounding boxes and confidence logits to create a Detections object containing converted bounding box coordinates and confidence scores."""
        boxes = boxes * torch.Tensor([source_w, source_h, source_w, source_h])
        xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
        confidence = logits.numpy()
        return sv.Detections(xyxy=xyxy, confidence=confidence)

    @staticmethod
    def phrases2classes(phrases: list[str], classes: list[str]) -> np.ndarray:
        """Converts a list of phrases to corresponding class IDs based on a list of classes, returning a NumPy array of the class IDs. Handles cases where a phrase is not found in the classes list by assigning None."""
        class_ids = []
        for phrase in phrases:
            try:
                class_ids.append(classes.index(phrase))
            except ValueError:
                class_ids.append(None)
        return np.array(class_ids)