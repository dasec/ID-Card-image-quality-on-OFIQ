
from typing import List, Tuple
import cv2
from ultralytics import YOLO # type: ignore
from ..session import OIDIQBatchSession, OIDIQPreprocessor, PreProcessors # type: ignore
from ..utils import OIDIQConfig, creates, config, get_device, batching
import numpy as np



class FaceMasking(OIDIQPreprocessor):
    @config("face_masking")
    def init_config(self, config: OIDIQConfig):
        config["device"] = get_device(config)
        config["model"] = YOLO(config["model_path"])
        config["model"].to(config["device"])
        config["model"].eval()

    @creates(PreProcessors.NORMALIZED_FACE_MASK, PreProcessors.NORMALIZED_FACE_BOXES)
    @config("face_masking")
    @batching(16)
    def mask_face(self, session: OIDIQBatchSession, config: OIDIQConfig) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        #imgs = np.array(session.get_normalized_image())
        #imgs = imgs[..., ::-1] 
        imgs = [img[:, :, ::-1] for img in session.get_normalized_image()]  # Convert RGB to BGR
        results = config["model"].predict(source=imgs, device=config["device"], conf=config["min_confidence"], classes=[0], verbose=False, batch=len(imgs))
        masks = []
        boxes = []
        for batch_idx, result in enumerate(results):
            mask, box = self._get_results(result, imgs[batch_idx].shape, session, batch_idx, config)
            masks.append(mask)
            boxes.append(box)
        return masks, boxes
    
    def _get_results(self, result, img_shape, session, batch_idx, config) -> Tuple[np.ndarray, np.ndarray]:
        mask = np.zeros(img_shape[:2], dtype=np.uint8)
        boxes = []
        session.log(self, f"Detected {len(result.boxes or [])} faces with confidences confidence >= {config['min_confidence']:.2f} in image {batch_idx}")
        for i in range(len(result.masks or [])):
            mask_part = result.masks.data[i].cpu().numpy().astype(np.uint8)*255
            mask_part = cv2.resize(mask_part, (img_shape[1], img_shape[0]), interpolation=cv2.INTER_NEAREST)
            mask = np.maximum(mask, mask_part)
            boxes.append(result.boxes.xyxy[i].cpu().numpy())
        return mask, np.array(boxes)