from typing import Optional, Tuple
from ..session import OIDIQSession, OIDIQPreprocessor, PreProcessors
from ..utils import OIDIQConfig, creates, config
import numpy as np

def _section_median_mean(img: np.ndarray, mask: np.ndarray, box: tuple[int, int, int, int] | np.ndarray) -> Tuple[Optional[float], float]:
    x1, y1, x2, y2 = box
    section = img[y1:y2, x1:x2]
    section_mask = mask[y1:y2, x1:x2]
    masked_pixels = section[section_mask]
    if masked_pixels.size == 0:
        return None, 0.0
    return float(np.median(masked_pixels)), float(np.mean(masked_pixels))


class FaceBackgroundMasking(OIDIQPreprocessor):
    @creates(PreProcessors.NORMALIZED_FACE_BACKGROUND_MASK)
    @config("face_background_masking")
    def mask_face_background(self, session: OIDIQSession, config: OIDIQConfig) -> np.ndarray:
        img = session.get_normalized_luminance()
        face_mask = session.get_normalized_face_mask()
        face_boxes = session.get_normalized_face_boxes()
        foreground_mask = session.get_normalized_foreground_mask()
        text_face_mask = (face_mask == 0) & (foreground_mask == 0)
        max_deviation = config.get("max_deviation", 30)
        background_mask = np.zeros(img.shape, dtype=bool)
        grow_amount = config.get("grow_amount", 30)

        def _mask_section(sec_x1, sec_y1, sec_x2, sec_y2):
            tmp_mask = text_face_mask.copy()
            tmp_mask[sec_y1:sec_y2, sec_x1:sec_x2] = False
            top_section_median, top_section_mean = _section_median_mean(img, text_face_mask & ~background_mask, (sec_x1, sec_y1, sec_x2, sec_y2))
            median_luminance, mean_luminance = _section_median_mean(img, tmp_mask, (0, 0, img.shape[1], img.shape[0]))
            if median_luminance is None:
                raise ValueError("Could not compute median luminance for face background masking.")
            if top_section_median is None or abs(top_section_mean - mean_luminance) <= max_deviation or abs(top_section_median - median_luminance) <= max_deviation:
                return False
            background_mask[sec_y1:sec_y2, sec_x1:sec_x2] = True
            return True         
        
        for box in face_boxes:
            x1, y1, x2, y2 = box.astype(int)
            
            section_median, section_mean = _section_median_mean(img, text_face_mask, (x1, y1, x2, y2))
            if section_median is None:
                continue

            tmp_mask = text_face_mask.copy()
            tmp_mask[y1:y2, x1:x2] = False
            median_luminance, mean_luminance = _section_median_mean(img, tmp_mask, (0, 0, img.shape[1], img.shape[0]))
            if median_luminance is None:
                raise ValueError("Could not compute median luminance for face background masking.")

            deviation_mean = abs(section_mean - mean_luminance)
            deviation_median = abs(section_median - median_luminance)
            if deviation_mean > max_deviation and deviation_median > max_deviation:
                background_mask[y1:y2, x1:x2] = True
            else:
                continue
            
       

            while True:
                sec_x1, sec_y1, sec_x2, sec_y2 = (x1, max(0, y1 - grow_amount), x2, y1)
                cont = _mask_section(sec_x1, sec_y1, sec_x2, sec_y2)
                if not cont:
                    break
                y1 = max(0, y1 - grow_amount)
            
            while True:
                sec_x1, sec_y1, sec_x2, sec_y2 = (x1, y2, x2, min(img.shape[0], y2 + grow_amount))
                cont = _mask_section(sec_x1, sec_y1, sec_x2, sec_y2)
                if not cont:
                    break
                y2 = min(img.shape[0], y2 + grow_amount)
                
            
            while True:
                sec_x1, sec_y1, sec_x2, sec_y2 = (max(0, x1 - grow_amount), y1, x1, y2)
                cont = _mask_section(sec_x1, sec_y1, sec_x2, sec_y2)
                if not cont:
                    break
                x1 = max(0, x1 - grow_amount)


            while True:
                cont = _mask_section(x2, y1, min(img.shape[1], x2 + grow_amount), y2)
                if not cont:
                    break
                x2 = min(img.shape[1], x2 + grow_amount)
            
            
        return background_mask