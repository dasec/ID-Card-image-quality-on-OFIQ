from typing import Tuple
from ..session import OIDIQSession, OIDIQMetricCreator
from ..utils import creates, QualityMetric, sigmoid, QualityMetricConfig, full_sigmoid, config, create_mask
import numpy as np
import cv2

def _calc_illumination_uniformity_score(img: np.ndarray, mask, config: QualityMetricConfig, session: OIDIQSession|None=None) -> Tuple[float, int]:
        normal_width = img.shape[1]
        normal_height = img.shape[0]
        sections_w_count = config["sections_w_count"]
        sections_h_count = config["sections_h_count"]
        section_w = normal_width // sections_w_count
        section_h = normal_height // sections_h_count
        section_w_offset = (normal_width % sections_w_count) // 2
        section_h_offset = (normal_height % sections_h_count) // 2

        min_pixels_per_sec = config["min_unmasked_proportion"] * (section_w * section_h)
        min_sec_count = config.get("min_section_count_proportion", 0.25) * (sections_w_count * sections_h_count)
        div = 1
        section_means = []
        while len(section_means) < min_sec_count:
            section_means = []
            for i in range(sections_h_count):
                for j in range(sections_w_count):
                    start_y = section_h_offset + i * section_h
                    end_y = section_h_offset + (i + 1) * section_h
                    start_x = section_w_offset + j * section_w
                    end_x = section_w_offset + (j + 1) * section_w
                    sec = img[start_y:end_y, start_x:end_x]

                    sec_mask = mask[start_y:end_y, start_x:end_x]
                    sec = sec[sec_mask].flatten()
                    if len(sec) >= min_pixels_per_sec:
                        section_means.append(np.median(sec))
            if len(section_means) < min_sec_count:
                div *= 2
                if session is not None:
                    session.log(IlluminationUniformity, f"Found only {len(section_means)}/{sections_w_count * sections_h_count} sections with at least {int(min_pixels_per_sec)} ({config['min_unmasked_proportion']*100:.2f}%) unmasked pixels, reducing requirement by factor {div} from {min_pixels_per_sec} to {min_pixels_per_sec // div} pixels.")
                min_pixels_per_sec //= div
                if min_pixels_per_sec == 0:
                    if session is not None:
                        session.log(IlluminationUniformity, f"All fully sections masked, returning 0 score for illumination uniformity.")
                    return 0.0, 0
            elif session is not None:
                session.log(IlluminationUniformity, f"Found {len(section_means)}/{sections_w_count * sections_h_count} sections with at least {int(min_pixels_per_sec)} ({config['min_unmasked_proportion']*100:.2f}%) unmasked pixels.")

        #section_means = np.array([np.mean(s) for s in sections], dtype=np.float32)
        if session is not None:
            session.log(IlluminationUniformity, f"Section medians: {section_means}")
        section_sdivs = float(np.std(section_means))
        score = int(np.clip(full_sigmoid(0.8 - section_sdivs / 100, 6.6, 0.6, 15, 0.48, 0.03), 0, 100))

        return section_sdivs, score


class IlluminationUniformity(OIDIQMetricCreator):
    @creates("illumination_uniformity")
    def normal_illumination_uniformity(self, session: OIDIQSession, config: QualityMetricConfig) -> QualityMetric:
        img = session.get_normalized_luminance()
        mask = create_mask(
            img,
            session,
            config,
            default_face_mask=True,
            default_dark_face_background=True,
            default_foreground=True,
        )
        
        session.log(self, f"Calculating illumination uniformity on {np.mean(mask)*100:.2f}% of pixels.")

        section_sdivs, score = _calc_illumination_uniformity_score(img, mask, config, session=session)

        return config.create_quality_metric(section_sdivs, score)
    

