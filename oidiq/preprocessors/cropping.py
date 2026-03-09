from typing import Tuple
from ..session import OIDIQSession, OIDIQPreprocessor, PreProcessors
from ..utils import OIDIQConfig, creates, config, QualityMetricConfig, map_to_image_borders
import numpy as np

class Cropping(OIDIQPreprocessor):
    @creates(PreProcessors.CROPPED_IMAGE)
    @config("cropping")
    def crop(self, session: OIDIQSession, config: OIDIQConfig) -> np.ndarray:
        raw_image = session.get_raw_image()
        corners = map_to_image_borders(
            session.get_id_card_corners(),
            raw_image.shape[1],
            raw_image.shape[0],
            config.get("clip_corners_mode", "linear"),
        )
        x1 = min(corners[:, 0])
        y1 = min(corners[:, 1])
        x2 = max(corners[:, 0])
        y2 = max(corners[:, 1])
        cropped_image = raw_image[y1:y2, x1:x2]
        return cropped_image
    


        


