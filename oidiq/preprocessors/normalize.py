from typing import Tuple

import cv2
from ..session import OIDIQSession, OIDIQPreprocessor, PreProcessors
from ..utils import OIDIQConfig, creates, config, map_to_image_borders
import numpy as np


class Normalizing(OIDIQPreprocessor):
    @creates(PreProcessors.NORMALIZED_IMAGE)
    @config("normalization")
    def warp_perspective(self, session: OIDIQSession, config: OIDIQConfig) -> np.ndarray:
        w = config["width"]
        h = config["height"]
        img = session.get_raw_image()
        corners = map_to_image_borders(
            session.get_id_card_corners(),
            img.shape[1],
            img.shape[0],
            config.get("map_corners_mode", "linear"),
        )

        mat = cv2.getPerspectiveTransform(
            corners.astype(np.float32),
            np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32),
        )
        normalized_img = cv2.warpPerspective(img, mat, (w, h), flags=config.get("interpolation", cv2.INTER_CUBIC))

        return normalized_img
