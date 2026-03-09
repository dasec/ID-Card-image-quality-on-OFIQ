from typing import List
import numpy as np

from ..session import OIDIQSession, PreProcessors, config, OIDIQConfig
from ..session import OIDIQPreprocessor
from ..utils import creates, create_mask


def _color_convert(v):
    return np.where(v > 0.04045, ((v + 0.055) / 1.055) ** 2.4, v / 12.92)


def _normalized_luminance(img: np.ndarray, rgb_weights:List[float]) -> np.ndarray:
    img = img.astype(np.float32) / 255.0

    rl = _color_convert(img[..., 0])
    gl = _color_convert(img[..., 1])
    bl = _color_convert(img[..., 2])
    return np.clip(np.floor(256 * (0.2126 * rl + 0.7152 * gl + 0.0722 * bl)), 0, 255).astype(np.uint8)


class NormalizedLuminanceHistogram(OIDIQPreprocessor):
    @creates(PreProcessors.NORMALIZED_LUMINANCE_HISTOGRAM)
    @config("luminance_histogram")
    def execute(self, session: OIDIQSession, config: OIDIQConfig) -> np.ndarray:
        lum = session.get_normalized_luminance()
        mask = create_mask(lum, session, config)
        session.log(self, f"Calculating luminance histogram of {np.mean(mask)*100:.2f}% of pixels.")
        lum =lum[mask].flatten()
        hist, _ = np.histogram(lum, bins=256, range=(0, 255))
        return hist / (hist.sum() + 1e-8)


class NormalizedLuminance(OIDIQPreprocessor):
    @creates(PreProcessors.NORMALIZED_LUMINANCE)
    @config("luminance")
    def execute(self, session: OIDIQSession, config: OIDIQConfig) -> np.ndarray:
        return _normalized_luminance(session.get_normalized_image(), config.get("rgb_weights", [0.2126, 0.7152, 0.0722]))
    
