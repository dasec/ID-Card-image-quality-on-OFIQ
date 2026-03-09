import unittest
from oidiq.metrics.illumination_uniformity import IlluminationUniformity
from oidiq.session import OIDIQMetricCreator, OIDIQSession, OIDIQPreprocessor, PreProcessors
import numpy as np
from PIL import Image


class TestIlluminationUniformity(unittest.TestCase):
    def test_one_color(self):
        img = np.zeros((100, 200)).astype(np.uint8)
        norm_img_getter = OIDIQPreprocessor.from_function(
            lambda s: img,
            PreProcessors.NORMALIZED_LUMINANCE,
        )
        mask = np.zeros((100, 200)).astype(np.bool)
        mask_getter = OIDIQPreprocessor.from_function(
            lambda s: tuple((mask, mask, mask)),
            (
                PreProcessors.NORMALIZED_FACE_MASK,
                PreProcessors.NORMALIZED_FACE_BACKGROUND_MASK,
                PreProcessors.NORMALIZED_FOREGROUND_MASK,
            ),
        )
        session = OIDIQSession(
            img,
            [norm_img_getter, mask_getter],
            [IlluminationUniformity(sections_w_count=3, sections_h_count=4, min_unmasked_proportion=0.75)],
        )
        r = session.get_score("illumination_uniformity")
        self.assertEqual(100, r.score)
        self.assertEqual(0, r.raw_value)

        img = (np.ones((100, 200)) * 255).astype(np.uint8)
        norm_img_getter = OIDIQPreprocessor.from_function(
            lambda s: img,
            PreProcessors.NORMALIZED_LUMINANCE,
        )
        session = OIDIQSession(
            img,
            [norm_img_getter, mask_getter],
            [IlluminationUniformity(sections_w_count=3, sections_h_count=4, min_unmasked_proportion=0.75)],
        )
        r = session.get_score("illumination_uniformity")
        self.assertEqual(100, r.score)
        self.assertEqual(0, r.raw_value)

    def test_two_black_white(self):
        img = np.array(Image.open("tests/data/black_white.png"))
        norm_img_getter = OIDIQPreprocessor.from_function(
            lambda s: img[:, :, 0],
            PreProcessors.NORMALIZED_LUMINANCE,
        )
        mask = np.zeros((img.shape[0], img.shape[1])).astype(np.bool)
        mask_getter = OIDIQPreprocessor.from_function(
            lambda s: tuple((mask, mask, mask)),
            (
                PreProcessors.NORMALIZED_FACE_MASK,
                PreProcessors.NORMALIZED_FACE_BACKGROUND_MASK,
                PreProcessors.NORMALIZED_FOREGROUND_MASK,
            ),
        )
        session = OIDIQSession(
            img,
            [norm_img_getter, mask_getter],
            [IlluminationUniformity(sections_w_count=3, sections_h_count=4, min_unmasked_proportion=0.75)],
        )
        r = session.get_score("illumination_uniformity")
        self.assertEqual(3, r.score)
        self.assertAlmostEqual(120.208, r.raw_value, places=3)

    def test_two_black_white_mask(self):
        img = np.array(Image.open("tests/data/black_white.png"))
        norm_img_getter = OIDIQPreprocessor.from_function(
            lambda s: img[:, :, 0],
            PreProcessors.NORMALIZED_LUMINANCE,
        )
        mask = np.zeros(img.shape[:2]).astype(np.bool)
        mask[:, : mask.shape[1] // 2] = 1
        mask_getter = OIDIQPreprocessor.from_function(
            lambda s: tuple((mask, mask, mask)),
            (
                PreProcessors.NORMALIZED_FACE_MASK,
                PreProcessors.NORMALIZED_FACE_BACKGROUND_MASK,
                PreProcessors.NORMALIZED_FOREGROUND_MASK,
            ),
        )
        session = OIDIQSession(
            img,
            [norm_img_getter, mask_getter],
            [IlluminationUniformity(sections_w_count=3, sections_h_count=4, min_unmasked_proportion=0.75)],
        )
        r = session.get_score("illumination_uniformity")
        self.assertEqual(100, r.score)
        self.assertAlmostEqual(0, r.raw_value, places=3)

    def test_almost_no_unmasked_areas(self):
        img = np.array(Image.open("tests/data/black_white.png"))
        norm_img_getter = OIDIQPreprocessor.from_function(
            lambda s: img[:, :, 0],
            PreProcessors.NORMALIZED_LUMINANCE,
        )
        mask = np.ones(img.shape[:2]).astype(np.bool)
        mask[: mask.shape[0] // 20, : mask.shape[1] // 20] = 0
        mask_getter = OIDIQPreprocessor.from_function(
            lambda s: tuple((mask, mask, mask)),
            (
                PreProcessors.NORMALIZED_FACE_MASK,
                PreProcessors.NORMALIZED_FACE_BACKGROUND_MASK,
                PreProcessors.NORMALIZED_FOREGROUND_MASK,
            ),
        )
        session = OIDIQSession(
            img,
            [norm_img_getter, mask_getter],
            [IlluminationUniformity(sections_w_count=3, sections_h_count=4, min_unmasked_proportion=0.75)],
        )
        r = session.get_score("illumination_uniformity")
        self.assertEqual(0, r.score)
        self.assertAlmostEqual(0.0, r.raw_value)

    def test_no_unmasked_areas(self):
        img = np.array(Image.open("tests/data/black_white.png"))
        norm_img_getter = OIDIQPreprocessor.from_function(
            lambda s: img[:, :, 0],
            PreProcessors.NORMALIZED_LUMINANCE,
        )
        mask = np.ones((img.shape[0], img.shape[1])).astype(np.bool)
        mask_getter = OIDIQPreprocessor.from_function(
            lambda s: tuple((mask, mask, mask)),
            (
                PreProcessors.NORMALIZED_FACE_MASK,
                PreProcessors.NORMALIZED_FACE_BACKGROUND_MASK,
                PreProcessors.NORMALIZED_FOREGROUND_MASK,
            ),
        )
        session = OIDIQSession(
            img,
            [norm_img_getter, mask_getter],
            [IlluminationUniformity(sections_w_count=3, sections_h_count=4, min_unmasked_proportion=0.75)],
        )

        r = session.get_score("illumination_uniformity")
        self.assertEqual(0, r.score)
        self.assertAlmostEqual(0.0, r.raw_value)
