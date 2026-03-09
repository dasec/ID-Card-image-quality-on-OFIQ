import unittest
from oidiq.preprocessors.corner_detection.corner_detection import estimate_unknown_corners, estimate_single_missing_corner, estimate_adjacent_missing_corners, resize_keep_ratio, get_corner_positions_from_heatmaps
import numpy as np

class TestCornerEstimation(unittest.TestCase):
    def test_estimate_single_missing_corner_0(self):
        corners = np.array([[-1, 1], [6, 1], [8, 7], [3, 9]])
        missing_index = 0
        estimated = estimate_single_missing_corner(corners, missing_index)
        self.assertEqual(estimated, (3, 1))

    def test_estimate_single_missing_corner_1(self):
        corners = np.array([[1, 1], [-1, -1], [8, 7], [3, 9]])
        missing_index = 1
        estimated = estimate_single_missing_corner(corners, missing_index)
        self.assertEqual(estimated, (8, 1))

    def test_estimate_single_missing_corner_2(self):
        corners = np.array([[1, 1], [6, 1], [-1, -1], [3, 9]])
        missing_index = 2
        estimated = estimate_single_missing_corner(corners, missing_index)
        self.assertEqual(estimated, (6, 9))

    def test_estimate_single_missing_corner_3(self):
        corners = np.array([[1, 1], [6, 1], [8, 7], [-1, -1]])
        missing_index = 3
        estimated = estimate_single_missing_corner(corners, missing_index)
        self.assertEqual(estimated, (1, 7))

    def test_estimate_adjacent_missing_corners_0_1(self):
        corners = np.array([[-1, -1], [-1, -1], [8, 7], [3, 9]])
        idx1, idx2 = 0, 1
        estimated1, estimated2 = estimate_adjacent_missing_corners(corners, idx1, idx2, 12, 10)
        self.assertEqual(estimated1, (3, 0))
        self.assertEqual(estimated2, (8, 0))

    def test_estimate_adjacent_missing_corners_1_2(self):
        corners = np.array([[1, 1], [-1, -1], [-1, -1], [3, 9]])
        idx1, idx2 = 1, 2
        estimated1, estimated2 = estimate_adjacent_missing_corners(corners, idx1, idx2, 12, 10)
        self.assertEqual(estimated1, (11, 1))
        self.assertEqual(estimated2, (11, 9))

    def test_estimate_adjacent_missing_corners_2_3(self):
        corners = np.array([[1, 1], [6, 1], [-1, -1], [-1, -1]])
        idx1, idx2 = 2, 3
        estimated1, estimated2 = estimate_adjacent_missing_corners(corners, idx1, idx2, 12, 10)
        self.assertEqual(estimated1, (6, 9))
        self.assertEqual(estimated2, (1, 9))

    def test_estimate_adjacent_missing_corners_3_0(self):
        corners = np.array([[-1, -1], [6, 1], [8, 7], [-1, -1]])
        idx1, idx2 = 3, 0
        estimated1, estimated2 = estimate_adjacent_missing_corners(corners, idx1, idx2, 12, 10)
        self.assertEqual(estimated1, (0, 1))
        self.assertEqual(estimated2, (0, 7))

    def test_estimate_unknown_corners_0_1(self):
        corners = np.array([[-1, -1], [6, 1], [-1, -1], [3, 9]])
        confidences = np.array([0.2, 0.9, 0.1, 0.8])
        estimated = estimate_unknown_corners(corners, confidences, thresh=0.5, w=12, h=10)
        expected = np.array([[3, 1], [6, 1], [6, 9], [3, 9]])
        np.testing.assert_array_equal(estimated, expected)
    
    def test_estimate_unknown_corners_1_2(self):
        corners = np.array([[1, 1], [-1, -1], [-1, -1], [3, 9]])
        confidences = np.array([0.9, 0.2, 0.3, 0.8])
        estimated = estimate_unknown_corners(corners, confidences, thresh=0.5, w=12, h=10)
        expected = np.array([[1, 1], [11, 1], [11, 9], [3, 9]])
        np.testing.assert_array_equal(estimated, expected)
    
    def test_estimate_unknown_corners_3_missing(self):
        corners = np.array([[1, 1], [6, 1], [8, 7], [-1, -1]])
        confidences = np.array([0.9, 0.8, 0.7, 0.2])
        estimated = estimate_unknown_corners(corners, confidences, thresh=0.5, w=12, h=10)
        expected = np.array([[1, 1], [6, 1], [8, 7], [1, 7]])
        np.testing.assert_array_equal(estimated, expected)
    
    def test_estimate_unknown_corners_1_3(self):
        corners = np.array([[1, 2], [-1, -1], [8, 7], [-1, -1]])
        confidences = np.array([0.9, 0.2, 0.8, 0.3])
        estimated = estimate_unknown_corners(corners, confidences, thresh=0.5, w=12, h=10)
        expected = np.array([[1, 2], [8, 2], [8, 7], [1, 7]])
        np.testing.assert_array_equal(estimated, expected)
    
    def test_estimate_unknown_corners_0_1_2(self):
        corners = np.array([[-1, -1], [-1, -1], [-1, -1], [3, 8]])
        confidences = np.array([0.1, 0.2, 0.3, 0.9])
        estimated = estimate_unknown_corners(corners, confidences, thresh=0.5, w=12, h=10)
        expected = np.array([[3, 0], [11, 0], [11, 8], [3, 8]])
        np.testing.assert_array_equal(estimated, expected)

    def test_estimate_unknown_corners_1_2_3(self):
        corners = np.array([[1, 1], [-1, -1], [-1, -1], [-1, -1]])
        confidences = np.array([0.9, 0.1, 0.2, 0.3])
        estimated = estimate_unknown_corners(corners, confidences, thresh=0.5, w=12, h=10)
        expected = np.array([[1, 1], [11, 1], [11, 9], [1, 9]])
        np.testing.assert_array_equal(estimated, expected)

    def test_estimate_unknown_corners_0_2_3(self):
        corners = np.array([[-1, -1], [6, 1], [-1, -1], [-1, -1]])
        confidences = np.array([0.2, 0.9, 0.1, 0.3])
        estimated = estimate_unknown_corners(corners, confidences, thresh=0.5, w=12, h=10)
        expected = np.array([[0, 1], [6, 1], [6, 9], [0, 9]])
        np.testing.assert_array_equal(estimated, expected)

    def test_estimate_unknown_corners_0_1_3(self):
        corners = np.array([[-1, -1], [-1, -1], [8, 7], [-1, -1]])
        confidences = np.array([0.2, 0.3, 0.9, 0.1])
        estimated = estimate_unknown_corners(corners, confidences, thresh=0.5, w=12, h=10)
        expected = np.array([[0, 0], [8, 0], [8, 7], [0, 7]])
        np.testing.assert_array_equal(estimated, expected)

    def test_estimate_unknown_corners_0_2(self):
        corners = np.array([[-1, -1], [6, 1], [-1, -1], [3, 9]])
        confidences = np.array([0.2, 0.9, 0.1, 0.8])
        estimated = estimate_unknown_corners(corners, confidences, thresh=0.5, w=12, h=10)
        expected = np.array([[3, 1], [6, 1], [6, 9], [3, 9]])
        np.testing.assert_array_equal(estimated, expected)

    def test_estimate_unknown_corners_3_0(self):
        corners = np.array([[-1, -1], [6, 1], [8, 7], [-1, -1]])
        confidences = np.array([0.2, 0.9, 0.8, 0.1])
        estimated = estimate_unknown_corners(corners, confidences, thresh=0.5, w=12, h=10)
        expected = np.array([[0, 1], [6, 1], [8, 7], [0, 7]])
        np.testing.assert_array_equal(estimated, expected)

    def test_estimate_unknown_corners_all_missing(self):
        corners = np.array([[-1, -1], [-1, -1], [-1, -1], [-1, -1]])
        confidences = np.array([0.1, 0.2, 0.3, 0.4])
        estimated = estimate_unknown_corners(corners, confidences, thresh=0.5, w=12, h=10)
        expected = np.array([[0, 0], [11, 0], [11, 9], [0, 9]])
        np.testing.assert_array_equal(estimated, expected)

class TestCornerDetection(unittest.TestCase):
    def test_resize_keep_ratio_width_greater_no_resize(self):
        img = np.zeros((100, 256, 3), dtype=np.uint8)
        img[40, 80, :] = 255
        resized_img, pad_w, pad_h = resize_keep_ratio(img, (256,256), 0)
        self.assertEqual(resized_img.shape[1], 256)
        self.assertEqual(resized_img.shape[0], 256)
        self.assertEqual(pad_w, 0)
        self.assertEqual(pad_h, (256 - 100) // 2)
        self.assertEqual(resized_img[40 + pad_h, 80, 0], 255)

    def test_resize_keep_ratio_width_greater(self):
        img = np.zeros((100, 512, 3), dtype=np.uint8)
        img[40, 80, :] = 255
        resized_img, pad_w, pad_h = resize_keep_ratio(img, (256, 256), 0)
        self.assertEqual(resized_img.shape[1], 256)
        self.assertEqual(resized_img.shape[0], 256)
        self.assertEqual(pad_w, 0)
        self.assertEqual(pad_h, (256 - 50) // 2)
        self.assertTrue(resized_img[40 // 2 + pad_h, 80 // 2, 0] > 0)

    def test_resize_keep_ratio_height_greater(self):
        img = np.zeros((512, 100, 3), dtype=np.uint8)
        img[80, 40, :] = 255
        resized_img, pad_w, pad_h = resize_keep_ratio(img, (256, 256), 0)
        self.assertEqual(resized_img.shape[1], 256)
        self.assertEqual(resized_img.shape[0], 256)
        self.assertEqual(pad_w, (256 - 50) // 2)
        self.assertEqual(pad_h, 0)
        self.assertTrue(resized_img[80 // 2, 40 // 2 + pad_w, 0] > 0)

    def test_padding_keep_ratio_square(self):
        img = np.zeros((300, 300, 3), dtype=np.uint8)
        img[150, 150, :] = 255
        resized_img, pad_w, pad_h = resize_keep_ratio(img, (256, 256), 0.25)
        self.assertEqual(resized_img.shape[1], 256)
        self.assertEqual(resized_img.shape[0], 256)
        self.assertEqual(pad_w, int(75 / (450 / 256)))
        self.assertEqual(pad_h, int(75 / (450 / 256)))
        self.assertTrue(resized_img[128, 128, 0] > 0)
    
    def test_padding_keep_ratio_width_greater(self):
        img = np.zeros((200, 400, 3), dtype=np.uint8)
        img[150, 150, :] = 255
        resized_img, pad_w, pad_h = resize_keep_ratio(img, (256, 256), 0.1)
        self.assertEqual(resized_img.shape[1], 256)
        self.assertEqual(resized_img.shape[0], 256)
        self.assertEqual(pad_w, int(40 / (480 / 256)))
        self.assertEqual(pad_h, 74)
        self.assertTrue(resized_img[int(290 / (480 / 256)), int(190 / (480 / 256)), 0] > 0) 
    
    def test_get_corner_positions_from_heatmaps(self):
        heatmap_1 = np.zeros((512, 256), dtype=np.float32)
        heatmap_1[100, 50] = 1.0
        resized_heatmap_1, pad_w_1, pad_h_1 = resize_keep_ratio(heatmap_1, (256, 256), 0)
        heatmap_2 = np.zeros((512, 256), dtype=np.float32)
        heatmap_2[200, 150] = 1.0
        resized_heatmap_2, pad_w_2, pad_h_2 = resize_keep_ratio(heatmap_2, (256, 256), 0)
        heatmap_3 = np.zeros((512, 256), dtype=np.float32)
        heatmap_3[300, 200] = 1.0
        resized_heatmap_3, pad_w_3, pad_h_3 = resize_keep_ratio(heatmap_3, (256, 256), 0)
        heatmap_4 = np.zeros((512, 256), dtype=np.float32)
        heatmap_4[400, 100] = 1.0
        resized_heatmap_4, pad_w_4, pad_h_4 = resize_keep_ratio(heatmap_4, (256, 256), 0)

        self.assertEqual(pad_w_1, pad_w_2)
        self.assertEqual(pad_w_1, pad_w_3)
        self.assertEqual(pad_w_1, pad_w_4)
        self.assertEqual(pad_h_1, pad_h_2)
        self.assertEqual(pad_h_1, pad_h_3)
        self.assertEqual(pad_h_1, pad_h_4)
        self.assertEqual(resized_heatmap_1.shape, resized_heatmap_2.shape)
        self.assertEqual(resized_heatmap_1.shape, resized_heatmap_3.shape)
        self.assertEqual(resized_heatmap_1.shape, resized_heatmap_4.shape)
        heatmaps = np.stack([resized_heatmap_1, resized_heatmap_2, resized_heatmap_3, resized_heatmap_4], axis=0)
        heatmaps = np.expand_dims(heatmaps, axis=0)  # Add batch dimension
        positions, confidences = get_corner_positions_from_heatmaps(heatmaps, pad_w_1, pad_h_1, (512, 256))
        expected_positions = [(50, 100), (150, 200), (200, 300), (100, 400)]
        for pos, expected in zip(positions, expected_positions):
            self.assertEqual(pos, expected)



    

        


    