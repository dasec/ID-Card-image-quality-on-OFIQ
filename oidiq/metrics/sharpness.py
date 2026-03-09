from typing import Tuple
from ..session import OIDIQSession, OIDIQMetricCreator
from ..utils import config, creates, QualityMetric, sigmoid, QualityMetricConfig
import numpy as np
import cv2


def _sharpness_features(session: OIDIQSession) -> np.ndarray:
    img = session.get_normalized_image()
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    features = []

    # calculate Laplacian features
    for kernel_size in [1, 3, 5, 7, 9]:
        lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=kernel_size)
        abs_lap = np.abs(lap)
        mean = np.mean(abs_lap)
        std = np.std(abs_lap)
        features.extend([mean, std])

    # calculate Mean Diff features
    for kernel_size in [3, 5, 7]:
        gray_mean_blur = cv2.blur(gray, (kernel_size, kernel_size))
        abs_diff = cv2.absdiff(gray, gray_mean_blur)
        mean, std = cv2.meanStdDev(abs_diff)
        features.extend([mean[0][0], std[0][0]])

    # calculate Sobel features
    for kernel_size in [1, 3, 5, 7, 9]:
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 1, ksize=kernel_size)
        abs_sobel = np.abs(sobel)
        mean = np.mean(abs_sobel)
        std = np.std(abs_sobel)
        features.extend([mean, std])

    features = np.array(features, dtype=np.float32)
    return features.reshape(1, -1)


class Sharpness(OIDIQMetricCreator):
    @config("sharpness")
    def init_config(self, config: QualityMetricConfig) -> None:
        config["tree"] = cv2.ml.RTrees.load(config["tree_path"])

    @creates("sharpness")
    def sharpness(self, session: OIDIQSession, config: QualityMetricConfig) -> QualityMetric:
        features = _sharpness_features(session)

        rtree: cv2.ml.RTrees = config["tree"]

        pred_results = rtree.predict(features, flags=1)  # cv::ml::StatModel::RAW_OUTPUT
        prediction = 50 - pred_results[0]
        score = int(np.round(115 * sigmoid(prediction, -20, 15) - 14))

        return config.create_quality_metric(float(prediction), score)
