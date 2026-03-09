from typing import Tuple
from ..session import OIDIQSession, OIDIQMetricCreator
from ..utils import creates, QualityMetric, sigmoid, QualityMetricConfig
import numpy as np


class LuminanceMoments(OIDIQMetricCreator):
    @creates(
        "luminance_mean",
        "luminance_variance",
        "luminance_skewness",
        "luminance_kurtosis",
    )
    def luminance_moments(
        self,
        session: OIDIQSession,
        mean_config: QualityMetricConfig,
        variance_config: QualityMetricConfig,
        skewness_config: QualityMetricConfig,
        kurtosis_config: QualityMetricConfig,
    ) -> Tuple[QualityMetric, QualityMetric, QualityMetric, QualityMetric]:
        hist = session.get_normalized_luminance_histogram()
        nl = np.arange(256)
        mean = np.sum(nl / 255.0 * hist)
        var = np.sum(((nl / 255.0 - mean) ** 2) * hist)
        std = np.sqrt(var) + 1e-8 
        skewness = np.sum((np.abs(nl / 255.0 - mean) ** 3) * hist) / (std**3)
        kurtosis = np.sum(((nl / 255.0 - mean) ** 4) * hist) / (std**4)

        mean_score = np.ceil(100 * sigmoid(mean, 0.2, 0.05) * (1 - sigmoid(mean, 0.8, 0.05)))
        variance_score = np.ceil(100 * np.sin(np.pi * 60 * std / (60 * std + 1)) + 0.5)
        skewness_score = np.ceil(100 * (1 - sigmoid(skewness, 2, 0.35)) + 0.5)
        kurtosis_score = np.ceil(100 * sigmoid(kurtosis, 2, 0.5) * (1 - sigmoid(kurtosis, 10, 2)) + 0.5)

        return (
            mean_config.create_quality_metric(mean, int(mean_score)),
            variance_config.create_quality_metric(var, int(variance_score)),
            skewness_config.create_quality_metric(skewness, int(skewness_score)),
            kurtosis_config.create_quality_metric(kurtosis, int(kurtosis_score)),
        )
