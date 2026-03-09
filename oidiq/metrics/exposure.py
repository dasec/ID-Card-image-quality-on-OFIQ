from typing import Tuple
from ..session import OIDIQSession, OIDIQMetricCreator
from ..utils import creates, QualityMetric, sigmoid, QualityMetricConfig
import numpy as np


class Exposure(OIDIQMetricCreator):
    @creates("under_exposure", "over_exposure")
    def over_under_exposure_scores(
        self, session: OIDIQSession, under_config: QualityMetricConfig, over_config: QualityMetricConfig
    ) -> Tuple[QualityMetric, QualityMetric]:
        under_threshold = under_config["threshold"]
        over_threshold = over_config["threshold"]
        nl = session.get_normalized_luminance_histogram()
        n_under = np.sum(nl[:under_threshold])
        n_over = np.sum(nl[-over_threshold:])

        under_score = np.clip(np.round(120 * (0.832 - sigmoid(n_under, 0.92, 0.05))), 0, 100)
        over_score = np.clip(np.round(100 * (1 / (n_over + 0.01))), 0, 100)

        return under_config.create_quality_metric(n_under, int(under_score)), over_config.create_quality_metric(
            n_over, int(over_score)
        )

