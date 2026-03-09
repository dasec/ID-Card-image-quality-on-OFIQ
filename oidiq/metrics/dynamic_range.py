from typing import Tuple
from ..session import OIDIQSession, OIDIQMetricCreator
from ..utils import creates, QualityMetric, sigmoid, QualityMetricConfig
import numpy as np


class DynamicRange(OIDIQMetricCreator):
    @creates("dynamic_range")
    def dynamic_range(self, session: OIDIQSession, config: QualityMetricConfig) -> QualityMetric:
        nl = session.get_normalized_luminance_histogram()
        nl = nl[nl > 0]
        dr = -np.sum(nl * np.log2(nl))
        dr_score = np.clip(np.round(12.5 * dr), 0, 100)
        return config.create_quality_metric(dr, int(dr_score))
