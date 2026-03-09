from typing import Dict, List, Optional, Tuple
import numpy as np
import yaml
from .session import OIDIQSession, OIDIQPreprocessor, OIDIQMetricCreator, OIDIQBatchSession
from .preprocessors import *
from .metrics import *


BASIC_METRICS: Tuple[type[OIDIQMetricCreator], ...] = (
    CompressionArtifacts,
    DynamicRange,
    IlluminationUniformity,
    LuminanceMoments,
    Sharpness,
    Exposure,
)

BASIC_PREPROCESSORS: Tuple[type[OIDIQPreprocessor], ...] = (
    Cropping,
    FaceMasking,
    NormalizedLuminance,
    NormalizedLuminanceHistogram,
    IDCardCornerDetection,
    Normalizing,
    ForegroundMasking,
    FaceBackgroundMasking,
)


class OIDIQSessionFactory:
    def __init__(self, config_path: str = "config.yaml", verbose: bool = False, use_default: bool = True, **kwargs):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        self.config.update(kwargs)

        #self.preprocessors = [cls(self.config) for cls in BASIC_PREPROCESSORS]
        #self.metric_creators = [cls(self.config) for cls in BASIC_METRICS]
        self.preprocessors: Dict[str, OIDIQPreprocessor] = {}
        self.metric_creators: Dict[str, OIDIQMetricCreator] = {}
        if use_default:
            for cls in BASIC_PREPROCESSORS:
                self.register_preprocessor(cls)
            for cls in BASIC_METRICS:
                self.register_metric_creator(cls)
        self.verbose = verbose

    def register_preprocessor(self, preprocessor: type[OIDIQPreprocessor] | OIDIQPreprocessor, *use: str):
        if isinstance(preprocessor, type):
            preprocessor_obj = preprocessor(self.config)
        else:
            preprocessor_obj = preprocessor
        if use:
            for name in use:
                self.preprocessors[name] = preprocessor_obj
        else:
            for name in preprocessor_obj.creates():
                self.preprocessors[name] = preprocessor_obj

    def register_metric_creator(self, metric_creator: type[OIDIQMetricCreator] | OIDIQMetricCreator, *use: str):
        if isinstance(metric_creator, type):
            metric_creator_obj = metric_creator(self.config)
        else:
            metric_creator_obj = metric_creator
        if use:
            for name in use:
                self.metric_creators[name] = metric_creator_obj
        else:
            for name in metric_creator_obj.creates():
                self.metric_creators[name] = metric_creator_obj

    def create_session(self, img: np.ndarray | str) -> OIDIQSession:
        return OIDIQSession(
            img,
            preprocessors=self.preprocessors,
            metric_creators=self.metric_creators,
            verbose=self.verbose,
        )
    
    def create_batch_session(self, *imgs: np.ndarray | str) -> OIDIQBatchSession:
        return OIDIQBatchSession(
            imgs,
            preprocessors=self.preprocessors,
            metric_creators=self.metric_creators,
            verbose=self.verbose,
        )
    
    def registered_preprocessors(self) -> List[str]:
        return list(self.preprocessors.keys())
    
    def registered_metric_creators(self) -> List[str]:
        return list(self.metric_creators.keys())
