from typing import Tuple
from ..session import OIDIQSession, OIDIQMetricCreator
from ..utils import config, creates, QualityMetric, QualityMetricConfig, scalar_conversion
import onnxruntime as ort
import numpy as np
import cv2

class CompressionArtifacts(OIDIQMetricCreator):
    @config("compression_artifacts")
    def init_config(self, config: QualityMetricConfig):
        config["ort_session"] = ort.InferenceSession(
            config["model_path"],
            providers=["CPUExecutionProvider"],
        )

    @creates("compression_artifacts")
    def compression_artifacts(self, session: OIDIQSession, config: QualityMetricConfig) -> QualityMetric:
        img = session.get_cropped_image()

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)

        mean = np.array([123.7, 116.3, 103.5], dtype=np.float32)
        std = np.array([58.4, 57.1, 57.4], dtype=np.float32)
        img -= mean
        img /= std

        crops = []
        for xs in range(0, img.shape[1] // 248 + 1):
            for ys in range(0, img.shape[0] // 248 + 1):
                start_y = ys * 248
                end_y = min(start_y + 248, img.shape[0])
                start_x = xs * 248
                end_x = min(start_x + 248, img.shape[1])
                if end_y - start_y == 248 and end_x - start_x == 248:
                    img_crop = img[start_y:end_y, start_x:end_x, :]
                    crops.append(img_crop)

        # Batch crops through ONNX Runtime
        if crops:
            blobs = [cv2.dnn.blobFromImage(crop) for crop in crops]
            batched_blobs = np.vstack(blobs)  # Stack all blobs into a single batch

            ort_session: ort.InferenceSession = config["ort_session"]
            input_name = ort_session.get_inputs()[0].name
            out = ort_session.run(None, {input_name: batched_blobs})
            
            raw_scores = out[0].flatten() # type: ignore
            scalar_scores = [scalar_conversion(score, 6.1, -0.64, 18.2, 0.87, 0.02) for score in raw_scores]
            
            raw_score = float(np.mean(raw_scores))
            scalar_score = int(np.mean(scalar_scores))
        else:
            raw_score = 0
            scalar_score = 0

        return config.create_quality_metric(raw_score, scalar_score)