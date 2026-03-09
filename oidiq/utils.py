from dataclasses import dataclass
from typing import Any, Callable, Dict, Tuple
import cv2
from torch import device
import torch

import numpy as np


@dataclass
class QualityMetric:
    name: str
    raw_value: float
    score: int
    description: str = ""

    def __str__(self):
        return f"{self.name}: {self.score} ({self.raw_value:.5f}) {self.description}"

    def __repr__(self):
        return str(self)


class OIDIQConfig(dict):
    def __init__(
        self,
        initial_data: Dict | None = None,
        description: str | None = None,
        _is_placeholder: bool = False,
    ):
        self._is_placeholder = _is_placeholder
        initial_data = initial_data or {}
        initial_data["description"] = description or initial_data.get("description", "")
        super().__init__(deep_copy_dict(initial_data) or {})


class QualityMetricConfig(OIDIQConfig):
    DEFAULT_NAME = "Unnamed metric"

    def __init__(
        self,
        initial_data: Dict | None = None,
        name: str | None = None,
        description: str | None = None,
        _is_placeholder: bool = False,
    ):
        initial_data = initial_data or {}
        if name is not None:
            initial_data["name"] = name

        super().__init__(initial_data, description=description, _is_placeholder=_is_placeholder)

    def get_name(self) -> str:
        return self["name"] if "name" in self else self.DEFAULT_NAME

    def create_quality_metric(
        self, raw_value: float | np.floating, score: int | float, description: str | None = None
    ) -> QualityMetric:
        return QualityMetric(
            name=self.get_name(),
            raw_value=float(raw_value),
            score=int(np.clip(score, 0, 100)),
            description=description or self["description"],
        )


def deep_copy_dict(d: Any) -> Any:
    if not isinstance(d, dict):
        if hasattr(d, "copy"):
            return d.copy()
        else:
            return d
    r = {}
    for k, v in d.items():
        r[k] = deep_copy_dict(v)
    return r


def sigmoid(x: float | np.number, x_0: float | np.number, w: float | np.number) -> float:
    return 1 / (1 + np.exp((x_0 - x) / w))


def full_sigmoid(
    x: float | np.number,
    h: float | np.number,
    a: float | np.number,
    s: float | np.number,
    x_0: float | np.number,
    w: float | np.number,
) -> float:
    return h * (a + s * sigmoid(x, x_0, w))


def scalar_conversion(
    x: float | np.number,
    h: float | np.number,
    a: float | np.number,
    s: float | np.number,
    x_0: float | np.number,
    w: float | np.number,
) -> int:
    return int(np.clip(np.round(full_sigmoid(x, h, a, s, x_0, w)), 0, 100))


def create_mask(
    img: np.ndarray,
    session,
    config: OIDIQConfig,
    default_face_mask=True,
    default_dark_face_background=False,
    default_foreground=False,
) -> np.ndarray:
    mask = np.ones_like(img, dtype=bool)
    if config.get("mask_face", default_face_mask):
        face_mask = session.get_normalized_face_mask()
        mask &= ~face_mask.astype(bool)
    if config.get("mask_dark_face_background", default_dark_face_background):
        face_background_mask = session.get_normalized_face_background_mask()
        mask &= ~face_background_mask.astype(bool)
    if config.get("mask_foreground", default_foreground):
        foreground_mask = session.get_normalized_foreground_mask()
        mask &= ~foreground_mask.astype(bool)

    return mask


def map_to_image_borders(
    corners: np.ndarray, w: int, h: int, method: str = "clip", allow_outside: bool = True
) -> np.ndarray:
    if method == "clip":
        return map_to_image_borders_clip(corners, w, h).astype(np.int32)
    elif method == "linear":
        return map_to_image_borders_linear(corners, w, h).astype(np.int32)
    elif method == "none":
        if allow_outside:
            return corners.astype(np.int32)
        else:
            raise ValueError(
                "Mapping method 'none' not allowed, corners must be mapped inside image borders either by 'clip' or 'linear'."
            )
    else:
        raise ValueError(f"Unknown method {method} for mapping corners to image borders.")


def map_to_image_borders_clip(corners: np.ndarray, w: int, h: int) -> np.ndarray:
    for i in range(4):
        x, y = corners[i]
        x = np.clip(x, 0, w - 1)
        y = np.clip(y, 0, h - 1)
        corners[i] = (x, y)
    return corners


def map_to_image_borders_linear(corners: np.ndarray, w: int, h: int) -> np.ndarray:
    r = np.zeros((4, 2), dtype=np.int32)
    for i in range(4):
        x, y = corners[i]
        if i == 0:
            other_x_x, other_x_y = corners[1, 0], corners[1, 1]
            other_y_x, other_y_y = corners[3, 0], corners[3, 1]
        elif i == 1:
            other_x_x, other_x_y = corners[0, 0], corners[0, 1]
            other_y_x, other_y_y = corners[2, 0], corners[2, 1]
        elif i == 2:
            other_x_x, other_x_y = corners[3, 0], corners[3, 1]
            other_y_x, other_y_y = corners[1, 0], corners[1, 1]
        else:  # i ==3
            other_x_x, other_x_y = corners[2, 0], corners[2, 1]
            other_y_x, other_y_y = corners[0, 0], corners[0, 1]

        while x < 0 or x >= w or y < 0 or y >= h:
            if x < 0 or x >= w:
                m = (other_x_y - y) / (other_x_x - x)
                b = y - m * x
                if x < 0:
                    x = 0
                    y = b
                else:
                    x = w - 1
                    y = m * x + b
            if y < 0 or y >= h:
                m = (other_y_x - x) / (other_y_y - y)
                b = x - m * y
                if y < 0:
                    y = 0
                    x = b
                else:
                    y = h - 1
                    x = m * y + b
        r[i, 0] = int(x)
        r[i, 1] = int(y)
    return r


def resize_keep_ratio(img: np.ndarray, target_size: Tuple[int, int], padding: float) -> Tuple[np.ndarray, int, int]:
    target_w, target_h = target_size
    h, w = img.shape[0:2]
    ratio = h / w
    if target_h / target_w > ratio:
        pad_h = int((target_h * w / target_w - h) / 2)
        pad_w = 0
    elif target_h / target_w < ratio:
        pad_h = 0
        pad_w = int((target_w * h / target_h - w) / 2)
    else:
        pad_h = 0
        pad_w = 0
    pad_h = int(padding * max(h, w)) + pad_h
    pad_w = int(padding * max(h, w)) + pad_w
    img = cv2.copyMakeBorder(
        img,
        pad_h,
        pad_h,
        pad_w,
        pad_w,
        cv2.BORDER_CONSTANT,
        value=[0, 0, 0],
    )

    img = cv2.resize(img, (target_w, target_h))
    pad_w = pad_w * target_w / (w + 2 * pad_w)
    pad_h = pad_h * target_h / (h + 2 * pad_h)
    return img.astype(np.float32), int(pad_w), int(pad_h)


def get_device(config: OIDIQConfig) -> device:
    if "device" in config and config["device"] in ["cpu", "cuda", "mps"]:
        return device(config["device"])
    elif torch.cuda.is_available():
        return device("cuda")
    elif torch.backends.mps.is_available():
        return device("mps")
    else:
        return device("cpu")


def calc_area_of_triangle(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    det = x1 * y2 + x2 * y3 + x3 * y1 - x2 * y1 - x3 * y2 - x1 * y3
    return abs(det) / 2.0


def calculate_4_point_polygon_area(corners: np.ndarray) -> float:
    p1, p2, p3, p4 = corners
    area1 = calc_area_of_triangle(p1, p2, p3)
    area2 = calc_area_of_triangle(p1, p3, p4)
    return area1 + area2


def creates(*names: str):
    """
    Specifies that a function creates certain outputs. Either QualityMetrics or preprocessor results.
    If multiple names are specified, the function must return a tuple of results in the same order.
    """

    def decorator(func):
        func._creates = names
        return func

    return decorator

def creates_see_class(func):
    """
    Specifies that a function creates an output which is specified by the class.
    Put the @creates decorator on the class and @creates_see_class on the function to link them.
    """
    func._creates_see_class = True  
    return func


def config(*config_keys: str):
    """
    Specifies the config keys that a function requires.
    If not specified, the creates decorator is used to determine the config keys.
    If multiple config keys are specified, the function will receive multiple OIDIQConfig objects as arguments.
    """

    def decorator(func):
        func._config_keys = config_keys
        return func

    return decorator


def batching(batch_size: int | None):
    """
    Specifies that a function should be executed in batches of the given size.
    Instead of scalar inputs, the function will receive batched inputs as lists of the given size.
    The function should return lists of outputs of the same size.
    Also, Instead of a single OIDIQConfig, the function will receive a list of OIDIQConfig objects of the same size.
    Instead of returning a single QualityMetric, the function should return a list of QualityMetrics.
    """

    def decorator(func):
        func._batch_size = batch_size
        return func

    return decorator
