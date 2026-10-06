"""Both predictors receive only uint8 pixels and frozen numeric parameters.

The baseline uses a fixed threshold. The proposed improvement applies a
Gaussian filter and Otsu thresholding. Both use the same contour classifier.
"""

from dataclasses import asdict, dataclass
from itertools import product
import math

import cv2
import numpy as np


@dataclass(frozen=True)
class Parameters:
    algorithm: str
    epsilon: float
    circularity: float
    threshold: int = 128
    blur_kernel: int = 5
    min_area: float = 80.0

    def data(self):
        return asdict(self)


def candidates(config, algorithm):
    if algorithm == "baseline":
        return [Parameters(algorithm, epsilon, circularity, threshold=threshold)
                for threshold, epsilon, circularity in product(config.baseline_thresholds,
                    config.polygon_epsilons, config.circularity_thresholds)]
    if algorithm == "improved":
        return [Parameters(algorithm, epsilon, circularity, blur_kernel=kernel)
                for kernel, epsilon, circularity in product(config.blur_kernels,
                    config.polygon_epsilons, config.circularity_thresholds)]
    raise ValueError(f"unknown algorithm {algorithm!r}")


def predict(image: np.ndarray, parameters: Parameters) -> str:
    """Classify a pixel array; filenames, manifest rows and labels are excluded."""
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.ndim != 2 or image.size == 0:
        raise ValueError("image must be a non-empty uint8 grayscale array")
    if parameters.algorithm == "baseline":
        _, binary = cv2.threshold(image, parameters.threshold, 255, cv2.THRESH_BINARY_INV)
    elif parameters.algorithm == "improved":
        blurred = cv2.GaussianBlur(image, (parameters.blur_kernel, parameters.blur_kernel), 0)
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    else:
        raise ValueError(f"unknown algorithm {parameters.algorithm!r}")
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return "unknown"
    contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    if area < parameters.min_area or perimeter == 0:
        return "unknown"
    vertices = len(cv2.approxPolyDP(contour, parameters.epsilon * perimeter, True))
    if vertices == 3:
        return "triangle"
    if vertices == 4:
        return "square"
    circularity = 4 * math.pi * area / (perimeter * perimeter)
    return "circle" if circularity >= parameters.circularity else "unknown"
