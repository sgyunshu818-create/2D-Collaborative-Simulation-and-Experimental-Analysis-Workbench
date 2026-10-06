"""Strict experiment configuration with a predeclared development search grid."""

import dataclasses
import hashlib
import json
import math
from pathlib import Path

from . import ExperimentError


@dataclasses.dataclass(frozen=True)
class ExperimentConfig:
    name: str = "synthetic_marker_robustness_v1"
    image_size: int = 128
    dev_seed: int = 171104
    test_seed: int = 910421
    dev_per_class: int = 16
    test_per_class: int = 40
    noise_sigmas: tuple[float, ...] = (25.0, 55.0)
    occlusion_fractions: tuple[float, ...] = (0.2, 0.4)
    baseline_thresholds: tuple[int, ...] = (100, 128, 150)
    polygon_epsilons: tuple[float, ...] = (0.015, 0.025, 0.035)
    circularity_thresholds: tuple[float, ...] = (0.72, 0.8)
    blur_kernels: tuple[int, ...] = (3, 5, 7)

    def data(self):
        return dataclasses.asdict(self)

    @property
    def sha256(self):
        return hashlib.sha256(json.dumps(self.data(), sort_keys=True, separators=(",", ":"),
                                         allow_nan=False).encode()).hexdigest()

    @property
    def conditions(self):
        return (("normal", 0.0),
                *((f"noise_{value:g}", value) for value in self.noise_sigmas),
                *((f"occlusion_{value:g}", value) for value in self.occlusion_fractions))


def config_from_data(data, path="<memory>"):
    def fail(field, message):
        raise ExperimentError(f"{path}: {field}: {message}")

    if not isinstance(data, dict):
        fail("$", "must be a JSON object")
    defaults = ExperimentConfig().data()
    if set(data) - set(defaults):
        fail("$", f"unknown fields: {sorted(set(data) - set(defaults))}")
    values = defaults | data
    if not isinstance(values["name"], str) or not values["name"].strip():
        fail("name", "must be a non-empty string")
    for field, lower, upper in (("image_size", 64, 512), ("dev_seed", 0, 2 ** 63 - 1),
                               ("test_seed", 0, 2 ** 63 - 1), ("dev_per_class", 2, 1000),
                               ("test_per_class", 34, 1000)):
        number = values[field]
        if isinstance(number, bool) or not isinstance(number, int) or not lower <= number <= upper:
            fail(field, f"must be an integer in [{lower}, {upper}]")
    if values["dev_seed"] == values["test_seed"]:
        fail("test_seed", "must differ from dev_seed")
    for field, lower, upper, integer, minimum in (
        ("noise_sigmas", 0, 100, False, 2),
        ("occlusion_fractions", 0, 0.75, False, 2),
        ("baseline_thresholds", 0, 255, True, 1),
        ("polygon_epsilons", 0.004, 0.1, False, 1),
        ("circularity_thresholds", 0, 1.001, False, 1),
        ("blur_kernels", 2, 16, True, 1),
    ):
        sequence = values[field]
        if not isinstance(sequence, (list, tuple)) or not minimum <= len(sequence) <= 6:
            fail(field, f"must contain {minimum} to 6 distinct values")
        for index, number in enumerate(sequence):
            if isinstance(number, bool) or not isinstance(number, (int, float)):
                fail(f"{field}[{index}]", "must be a finite number")
            try:
                valid = math.isfinite(number) and lower < number < upper
            except OverflowError:
                valid = False
            if not valid or (integer and not isinstance(number, int)):
                fail(f"{field}[{index}]", f"must be {'an integer' if integer else 'a number'} in ({lower}, {upper})")
            if field == "blur_kernels" and number % 2 != 1:
                fail(f"{field}[{index}]", "Gaussian kernel must be odd")
        if len(set(sequence)) != len(sequence):
            fail(field, "duplicate strengths or candidates are not accepted")
        values[field] = tuple(sequence)
    return ExperimentConfig(**values)


def load_config(path):
    path = Path(path)
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, ValueError, RecursionError) as error:
        raise ExperimentError(f"{path}: cannot read configuration: {error}") from error
    return config_from_data(data, path)
