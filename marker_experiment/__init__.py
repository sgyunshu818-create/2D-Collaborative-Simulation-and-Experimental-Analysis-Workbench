"""Independent, pixel-only classification of synthetic geometric markers."""

GENERATOR_VERSION = 1
LABELS = ("circle", "square", "triangle")
PREDICTIONS = LABELS + ("unknown",)


class ExperimentError(ValueError):
    """A readable configuration, dataset or experiment error."""
