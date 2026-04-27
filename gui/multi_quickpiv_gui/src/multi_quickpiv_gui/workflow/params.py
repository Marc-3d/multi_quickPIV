"""Parameter models for GUI and processing workflows."""

from __future__ import annotations

from dataclasses import dataclass, field

CorrAlg = str
Size2D = tuple[int, int]


@dataclass(slots=True)
class PIVRunParams:
    """Parameters that directly control the PIV computation."""

    inter_size: Size2D = (64, 64)
    search_margin: Size2D = (128, 128)
    step: Size2D = (32, 32)
    compute_sn: bool = True
    corr_alg: CorrAlg = "nsqecc"

    def validate(self) -> None:
        """Validate the full run configuration."""
        self._validate_size("inter_size", self.inter_size)
        self._validate_size("search_margin", self.search_margin)
        self._validate_size("step", self.step)

    @staticmethod
    def _validate_size(name: str, value: Size2D) -> None:
        """Validate a 2D integer size tuple."""
        if len(value) != 2:
            raise ValueError(f"{name} must contain exactly 2 integers.")

        h, w = value
        if h <= 0 or w <= 0:
            raise ValueError(f"{name} values must be greater than 0.")


@dataclass(slots=True)
class MedianDespikeParams:
    """Configuration for median-despike post-processing."""

    enabled: bool = False
    ksize: int = 3
    threshold: float = 3.5

    def validate(self) -> None:
        """Validate median-despike settings."""
        if self.ksize < 1:
            raise ValueError("Median-despike window size must be at least 1.")
        if self.ksize % 2 == 0:
            raise ValueError("Median-despike window size must be odd.")
        if self.threshold <= 0:
            raise ValueError("Median-despike threshold must be greater than 0.")


@dataclass(slots=True)
class SNFilterParams:
    """Configuration for signal-to-noise post-processing."""

    enabled: bool = False
    minimum: float = 1.0

    def validate(self, *, compute_sn: bool) -> None:
        """Validate signal-to-noise filtering settings."""
        if self.enabled and not compute_sn:
            raise ValueError("SN filtering requires compute_sn=True.")
        if self.enabled and self.minimum <= 0:
            raise ValueError("SN minimum threshold must be greater than 0.")


@dataclass(slots=True)
class PostProcessParams:
    """Collection of post-processing settings for a computed vector field."""

    median_despike: MedianDespikeParams = field(
        default_factory=MedianDespikeParams
    )
    sn_filter: SNFilterParams = field(default_factory=SNFilterParams)

    def validate(self, *, run: PIVRunParams) -> None:
        """Validate all post-processing settings."""
        self.median_despike.validate()
        self.sn_filter.validate(compute_sn=run.compute_sn)


@dataclass(slots=True)
class WorkflowParams:
    """Complete workflow configuration for one analysis run."""

    run: PIVRunParams = field(default_factory=PIVRunParams)
    postprocess: PostProcessParams = field(default_factory=PostProcessParams)

    def validate(self) -> None:
        """Validate the complete workflow configuration."""
        self.run.validate()
        self.postprocess.validate(run=self.run)