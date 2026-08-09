"""Common calibrated geometry for the proposed HOPE-R reference design.

This module is a proposed repair, not a paper-literal implementation. All
calibrated quantities use one explicit weighted empirical measure. In
particular, no pair is allowed to silently select a different sample law.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np


ArrayLike = Sequence[float] | np.ndarray
MatrixLike = Sequence[Sequence[float]] | np.ndarray


def _finite_scalar(value: float, *, name: str) -> float:
    array = np.asarray(value)
    if array.ndim != 0 or array.dtype.kind in {"b", "c", "O", "S", "U", "V"}:
        raise ValueError(f"{name} must be one real scalar")
    result = float(array)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _finite_vector(value: ArrayLike, *, name: str, nonempty: bool = True) -> np.ndarray:
    original = np.asarray(value)
    if original.dtype.kind in {"b", "c", "O", "S", "U", "V"}:
        raise ValueError(f"{name} must contain real numerical values")
    vector = np.asarray(value, dtype=np.float64)
    if vector.ndim != 1 or (nonempty and vector.size == 0):
        qualifier = "nonempty " if nonempty else ""
        raise ValueError(f"{name} must be a {qualifier}one-dimensional array")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite float64 values")
    return np.array(vector, dtype=np.float64, copy=True)


def _finite_matrix(value: MatrixLike, *, name: str) -> np.ndarray:
    original = np.asarray(value)
    if original.dtype.kind in {"b", "c", "O", "S", "U", "V"}:
        raise ValueError(f"{name} must contain real numerical values")
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.ndim != 2 or 0 in matrix.shape:
        raise ValueError(f"{name} must be a nonempty two-dimensional array")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain only finite float64 values")
    return np.array(matrix, dtype=np.float64, copy=True)


def _readonly(value: np.ndarray) -> np.ndarray:
    value.setflags(write=False)
    return value


@dataclass(frozen=True)
class FoldedAffine:
    """The inference-time affine map produced by a BatchNorm fold."""

    weight: np.ndarray
    bias: float
    batch_norm_scale: float


def fold_batch_norm(
    raw_weight: ArrayLike,
    raw_bias: float,
    gamma: float,
    beta: float,
    running_mean: float,
    running_variance: float,
    epsilon: float,
) -> FoldedAffine:
    """Fold scalar-channel BatchNorm into its preceding affine map.

    The exact inference map is

    ``scale = gamma / sqrt(running_variance + epsilon)``,
    ``weight = scale * raw_weight``, and
    ``bias = beta + scale * (raw_bias - running_mean)``.

    ``gamma == 0`` is handled before multiplication. This makes its constant
    ``beta`` boundary deterministic even when an otherwise harmless
    subtraction would overflow. Epsilon may be zero, but the denominator must
    remain strictly positive.
    """

    weight = _finite_vector(raw_weight, name="raw_weight")
    raw_bias_value = _finite_scalar(raw_bias, name="raw_bias")
    gamma_value = _finite_scalar(gamma, name="gamma")
    beta_value = _finite_scalar(beta, name="beta")
    mean_value = _finite_scalar(running_mean, name="running_mean")
    variance_value = _finite_scalar(running_variance, name="running_variance")
    epsilon_value = _finite_scalar(epsilon, name="epsilon")
    if variance_value < 0.0:
        raise ValueError("running_variance must be nonnegative")
    if epsilon_value < 0.0:
        raise ValueError("epsilon must be nonnegative")
    denominator_squared = variance_value + epsilon_value
    if not math.isfinite(denominator_squared):
        raise FloatingPointError("BatchNorm variance plus epsilon was non-finite")
    if denominator_squared <= 0.0:
        raise ValueError("running_variance plus epsilon must be strictly positive")

    if gamma_value == 0.0:
        return FoldedAffine(
            weight=_readonly(np.zeros_like(weight, dtype=np.float64)),
            bias=beta_value,
            batch_norm_scale=0.0,
        )

    scale = gamma_value / math.sqrt(denominator_squared)
    if not math.isfinite(scale):
        raise FloatingPointError("folded BatchNorm scale was non-finite")
    with np.errstate(over="raise", invalid="raise"):
        try:
            folded_weight = scale * weight
            folded_bias = beta_value + scale * (raw_bias_value - mean_value)
        except FloatingPointError as error:
            raise FloatingPointError("BatchNorm folding overflowed float64") from error
    if not np.all(np.isfinite(folded_weight)) or not math.isfinite(folded_bias):
        raise FloatingPointError("BatchNorm folding produced a non-finite affine map")
    return FoldedAffine(
        weight=_readonly(np.asarray(folded_weight, dtype=np.float64)),
        bias=float(folded_bias),
        batch_norm_scale=float(scale),
    )


@dataclass(frozen=True)
class FunctionalNeuron:
    """A physical scalar ReLU feature and its complete outgoing vector."""

    incoming_weight: np.ndarray
    bias: float
    outgoing_weight: np.ndarray

    def __post_init__(self) -> None:
        incoming = _readonly(
            _finite_vector(self.incoming_weight, name="incoming_weight")
        )
        outgoing = _readonly(
            _finite_vector(self.outgoing_weight, name="outgoing_weight")
        )
        bias = _finite_scalar(self.bias, name="bias")
        object.__setattr__(self, "incoming_weight", incoming)
        object.__setattr__(self, "outgoing_weight", outgoing)
        object.__setattr__(self, "bias", bias)

    @property
    def augmented_weight(self) -> np.ndarray:
        """Return ``[incoming_weight, bias]`` as an immutable float64 vector."""

        augmented = np.concatenate(
            [self.incoming_weight, np.array([self.bias], dtype=np.float64)]
        )
        return _readonly(augmented)

    def activations(self, samples: MatrixLike) -> np.ndarray:
        """Evaluate the scalar ReLU feature on a finite sample matrix."""

        matrix = _finite_matrix(samples, name="samples")
        if matrix.shape[1] != self.incoming_weight.size:
            raise ValueError("sample width must match incoming_weight length")
        with np.errstate(over="raise", invalid="raise"):
            try:
                preactivation = matrix @ self.incoming_weight + self.bias
            except FloatingPointError as error:
                raise FloatingPointError("affine evaluation overflowed float64") from error
        if not np.all(np.isfinite(preactivation)):
            raise FloatingPointError("affine evaluation produced non-finite values")
        return np.maximum(preactivation, 0.0).astype(np.float64, copy=False)

    def contributions(self, samples: MatrixLike) -> np.ndarray:
        """Evaluate ``ReLU(w.T x + b) * outgoing_weight``."""

        activation = self.activations(samples)
        with np.errstate(over="raise", invalid="raise"):
            try:
                result = activation[:, np.newaxis] * self.outgoing_weight
            except FloatingPointError as error:
                raise FloatingPointError("functional contribution overflowed float64") from error
        if not np.all(np.isfinite(result)):
            raise FloatingPointError("functional contribution was non-finite")
        return np.asarray(result, dtype=np.float64)

    def positive_rescaled(self, scale: float) -> FunctionalNeuron:
        """Return the same ReLU function under positive incoming rescaling."""

        value = _finite_scalar(scale, name="scale")
        if value <= 0.0:
            raise ValueError("scale must be strictly positive")
        with np.errstate(over="raise", invalid="raise", under="ignore"):
            try:
                incoming = self.incoming_weight * value
                bias = self.bias * value
                outgoing = self.outgoing_weight / value
            except FloatingPointError as error:
                raise FloatingPointError("positive rescaling overflowed float64") from error
        return FunctionalNeuron(incoming, bias, outgoing)


@dataclass(frozen=True)
class CalibrationMeasure:
    """One declared weighted empirical probability measure over input rows."""

    samples: np.ndarray
    weights: np.ndarray
    label: str = "declared_weighted_empirical"

    def __post_init__(self) -> None:
        samples = _readonly(_finite_matrix(self.samples, name="samples"))
        weights = _finite_vector(self.weights, name="weights")
        if weights.size != samples.shape[0]:
            raise ValueError("weights length must equal the number of sample rows")
        if np.any(weights < 0.0):
            raise ValueError("weights must be nonnegative")
        try:
            total = math.fsum(float(value) for value in weights)
        except OverflowError as error:
            raise ValueError("weights must have a finite positive sum") from error
        if not math.isfinite(total) or total <= 0.0:
            raise ValueError("weights must have a finite positive sum")
        probabilities = weights / total
        if not np.all(np.isfinite(probabilities)):
            raise FloatingPointError("normalized calibration weights were non-finite")
        label = str(self.label).strip()
        if not label:
            raise ValueError("label must be nonempty")
        object.__setattr__(self, "samples", samples)
        object.__setattr__(self, "weights", _readonly(probabilities))
        object.__setattr__(self, "label", label)

    @classmethod
    def uniform(cls, samples: MatrixLike, *, label: str = "uniform_empirical") -> CalibrationMeasure:
        """Construct the explicitly uniform empirical measure on ``samples``."""

        matrix = _finite_matrix(samples, name="samples")
        return cls(
            matrix,
            np.ones(matrix.shape[0], dtype=np.float64),
            label=label,
        )

    def activations(self, neuron: FunctionalNeuron) -> np.ndarray:
        """Evaluate one neuron under this measure's shared input rows."""

        if not isinstance(neuron, FunctionalNeuron):
            raise TypeError("neuron must be a FunctionalNeuron")
        return neuron.activations(self.samples)


def relu_self_kernel(beta: float, gamma: float) -> float:
    """Return ``E[ReLU(Y)^2]`` for ``Y ~ N(beta, gamma^2)``.

    This is the analytic biased self-kernel. The exact deterministic
    ``gamma == 0`` limit is ``ReLU(beta)^2``; it is not replaced by a small
    positive scale.
    """

    mean_value = _finite_scalar(beta, name="beta")
    scale_value = abs(_finite_scalar(gamma, name="gamma"))
    if scale_value == 0.0:
        result = max(0.0, mean_value) ** 2
    else:
        standardized = mean_value / scale_value
        if not math.isfinite(standardized):
            result = mean_value * mean_value + scale_value * scale_value if mean_value > 0.0 else 0.0
        else:
            cdf = 0.5 * math.erfc(-standardized / math.sqrt(2.0))
            pdf = math.exp(-0.5 * standardized * standardized) / math.sqrt(
                2.0 * math.pi
            )
            result = (
                (scale_value * scale_value + mean_value * mean_value) * cdf
                + mean_value * scale_value * pdf
            )
            roundoff = 128.0 * np.finfo(np.float64).eps * max(
                1.0,
                abs((scale_value * scale_value + mean_value * mean_value) * cdf),
                abs(mean_value * scale_value * pdf),
            )
            if result < 0.0 and result >= -roundoff:
                result = 0.0
    if not math.isfinite(result):
        raise FloatingPointError("ReLU self-kernel was non-finite")
    if result < 0.0:
        raise FloatingPointError("ReLU self-kernel was materially negative")
    return float(result)


def calibrated_cross_kernel(
    measure: CalibrationMeasure,
    first: FunctionalNeuron,
    second: FunctionalNeuron,
) -> float:
    """Return the exact weighted empirical ReLU cross-kernel.

    Exactness here is with respect to the rows and normalized weights stored
    by ``measure``. It is not a population-law claim.
    """

    if not isinstance(measure, CalibrationMeasure):
        raise TypeError("measure must be a CalibrationMeasure")
    first_values = measure.activations(first)
    second_values = measure.activations(second)
    with np.errstate(over="raise", invalid="raise"):
        try:
            value = float(np.dot(measure.weights, first_values * second_values))
        except FloatingPointError as error:
            raise FloatingPointError("calibrated cross-kernel overflowed float64") from error
    if not math.isfinite(value):
        raise FloatingPointError("calibrated cross-kernel was non-finite")
    if value < 0.0:
        raise FloatingPointError("ReLU calibrated cross-kernel was negative")
    return value


@dataclass(frozen=True)
class PSDValidation:
    """Numerical certificate returned after explicit PSD validation."""

    minimum_eigenvalue: float
    tolerance: float
    symmetry_error: float
    dimension: int


def validate_psd(matrix: MatrixLike, *, tolerance: float | None = None) -> PSDValidation:
    """Validate that a finite square float64 matrix is symmetric and PSD.

    A small negative eigenvalue within the declared roundoff tolerance is
    accepted and reported. Material asymmetry or negative curvature raises
    ``ValueError``.
    """

    value = _finite_matrix(matrix, name="matrix")
    if value.shape[0] != value.shape[1]:
        raise ValueError("matrix must be square")
    scale = max(1.0, float(np.linalg.norm(value, ord=np.inf)))
    if not math.isfinite(scale):
        raise FloatingPointError("PSD validation scale was non-finite")
    if tolerance is None:
        tolerance_value = 512.0 * np.finfo(np.float64).eps * scale * value.shape[0]
    else:
        tolerance_value = _finite_scalar(tolerance, name="tolerance")
        if tolerance_value < 0.0:
            raise ValueError("tolerance must be nonnegative")
    symmetry_error = float(np.max(np.abs(value - value.T)))
    if symmetry_error > tolerance_value:
        raise ValueError(
            "matrix was not symmetric within the declared PSD tolerance"
        )
    symmetric = 0.5 * (value + value.T)
    try:
        eigenvalues = np.linalg.eigvalsh(symmetric)
    except np.linalg.LinAlgError as error:
        raise ValueError("PSD eigendecomposition did not converge") from error
    minimum = float(eigenvalues[0])
    if minimum < -tolerance_value:
        raise ValueError(
            f"matrix was not positive semidefinite: minimum eigenvalue {minimum}"
        )
    return PSDValidation(
        minimum_eigenvalue=minimum,
        tolerance=float(tolerance_value),
        symmetry_error=symmetry_error,
        dimension=value.shape[0],
    )


def calibrated_activation_gram(
    measure: CalibrationMeasure,
    neurons: Sequence[FunctionalNeuron],
) -> np.ndarray:
    """Return and explicitly PSD-validate one shared calibrated ReLU Gram."""

    if not isinstance(measure, CalibrationMeasure):
        raise TypeError("measure must be a CalibrationMeasure")
    neuron_list = list(neurons)
    if not neuron_list:
        raise ValueError("neurons must be nonempty")
    if any(not isinstance(neuron, FunctionalNeuron) for neuron in neuron_list):
        raise TypeError("every neuron must be a FunctionalNeuron")
    activations = np.column_stack(
        [measure.activations(neuron) for neuron in neuron_list]
    ).astype(np.float64, copy=False)
    root_weights = np.sqrt(measure.weights)
    with np.errstate(over="raise", invalid="raise"):
        try:
            weighted = activations * root_weights[:, np.newaxis]
            gram = weighted.T @ weighted
        except FloatingPointError as error:
            raise FloatingPointError("calibrated activation Gram overflowed float64") from error
    if not np.all(np.isfinite(gram)):
        raise FloatingPointError("calibrated activation Gram was non-finite")
    gram = np.asarray(0.5 * (gram + gram.T), dtype=np.float64)
    validate_psd(gram)
    return gram


def zero_bias_cross_kernel_approximation(
    first_self_kernel: float,
    second_self_kernel: float,
    correlation: float,
) -> float:
    """Return the explicitly labeled zero-bias angular approximation.

    The self-kernels may be biased. Therefore this function remains an
    approximation unless both underlying Gaussian means are zero.
    """

    first = _finite_scalar(first_self_kernel, name="first_self_kernel")
    second = _finite_scalar(second_self_kernel, name="second_self_kernel")
    rho = _finite_scalar(correlation, name="correlation")
    if first < 0.0 or second < 0.0:
        raise ValueError("self-kernels must be nonnegative")
    if rho < -1.0 or rho > 1.0:
        raise ValueError("correlation must lie in [-1, 1]")
    angular = (
        math.sqrt(max(0.0, 1.0 - rho * rho))
        + (math.pi - math.acos(rho)) * rho
    ) / math.pi
    value = angular * math.sqrt(first) * math.sqrt(second)
    if not math.isfinite(value):
        raise FloatingPointError("zero-bias cross-kernel approximation was non-finite")
    return float(value)


def functional_capacity(self_kernel: float, outgoing_weight: ArrayLike) -> float:
    """Return ``sqrt(K_ii) * ||outgoing_weight||_2``."""

    kernel = _finite_scalar(self_kernel, name="self_kernel")
    outgoing = _finite_vector(outgoing_weight, name="outgoing_weight")
    if kernel < 0.0:
        raise ValueError("self_kernel must be nonnegative")
    norm = float(np.linalg.norm(outgoing))
    value = math.sqrt(kernel) * norm
    if not math.isfinite(value):
        raise FloatingPointError("functional capacity was non-finite")
    return value


def neuron_capacities(
    activation_gram: MatrixLike,
    outgoing_weights: MatrixLike,
) -> np.ndarray:
    """Return all functional capacities after explicit Gram PSD validation."""

    gram = _finite_matrix(activation_gram, name="activation_gram")
    certificate = validate_psd(gram)
    outgoing = _finite_matrix(outgoing_weights, name="outgoing_weights")
    if outgoing.shape[0] != certificate.dimension:
        raise ValueError("outgoing_weights row count must match Gram dimension")
    diagonal = np.diag(gram)
    if np.any(diagonal < -certificate.tolerance):
        raise ValueError("activation Gram has a materially negative diagonal")
    norms = np.linalg.norm(outgoing, axis=1)
    with np.errstate(over="raise", invalid="raise"):
        try:
            capacities = np.sqrt(np.maximum(diagonal, 0.0)) * norms
        except FloatingPointError as error:
            raise FloatingPointError("neuron capacities overflowed float64") from error
    if not np.all(np.isfinite(capacities)):
        raise FloatingPointError("neuron capacities were non-finite")
    return np.asarray(capacities, dtype=np.float64)
