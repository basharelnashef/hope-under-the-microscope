#!/usr/bin/env python3
"""Fail-closed Part 3 audit of BN marginals and Gaussian surrogates.

The runtime path lazily imports Torch. Pure numerical helpers and output
validation require only NumPy and the Python standard library, which keeps the
focused unit tests usable in environments without Torch.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, NoReturn, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RESULT_DIR = ROOT / "artifacts" / "runs" / "part_03" / "manual"
DEFAULT_CHECKPOINT = (
    ROOT / "artifacts" / "runs" / "vgg8_cifar10_adapted" / "vgg8_cifar10.pt"
)
DEFAULT_DATA_DIR = (
    ROOT / "experiments" / "paper_reproduction" / "vgg8_cifar10" / "data"
)
PAPER_MANIFEST_PATH = ROOT / "paper" / "manifest.json"
PAPER_SHA256 = "8106a86c0c72aee05b0fbcc5e13095edc03fe725538b1bfa5c50c18ea806067b"
AUDIT_ID = "part03_data_free_gaussian_marginal"
AUDIT_SCHEMA_VERSION = 1

DEFAULT_SEED = 20260802
DEFAULT_MAX_EXAMPLES = 2000
DEFAULT_BATCH_SIZE = 64
DEFAULT_SPATIAL_POSITIONS = 16
DEFAULT_CHANNELS_PER_LAYER = 16
DEFAULT_PAIRS_PER_LAYER = 8
DEFAULT_BVN_ORDER = 96

LAYER_NAMES = ("conv0", "conv2", "conv5", "dense")
LAYER_MODULE_PATHS = {
    "conv0": "features.1",
    "conv2": "features.9",
    "conv5": "features.19",
    "dense": "bn_b",
}
CLAIM_IDS = (
    "P03-DATA-FREE-SCOPE",
    "P03-BN-MARGINAL-CONSTRAINTS",
    "P03-EQ76-COLUMN-SPACE",
    "P03-BN-NOT-JOINT",
    "P03-GAUSSIAN-SURROGATE-BOUNDARY",
    "P03-CHECKPOINT-GAUSSIAN-DIAGNOSTIC",
)

TOLERANCES = {
    "decomposition_atol": 2e-12,
    "decomposition_rtol": 2e-11,
    "common_map_atol": 3e-6,
    "common_map_rtol": 3e-5,
    "shuffle_marginal_atol": 2e-13,
    "shuffle_marginal_rtol": 2e-13,
    "shuffle_change_floor": 1e-8,
    "formula_atol": 2e-11,
    "formula_rtol": 2e-10,
    "variance_floor": 1e-15,
}

METHOD_ORIGINS = {
    "checkpoint_measurement": {
        "label": "independent_checkpoint_measurement",
        "evidence_label": "verified_fact",
        "description": (
            "Bounded deterministic measurements from an adapted VGG-8 checkpoint "
            "on selected CIFAR-10 test examples."
        ),
    },
    "gaussian_surrogate": {
        "label": "paper_formula_with_empirical_moments",
        "evidence_label": "independent_inference",
        "description": (
            "Gaussian ReLU moments evaluated from measured marginal or pair moments; "
            "fit error is diagnostic and is not a pass condition."
        ),
    },
    "shuffle_witness": {
        "label": "deterministic_dependence_counterexample",
        "evidence_label": "counterexample",
        "description": (
            "A deterministic permutation preserves each selected marginal while "
            "changing its pairing with another channel."
        ),
    },
    "eq76_projection": {
        "label": "checkpoint_condition_measurement",
        "evidence_label": "verified_fact",
        "description": (
            "Measures the Eq. 76 column-space condition and separately checks the "
            "same-sample bias-free linear-map identity."
        ),
    },
}

CSV_FIELDS = (
    "section",
    "layer",
    "case_id",
    "channel_i",
    "channel_j",
    "metric",
    "value",
    "reference_value",
    "abs_error",
    "rel_error",
    "sample_count",
    "method_origin",
    "evidence_label",
    "is_quality_gate",
    "passed",
    "notes",
)

SOURCE_RELATIVE_PATHS = (
    "experiments/audits/part03_data_free/README.md",
    "experiments/audits/part03_data_free/run.py",
    "experiments/audits/part03_data_free/test_run.py",
    "src/hope_repro/paper_literal/vgg8.py",
    "paper/manifest.json",
)


class AuditFailure(RuntimeError):
    """Raised when an audit invariant or evidence-quality gate fails."""


def _fail(message: str) -> NoReturn:
    raise AuditFailure(message)


def _normal_pdf(value: float) -> float:
    return math.exp(-0.5 * value * value) / math.sqrt(2.0 * math.pi)


def _normal_cdf(value: float) -> float:
    return 0.5 * math.erfc(-value / math.sqrt(2.0))


def _validate_bvn_order(order: int) -> int:
    if (
        isinstance(order, bool)
        or not isinstance(order, (int, np.integer))
        or order < 32
        or order > 512
    ):
        raise ValueError("bivariate CDF order must be an integer in [32, 512]")
    return int(order)


def _relu_gaussian_first_moment(mean: float, std: float) -> float:
    """Return E[ReLU(Y)] for a univariate Gaussian, including point masses."""
    if std == 0.0:
        return max(0.0, mean)
    ratio = mean / std
    value = std * _normal_pdf(ratio) + mean * _normal_cdf(ratio)
    if not math.isfinite(value) or value < -2e-14:
        raise FloatingPointError("invalid Gaussian first-moment result")
    return max(0.0, value)


def _z_phi(value: float) -> float:
    """Evaluate z phi(z), using its zero limit at either infinity."""
    if math.isinf(value):
        return 0.0
    return value * _normal_pdf(value)


def _degenerate_relu_pair_kernel(
    mean_i: float,
    mean_j: float,
    std_i: float,
    std_j: float,
    correlation: float,
) -> float:
    """Integrate a degenerate bivariate Gaussian without quadrature."""
    if std_i == 0.0:
        return max(0.0, mean_i) * _relu_gaussian_first_moment(mean_j, std_j)
    if std_j == 0.0:
        return max(0.0, mean_j) * _relu_gaussian_first_moment(mean_i, std_i)

    lower = -mean_i / std_i
    if correlation == 1.0:
        lower = max(lower, -mean_j / std_j)
        if lower == math.inf:
            return 0.0
        if lower == -math.inf:
            mass, first_moment, second_moment = 1.0, 0.0, 1.0
        else:
            density = _normal_pdf(lower)
            mass = _normal_cdf(-lower)
            first_moment = density
            second_moment = mass + lower * density
        value = (
            mean_i * mean_j * mass
            + (mean_i * std_j + mean_j * std_i) * first_moment
            + std_i * std_j * second_moment
        )
    else:
        upper = mean_j / std_j
        if lower >= upper:
            return 0.0
        mass = _normal_cdf(upper) - _normal_cdf(lower)
        first_moment = _normal_pdf(lower) - _normal_pdf(upper)
        second_moment = mass + _z_phi(lower) - _z_phi(upper)
        value = (
            mean_i * mean_j * mass
            + (std_i * mean_j - mean_i * std_j) * first_moment
            - std_i * std_j * second_moment
        )
    if not math.isfinite(value) or value < -2e-12:
        raise FloatingPointError("invalid degenerate Gaussian pair-kernel result")
    return max(0.0, value)


def relu_gaussian_self_kernel(mean: float, std: float) -> float:
    """Return E[ReLU(Y)^2] for Y distributed as N(mean, std^2)."""
    mean = float(mean)
    std = float(std)
    if not math.isfinite(mean) or not math.isfinite(std) or std < 0.0:
        raise ValueError("mean and std must be finite and std must be nonnegative")
    if std == 0.0:
        return max(0.0, mean) ** 2
    ratio = mean / std
    value = (std * std + mean * mean) * _normal_cdf(ratio)
    value += mean * std * _normal_pdf(ratio)
    if not math.isfinite(value) or value < -1e-14:
        raise FloatingPointError("invalid Gaussian self-kernel result")
    return max(0.0, value)


def bivariate_normal_cdf(
    first: float,
    second: float,
    correlation: float,
    *,
    order: int = DEFAULT_BVN_ORDER,
) -> float:
    """Evaluate Phi_2 using Plackett's one-dimensional correlation integral."""
    first = float(first)
    second = float(second)
    correlation = float(correlation)
    if not all(math.isfinite(v) for v in (first, second, correlation)):
        raise ValueError("bivariate CDF inputs must be finite")
    if correlation < -1.0 or correlation > 1.0:
        raise ValueError("correlation must lie in [-1, 1]")
    order = _validate_bvn_order(order)
    if correlation == 0.0:
        return _normal_cdf(first) * _normal_cdf(second)
    if correlation == 1.0:
        return _normal_cdf(min(first, second))
    if correlation == -1.0:
        return max(0.0, _normal_cdf(first) - _normal_cdf(-second))

    nodes, weights = np.polynomial.legendre.leggauss(order)
    values = 0.5 * correlation * (nodes + 1.0)
    one_minus = 1.0 - values * values
    exponent = -(
        first * first - 2.0 * values * first * second + second * second
    ) / (2.0 * one_minus)
    integrand = np.exp(exponent) / (2.0 * math.pi * np.sqrt(one_minus))
    integral = 0.5 * correlation * float(np.dot(weights, integrand))
    result = _normal_cdf(first) * _normal_cdf(second) + integral
    if not math.isfinite(result) or result < -2e-13 or result > 1.0 + 2e-13:
        raise FloatingPointError("invalid bivariate normal CDF result")
    return min(1.0, max(0.0, result))


def relu_gaussian_pair_kernel(
    mean_i: float,
    mean_j: float,
    std_i: float,
    std_j: float,
    correlation: float,
    *,
    order: int = DEFAULT_BVN_ORDER,
) -> float:
    """Return the exact biased Gaussian ReLU pair moment from HOPE Eq. 83."""
    values = (mean_i, mean_j, std_i, std_j, correlation)
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("pair-kernel inputs must be finite")
    if std_i < 0.0 or std_j < 0.0:
        raise ValueError("standard deviations must be nonnegative")
    if correlation < -1.0 or correlation > 1.0:
        raise ValueError("correlation must lie in [-1, 1]")
    order = _validate_bvn_order(order)
    if std_i == 0.0 or std_j == 0.0 or abs(correlation) == 1.0:
        value = _degenerate_relu_pair_kernel(
            float(mean_i),
            float(mean_j),
            float(std_i),
            float(std_j),
            float(correlation),
        )
        if not math.isfinite(value):
            raise FloatingPointError("invalid Gaussian pair-kernel result")
        return value

    ci = mean_i / std_i
    cj = mean_j / std_j
    root = math.sqrt(1.0 - correlation * correlation)
    cij = (ci - correlation * cj) / root
    cji = (cj - correlation * ci) / root
    phi2 = math.exp(
        -(ci * ci - 2.0 * correlation * ci * cj + cj * cj)
        / (2.0 * (1.0 - correlation * correlation))
    ) / (2.0 * math.pi * root)
    standardized = (
        (ci * cj + correlation)
        * bivariate_normal_cdf(ci, cj, correlation, order=order)
        + ci * _normal_pdf(cj) * _normal_cdf(cij)
        + cj * _normal_pdf(ci) * _normal_cdf(cji)
        + (1.0 - correlation * correlation) * phi2
    )
    value = std_i * std_j * standardized
    if not math.isfinite(value) or value < -2e-12:
        raise FloatingPointError("invalid Gaussian pair-kernel result")
    return max(0.0, value)


def zero_bias_pair_kernel(std_i: float, std_j: float, correlation: float) -> float:
    """Return the zero-bias arc-cosine ReLU pair moment."""
    if std_i < 0.0 or std_j < 0.0 or not -1.0 <= correlation <= 1.0:
        raise ValueError("invalid zero-bias pair-kernel arguments")
    factor = (
        math.sqrt(max(0.0, 1.0 - correlation * correlation))
        + (math.pi - math.acos(correlation)) * correlation
    ) / (2.0 * math.pi)
    return std_i * std_j * factor


def distribution_diagnostics(values: np.ndarray) -> dict[str, float | int]:
    """Compute fitted-normal diagnostics without interpreting them as pass gates."""
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size < 8 or not np.all(np.isfinite(array)):
        raise ValueError("diagnostics require at least eight finite observations")
    mean = float(np.mean(array))
    centered = array - mean
    variance = float(np.mean(centered * centered))
    std = math.sqrt(max(0.0, variance))
    if std <= TOLERANCES["variance_floor"]:
        skewness = 0.0
        excess_kurtosis = -3.0
        ks_distance = 0.5
    else:
        standardized = centered / std
        skewness = float(np.mean(standardized**3))
        excess_kurtosis = float(np.mean(standardized**4) - 3.0)
        ordered = np.sort(standardized)
        cdf = np.fromiter(
            (_normal_cdf(float(value)) for value in ordered),
            dtype=np.float64,
            count=ordered.size,
        )
        upper = np.arange(1, ordered.size + 1, dtype=np.float64) / ordered.size
        lower = np.arange(0, ordered.size, dtype=np.float64) / ordered.size
        ks_distance = float(max(np.max(upper - cdf), np.max(cdf - lower)))
    relu = np.maximum(array, 0.0)
    return {
        "sample_count": int(array.size),
        "mean": mean,
        "variance": variance,
        "std": std,
        "skewness": skewness,
        "excess_kurtosis": excess_kurtosis,
        "ks_distance_fitted_normal": ks_distance,
        "nonpositive_fraction": float(np.mean(array <= 0.0)),
        "relu_second_moment": float(np.mean(relu * relu)),
    }


def standardized_histogram(
    values: np.ndarray,
    *,
    bins: int = 40,
    z_limit: float = 5.0,
) -> dict[str, Any]:
    """Return fixed-bin fitted-standard-normal densities for visual rendering."""
    diagnostics = distribution_diagnostics(values)
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    std = float(diagnostics["std"])
    if std <= TOLERANCES["variance_floor"]:
        standardized = np.zeros_like(array)
    else:
        standardized = (array - float(diagnostics["mean"])) / std
    edges = np.linspace(-z_limit, z_limit, bins + 1, dtype=np.float64)
    counts, _ = np.histogram(standardized, bins=edges)
    width = float(edges[1] - edges[0])
    rows = []
    for index, count in enumerate(counts.tolist()):
        lower = float(edges[index])
        upper = float(edges[index + 1])
        rows.append(
            {
                "bin_index": index,
                "bin_lower": lower,
                "bin_upper": upper,
                "bin_center": 0.5 * (lower + upper),
                "empirical_density": float(count) / (array.size * width),
                "standard_normal_density": (
                    _normal_cdf(upper) - _normal_cdf(lower)
                )
                / width,
            }
        )
    return {
        "sample_count": int(array.size),
        "standardization_mean": float(diagnostics["mean"]),
        "standardization_std": std,
        "bin_count": bins,
        "z_limit": z_limit,
        "lower_tail_fraction": float(np.mean(standardized < -z_limit)),
        "upper_tail_fraction": float(np.mean(standardized > z_limit)),
        "rows": rows,
    }


def scaled_error(
    value: float,
    reference: float,
    *,
    atol: float,
    rtol: float,
) -> float:
    return abs(value - reference) / (atol + rtol * abs(reference))


def symmetric_relative_error(value: float, reference: float) -> float:
    """Return a bounded [0, 2] error that remains defined near zero."""
    return 2.0 * abs(value - reference) / max(
        abs(value) + abs(reference), TOLERANCES["variance_floor"]
    )


def projection_metrics(weight: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    """Measure the condition W W+ target = target without width inference."""
    matrix = np.asarray(weight, dtype=np.float64)
    vector = np.asarray(target, dtype=np.float64).reshape(-1)
    if matrix.ndim != 2 or matrix.shape[0] != vector.size:
        raise ValueError("projection matrix and target shapes do not align")
    solution, _, rank, singular_values = np.linalg.lstsq(matrix, vector, rcond=None)
    projected = matrix @ solution
    residual = projected - vector
    norm = float(np.linalg.norm(vector))
    residual_norm = float(np.linalg.norm(residual))
    return {
        "rows": int(matrix.shape[0]),
        "columns": int(matrix.shape[1]),
        "rank": int(rank),
        "target_norm": norm,
        "residual_norm": residual_norm,
        "relative_residual": residual_norm / max(norm, np.finfo(np.float64).tiny),
        "max_abs_residual": float(np.max(np.abs(residual))),
        "projected": projected,
        "minimum_singular_value": (
            float(singular_values[-1]) if singular_values.size else 0.0
        ),
        "maximum_singular_value": (
            float(singular_values[0]) if singular_values.size else 0.0
        ),
    }


def select_evenly(total: int, limit: int) -> list[int]:
    if total <= 0 or limit <= 0:
        raise ValueError("selection sizes must be positive")
    count = min(total, limit)
    if count == total:
        return list(range(total))
    return np.linspace(0, total - 1, count, dtype=np.int64).tolist()


def select_pairs(channels: Sequence[int], limit: int) -> list[tuple[int, int]]:
    unique = list(dict.fromkeys(int(value) for value in channels))
    if len(unique) < 2 or limit <= 0:
        raise ValueError("pair selection needs two channels and a positive limit")
    candidates: list[tuple[int, int]] = []
    for offset in range(1, len(unique)):
        for index in range(len(unique) - offset):
            candidates.append((unique[index], unique[index + offset]))
    return candidates[:limit]


def select_stratified_indices(
    targets: Sequence[int],
    count: int,
    seed: int,
) -> list[int]:
    """Select a deterministic near-equal class-stratified subset."""
    labels = np.asarray(targets, dtype=np.int64)
    if labels.ndim != 1 or labels.size == 0:
        raise ValueError("targets must be a nonempty one-dimensional sequence")
    if count <= 0 or count > labels.size:
        raise ValueError("stratified sample count is outside the dataset")
    classes = np.unique(labels)
    base, remainder = divmod(count, classes.size)
    rng = np.random.default_rng(seed)
    selected: list[int] = []
    for class_position, label in enumerate(classes.tolist()):
        candidates = np.flatnonzero(labels == label)
        quota = base + int(class_position < remainder)
        if quota > candidates.size:
            raise ValueError(f"class {label} has fewer examples than its quota")
        chosen = rng.choice(candidates, size=quota, replace=False)
        selected.extend(int(value) for value in chosen)
    return sorted(selected)


def fixed_formula_audit(order: int) -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    fixtures = (
        ("independent_biased", 0.7, -0.2, 1.1, 0.8, 0.0),
        ("zero_bias_negative", 0.0, 0.0, 1.4, 0.6, -0.65),
        ("zero_bias_positive", 0.0, 0.0, 0.9, 1.8, 0.75),
    )
    for case_id, mi, mj, si, sj, rho in fixtures:
        observed = relu_gaussian_pair_kernel(mi, mj, si, sj, rho, order=order)
        if rho == 0.0:
            first = si * _normal_pdf(mi / si) + mi * _normal_cdf(mi / si)
            second = sj * _normal_pdf(mj / sj) + mj * _normal_cdf(mj / sj)
            reference = first * second
        else:
            reference = zero_bias_pair_kernel(si, sj, rho)
        error = scaled_error(
            observed,
            reference,
            atol=TOLERANCES["formula_atol"],
            rtol=TOLERANCES["formula_rtol"],
        )
        cases.append(
            {
                "case_id": case_id,
                "value": observed,
                "reference": reference,
                "scaled_error": error,
                "passed": error <= 1.0,
            }
        )
    return {
        "cases": cases,
        "max_scaled_error": max(case["scaled_error"] for case in cases),
        "all_cases_passed": all(case["passed"] for case in cases),
    }


class _LayerCapture:
    """Bounded activation and common-input-moment capture for one layer."""

    def __init__(
        self,
        name: str,
        op: Any,
        bn: Any,
        channels_per_layer: int,
        spatial_positions: int,
    ) -> None:
        self.name = name
        self.op = op
        self.bn = bn
        self.channel_indices = select_evenly(int(op.weight.shape[0]), channels_per_layer)
        self.spatial_positions = spatial_positions
        self.pre_chunks: list[np.ndarray] = []
        self.post_chunks: list[np.ndarray] = []
        self.pre_sum = np.zeros(int(op.weight.shape[0]), dtype=np.float64)
        self.input_sum = np.zeros(int(op.weight[0].numel()), dtype=np.float64)
        self.observation_count = 0
        self._last_positions: list[int] | None = None
        self._last_spatial_size: int | None = None

    @staticmethod
    def _activation_rows(tensor: Any, positions: Sequence[int]) -> Any:
        if tensor.ndim == 2:
            return tensor
        batch, channels, height, width = tensor.shape
        flattened = tensor.permute(0, 2, 3, 1).reshape(batch, height * width, channels)
        return flattened[:, list(positions), :].reshape(-1, channels)

    def pre_hook(self, _module: Any, inputs: tuple[Any, ...], output: Any) -> None:
        import torch.nn.functional as torch_functional

        if output.ndim == 2:
            positions = [0]
            spatial_size = 1
        else:
            spatial_size = int(output.shape[2] * output.shape[3])
            positions = select_evenly(spatial_size, self.spatial_positions)
        rows = self._activation_rows(output.detach(), positions)
        selected = rows[:, self.channel_indices].to(device="cpu", dtype=__import__("torch").float64)
        self.pre_chunks.append(selected.numpy().copy())
        self.pre_sum += rows.sum(dim=0, dtype=__import__("torch").float64).cpu().numpy()
        self.observation_count += int(rows.shape[0])

        layer_input = inputs[0].detach()
        if output.ndim == 2:
            input_rows = layer_input.reshape(layer_input.shape[0], -1)
        else:
            patches = torch_functional.unfold(
                layer_input,
                kernel_size=self.op.kernel_size,
                dilation=self.op.dilation,
                padding=self.op.padding,
                stride=self.op.stride,
            ).transpose(1, 2)
            if int(patches.shape[1]) != spatial_size:
                raise AuditFailure(f"unfold/output position mismatch for {self.name}")
            input_rows = patches[:, positions, :].reshape(-1, patches.shape[2])
        self.input_sum += (
            input_rows.sum(dim=0, dtype=__import__("torch").float64).cpu().numpy()
        )
        self._last_positions = positions
        self._last_spatial_size = spatial_size

    def post_hook(self, _module: Any, _inputs: tuple[Any, ...], output: Any) -> None:
        if self._last_positions is None:
            raise AuditFailure(f"post-BN hook ran before pre-BN hook for {self.name}")
        rows = self._activation_rows(output.detach(), self._last_positions)
        selected = rows[:, self.channel_indices].to(device="cpu", dtype=__import__("torch").float64)
        self.post_chunks.append(selected.numpy().copy())

    def finalize(self) -> dict[str, Any]:
        if not self.pre_chunks or not self.post_chunks or self.observation_count <= 0:
            raise AuditFailure(f"no observations captured for {self.name}")
        pre = np.concatenate(self.pre_chunks, axis=0)
        post = np.concatenate(self.post_chunks, axis=0)
        if pre.shape != post.shape or pre.shape[0] != self.observation_count:
            raise AuditFailure(f"pre/post observation mismatch for {self.name}")
        return {
            "name": self.name,
            "pre": pre,
            "post": post,
            "channel_indices": list(self.channel_indices),
            "pre_mean_all": self.pre_sum / self.observation_count,
            "input_mean": self.input_sum / self.observation_count,
            "observation_count": self.observation_count,
            "weight": self.op.weight.detach().cpu().numpy().reshape(
                int(self.op.weight.shape[0]), -1
            ),
            "running_mean": self.bn.running_mean.detach().cpu().numpy(),
            "running_var": self.bn.running_var.detach().cpu().numpy(),
            "gamma": self.bn.weight.detach().cpu().numpy(),
            "beta": self.bn.bias.detach().cpu().numpy(),
            "epsilon": float(self.bn.eps),
        }


def _load_runtime() -> tuple[Any, Any, Any]:
    try:
        import torch
        import torchvision
        import torchvision.transforms as transforms
    except ImportError as error:
        raise AuditFailure(
            "Torch and torchvision are required for the checkpoint audit; "
            "install the project paper extra"
        ) from error
    src = str(ROOT / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    return torch, torchvision, transforms


def _vgg8_wiring(model: Any) -> list[tuple[str, Any, Any]]:
    features = model.features
    return [
        ("conv0", features[0], features[1]),
        ("conv1", features[4], features[5]),
        ("conv2", features[8], features[9]),
        ("conv3", features[11], features[12]),
        ("conv4", features[15], features[16]),
        ("conv5", features[18], features[19]),
        ("dense", model.bottleneck, model.bn_b),
    ]


def _hash_selected_examples(dataset: Any, indices: Sequence[int]) -> str:
    digest = hashlib.sha256()
    raw_data = getattr(dataset, "data", None)
    targets = getattr(dataset, "targets", None)
    if raw_data is None or targets is None:
        raise AuditFailure("CIFAR-10 dataset does not expose raw data and targets")
    selected = np.asarray(raw_data)[list(indices)]
    labels = np.asarray(targets, dtype=np.int64)[list(indices)]
    digest.update(selected.tobytes(order="C"))
    digest.update(labels.astype("<i8", copy=False).tobytes(order="C"))
    return digest.hexdigest()


def _load_checkpoint(checkpoint_path: Path, torch: Any) -> tuple[Any, Mapping[str, Any]]:
    if not checkpoint_path.is_file():
        raise AuditFailure(f"checkpoint is missing: {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError as error:
        raise AuditFailure(
            "checkpoint loading requires torch.load(..., weights_only=True); "
            "this Torch runtime does not support that safe mode"
        ) from error
    if not isinstance(checkpoint, Mapping) or not isinstance(
        checkpoint.get("model"), Mapping
    ):
        raise AuditFailure("checkpoint must contain a model state dictionary")
    from hope_repro.paper_literal.vgg8 import VGG8

    model = VGG8()
    model.load_state_dict(checkpoint["model"], strict=True)
    for name, tensor in model.state_dict().items():
        if not bool(torch.isfinite(tensor).all()):
            raise AuditFailure(f"checkpoint contains non-finite tensor values: {name}")
    model.eval()
    return model, checkpoint


def capture_checkpoint_data(
    *,
    checkpoint_path: Path,
    data_dir: Path,
    seed: int,
    max_examples: int,
    batch_size: int,
    spatial_positions: int,
    channels_per_layer: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load the adapted route and return bounded deterministic layer samples."""
    torch, torchvision, transforms = _load_runtime()
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    model, checkpoint = _load_checkpoint(checkpoint_path, torch)
    try:
        dataset = torchvision.datasets.CIFAR10(
            root=data_dir,
            train=False,
            download=False,
            transform=transforms.ToTensor(),
        )
    except Exception as error:
        raise AuditFailure(
            f"could not load local CIFAR-10 test data from {data_dir}: {error}"
        ) from error
    if max_examples <= 0 or max_examples > len(dataset):
        raise ValueError("max_examples must be positive and no larger than the dataset")
    selected_indices = np.asarray(
        select_stratified_indices(dataset.targets, max_examples, seed),
        dtype=np.int64,
    )
    subset = torch.utils.data.Subset(dataset, selected_indices.tolist())
    loader = torch.utils.data.DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )

    captures = [
        _LayerCapture(name, op, bn, channels_per_layer, spatial_positions)
        for name, op, bn in _vgg8_wiring(model)
        if name in LAYER_NAMES
    ]
    if [capture.name for capture in captures] != list(LAYER_NAMES):
        raise AuditFailure("the VGG-8 audit-layer registry differs from the declared order")
    handles = []
    for capture in captures:
        handles.append(capture.op.register_forward_hook(capture.pre_hook))
        handles.append(capture.bn.register_forward_hook(capture.post_hook))
    observed_examples = 0
    try:
        with torch.no_grad():
            for inputs, _labels in loader:
                model(inputs)
                observed_examples += int(inputs.shape[0])
    finally:
        for handle in handles:
            handle.remove()
    if observed_examples != max_examples:
        raise AuditFailure("the data loader did not yield the configured example count")

    checkpoint_config = checkpoint.get("config")
    state_schema = [
        {
            "key": str(key),
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
        }
        for key, tensor in sorted(checkpoint["model"].items())
    ]
    raw_dataset_files = []
    dataset_root = Path(dataset.root) / dataset.base_folder
    for logical_role, filename in (
        ("cifar10_test_batch", "test_batch"),
        ("cifar10_batches_meta", "batches.meta"),
    ):
        path = dataset_root / filename
        if not path.is_file():
            raise AuditFailure(f"required CIFAR-10 raw file is missing: {filename}")
        raw_dataset_files.append(
            {
                "logical_role": logical_role,
                "filename": filename,
                "sha256": _sha256_path(path),
                "bytes": path.stat().st_size,
            }
        )
    provenance = {
        "checkpoint": {
            "filename": checkpoint_path.name,
            "sha256": _sha256_path(checkpoint_path),
            "bytes": checkpoint_path.stat().st_size,
            "state_schema_sha256": _sha256_json(state_schema),
            "state_schema_tensor_count": len(state_schema),
            "strict_state_dict_load": True,
            "stored_test_accuracy": checkpoint.get("test_acc"),
            "stored_reproduction_scope": checkpoint.get("reproduction_scope"),
            "stored_config_sha256": (
                _sha256_json(checkpoint_config)
                if isinstance(checkpoint_config, Mapping)
                else None
            ),
            "historical_training_provenance": {
                "status": "incomplete",
                "available": [
                    "checkpoint bytes and SHA-256",
                    "ordered state schema digest",
                    "stored test accuracy",
                    "stored reproduction scope",
                    "stored configuration digest when present",
                ],
                "missing_from_checkpoint": [
                    "epoch-by-epoch training history",
                    "optimizer and scheduler state",
                    "training dataset file hashes",
                    "original generating command and environment record",
                ],
            },
        },
        "dataset": {
            "name": "CIFAR-10",
            "split": "test",
            "dataset_length": len(dataset),
            "selection_rule": "sorted seeded near-equal class-stratified sample without replacement",
            "selection_seed": seed,
            "selected_example_count": max_examples,
            "selected_indices_sha256": _sha256_bytes(
                selected_indices.astype("<i8", copy=False).tobytes(order="C")
            ),
            "selected_examples_sha256": _hash_selected_examples(
                dataset, selected_indices.tolist()
            ),
            "raw_files": raw_dataset_files,
            "preprocessing": {
                "transform": "ToTensor",
                "normalization": "none",
                "augmentation": "none",
            },
        },
        "runtime": {
            "torch_version": torch.__version__,
            "torchvision_version": torchvision.__version__,
            "device": "cpu",
            "deterministic_algorithms": True,
            "torch_threads": 1,
            "data_loader_workers": 0,
        },
    }
    return [capture.finalize() for capture in captures], provenance


def _diagnostic_record(
    *,
    layer: str,
    channel: int,
    domain: str,
    values: np.ndarray,
) -> dict[str, Any]:
    record = distribution_diagnostics(values)
    return {
        "layer": layer,
        "channel": channel,
        "domain": domain,
        **record,
    }


def analyze_layer(
    captured: Mapping[str, Any],
    *,
    pairs_per_layer: int,
    bvn_order: int,
    seed: int,
) -> dict[str, Any]:
    name = str(captured["name"])
    channels = [int(value) for value in captured["channel_indices"]]
    pre = np.asarray(captured["pre"], dtype=np.float64)
    post = np.asarray(captured["post"], dtype=np.float64)
    running_mean = np.asarray(captured["running_mean"], dtype=np.float64)
    running_var = np.asarray(captured["running_var"], dtype=np.float64)
    gamma = np.asarray(captured["gamma"], dtype=np.float64)
    beta = np.asarray(captured["beta"], dtype=np.float64)
    epsilon = float(captured["epsilon"])
    if np.any(running_var < 0.0):
        raise AuditFailure(f"negative BN running variance in {name}")

    normality: list[dict[str, Any]] = []
    moment_comparisons: list[dict[str, Any]] = []
    self_kernels: list[dict[str, Any]] = []
    for local_index, channel in enumerate(channels):
        pre_diag = _diagnostic_record(
            layer=name, channel=channel, domain="pre_bn", values=pre[:, local_index]
        )
        post_diag = _diagnostic_record(
            layer=name, channel=channel, domain="post_bn", values=post[:, local_index]
        )
        normality.extend((pre_diag, post_diag))

        finite_expected_variance = float(gamma[channel] ** 2) * float(
            running_var[channel]
        ) / (float(running_var[channel]) + epsilon)
        moment_comparisons.append(
            {
                "layer": name,
                "channel": channel,
                "sample_count": int(post.shape[0]),
                "pre_bn_empirical_mean": float(pre_diag["mean"]),
                "pre_bn_running_mean": float(running_mean[channel]),
                "pre_bn_mean_absolute_difference": abs(
                    float(pre_diag["mean"]) - float(running_mean[channel])
                ),
                "pre_bn_mean_symmetric_relative_error": symmetric_relative_error(
                    float(pre_diag["mean"]), float(running_mean[channel])
                ),
                "pre_bn_empirical_variance": float(pre_diag["variance"]),
                "pre_bn_running_variance": float(running_var[channel]),
                "pre_bn_variance_absolute_difference": abs(
                    float(pre_diag["variance"]) - float(running_var[channel])
                ),
                "pre_bn_variance_symmetric_relative_error": symmetric_relative_error(
                    float(pre_diag["variance"]), float(running_var[channel])
                ),
                "post_bn_empirical_mean": float(post_diag["mean"]),
                "post_bn_expected_mean_beta": float(beta[channel]),
                "post_bn_mean_absolute_difference": abs(
                    float(post_diag["mean"]) - float(beta[channel])
                ),
                "post_bn_mean_symmetric_relative_error": symmetric_relative_error(
                    float(post_diag["mean"]), float(beta[channel])
                ),
                "post_bn_empirical_variance": float(post_diag["variance"]),
                "post_bn_finite_epsilon_expected_variance": finite_expected_variance,
                "post_bn_variance_absolute_difference": abs(
                    float(post_diag["variance"]) - finite_expected_variance
                ),
                "post_bn_variance_symmetric_relative_error": symmetric_relative_error(
                    float(post_diag["variance"]), finite_expected_variance
                ),
                "moment_discrepancies_are_quality_gates": False,
            }
        )

        empirical = float(post_diag["relu_second_moment"])
        fitted = relu_gaussian_self_kernel(
            float(post_diag["mean"]), float(post_diag["std"])
        )
        finite_std = abs(float(gamma[channel])) * math.sqrt(
            float(running_var[channel]) / (float(running_var[channel]) + epsilon)
        )
        finite_state = relu_gaussian_self_kernel(float(beta[channel]), finite_std)
        negligible_epsilon = relu_gaussian_self_kernel(
            float(beta[channel]), abs(float(gamma[channel]))
        )
        shape_error = empirical - fitted
        moment_state_error = fitted - finite_state
        epsilon_error = finite_state - negligible_epsilon
        total_error = empirical - negligible_epsilon
        residual = total_error - (shape_error + moment_state_error + epsilon_error)
        tolerance = TOLERANCES["decomposition_atol"] + TOLERANCES[
            "decomposition_rtol"
        ] * abs(total_error)
        self_kernels.append(
            {
                "layer": name,
                "channel": channel,
                "sample_count": int(post.shape[0]),
                "empirical_relu_second_moment": empirical,
                "gaussian_empirical_moments": fitted,
                "gaussian_fit_absolute_error": abs(empirical - fitted),
                "gaussian_fit_symmetric_relative_error": symmetric_relative_error(
                    empirical, fitted
                ),
                "gaussian_bn_finite_epsilon": finite_state,
                "gaussian_bn_negligible_epsilon": negligible_epsilon,
                "shape_error": shape_error,
                "moment_state_error": moment_state_error,
                "finite_epsilon_error": epsilon_error,
                "total_error": total_error,
                "decomposition_residual": residual,
                "decomposition_tolerance": tolerance,
                "decomposition_passed": abs(residual) <= tolerance,
                "hypothesis_fit_is_quality_gate": False,
            }
        )

    pair_records: list[dict[str, Any]] = []
    pair_candidates = select_pairs(channels, pairs_per_layer)
    index_by_channel = {channel: index for index, channel in enumerate(channels)}
    for channel_i, channel_j in pair_candidates:
        first = post[:, index_by_channel[channel_i]]
        second = post[:, index_by_channel[channel_j]]
        first_diag = distribution_diagnostics(first)
        second_diag = distribution_diagnostics(second)
        centered_i = first - float(first_diag["mean"])
        centered_j = second - float(second_diag["mean"])
        covariance = float(np.mean(centered_i * centered_j))
        denominator = float(first_diag["std"]) * float(second_diag["std"])
        correlation = 0.0 if denominator <= TOLERANCES["variance_floor"] else covariance / denominator
        correlation = float(np.clip(correlation, -1.0, 1.0))
        empirical = float(np.mean(np.maximum(first, 0.0) * np.maximum(second, 0.0)))
        gaussian = relu_gaussian_pair_kernel(
            float(first_diag["mean"]),
            float(second_diag["mean"]),
            float(first_diag["std"]),
            float(second_diag["std"]),
            correlation,
            order=bvn_order,
        )
        pair_records.append(
            {
                "layer": name,
                "channel_i": channel_i,
                "channel_j": channel_j,
                "sample_count": int(post.shape[0]),
                "empirical_mean_i": float(first_diag["mean"]),
                "empirical_mean_j": float(second_diag["mean"]),
                "empirical_std_i": float(first_diag["std"]),
                "empirical_std_j": float(second_diag["std"]),
                "empirical_covariance": covariance,
                "empirical_correlation": correlation,
                "empirical_relu_pair_kernel": empirical,
                "gaussian_exact_biased_pair_kernel": gaussian,
                "gaussian_fit_error": empirical - gaussian,
                "gaussian_fit_absolute_error": abs(empirical - gaussian),
                "gaussian_fit_symmetric_relative_error": symmetric_relative_error(
                    empirical, gaussian
                ),
                "gaussian_fit_relative_error": (empirical - gaussian)
                / max(abs(gaussian), np.finfo(np.float64).tiny),
                "hypothesis_fit_is_quality_gate": False,
            }
        )

    rng = np.random.default_rng(seed + 1009 * (LAYER_NAMES.index(name) + 1))
    permutation = rng.permutation(post.shape[0])
    if np.array_equal(permutation, np.arange(post.shape[0])):
        permutation = np.roll(permutation, 1)
    witnesses: list[dict[str, Any]] = []
    for record in pair_records:
        channel_i = int(record["channel_i"])
        channel_j = int(record["channel_j"])
        first = post[:, index_by_channel[channel_i]]
        second = post[:, index_by_channel[channel_j]]
        shuffled = second[permutation]
        original_marginal = distribution_diagnostics(second)
        shuffled_marginal = distribution_diagnostics(shuffled)
        marginal_keys = (
            "mean",
            "variance",
            "skewness",
            "excess_kurtosis",
            "relu_second_moment",
        )
        marginal_difference = max(
            abs(float(original_marginal[key]) - float(shuffled_marginal[key]))
            for key in marginal_keys
        )
        original_covariance = float(
            np.mean((first - np.mean(first)) * (second - np.mean(second)))
        )
        shuffled_covariance = float(
            np.mean((first - np.mean(first)) * (shuffled - np.mean(shuffled)))
        )
        original_kernel = float(
            np.mean(np.maximum(first, 0.0) * np.maximum(second, 0.0))
        )
        shuffled_kernel = float(
            np.mean(np.maximum(first, 0.0) * np.maximum(shuffled, 0.0))
        )
        scale = max(
            abs(original_covariance),
            abs(original_kernel),
            float(np.std(first) * np.std(second)),
            np.finfo(np.float64).tiny,
        )
        dependence_change = max(
            abs(shuffled_covariance - original_covariance),
            abs(shuffled_kernel - original_kernel),
        )
        witnesses.append(
            {
                "layer": name,
                "channel_i": channel_i,
                "channel_j": channel_j,
                "sample_count": int(post.shape[0]),
                "permutation_sha256": _sha256_bytes(
                    permutation.astype("<i8", copy=False).tobytes(order="C")
                ),
                "max_marginal_abs_difference": marginal_difference,
                "original_covariance": original_covariance,
                "shuffled_covariance": shuffled_covariance,
                "original_relu_pair_kernel": original_kernel,
                "shuffled_relu_pair_kernel": shuffled_kernel,
                "dependence_abs_change": dependence_change,
                "dependence_scaled_change": dependence_change / scale,
            }
        )
    shuffle_witness = max(witnesses, key=lambda item: item["dependence_scaled_change"])

    weight = np.asarray(captured["weight"], dtype=np.float64)
    pre_mean_all = np.asarray(captured["pre_mean_all"], dtype=np.float64)
    input_mean = np.asarray(captured["input_mean"], dtype=np.float64)
    common_map_reconstruction = weight @ input_mean
    common_map_residual = common_map_reconstruction - pre_mean_all
    common_scale = float(np.linalg.norm(pre_mean_all))
    common_tolerance = TOLERANCES["common_map_atol"] + TOLERANCES[
        "common_map_rtol"
    ] * common_scale
    stored_projection = projection_metrics(weight, running_mean)
    empirical_projection = projection_metrics(weight, pre_mean_all)
    projected_running = np.asarray(stored_projection.pop("projected"), dtype=np.float64)
    empirical_projection.pop("projected")
    bn_scale = gamma / np.sqrt(running_var + epsilon)
    eq76_reconstructed_mean = bn_scale * projected_running + beta - bn_scale * running_mean
    eq76_delta = eq76_reconstructed_mean - beta
    eq76 = {
        "layer": name,
        "rows": int(weight.shape[0]),
        "columns": int(weight.shape[1]),
        "width_regime": (
            "expansion" if weight.shape[0] > weight.shape[1] else "non_expansion"
        ),
        "causal_width_claim_made": False,
        "stored_running_mean_projection": stored_projection,
        "empirical_pre_mean_projection": empirical_projection,
        "empirical_common_map_residual_norm": float(np.linalg.norm(common_map_residual)),
        "empirical_common_map_relative_residual": float(np.linalg.norm(common_map_residual))
        / max(common_scale, np.finfo(np.float64).tiny),
        "empirical_common_map_tolerance": common_tolerance,
        "empirical_common_map_passed": float(np.linalg.norm(common_map_residual))
        <= common_tolerance,
        "eq76_reconstructed_post_mean_max_abs_delta_from_beta": float(
            np.max(np.abs(eq76_delta))
        ),
        "condition_interpretation": (
            "The stored-statistic residual is a checkpoint condition measurement. "
            "Expansion versus contraction alone is not interpreted as its cause."
        ),
    }
    representative_local_index = len(channels) // 2
    representative_channel = channels[representative_local_index]
    histogram = standardized_histogram(post[:, representative_local_index])
    return {
        "layer": name,
        "module_path": LAYER_MODULE_PATHS[name],
        "observation_count": int(captured["observation_count"]),
        "selected_channels": channels,
        "representative_channel": representative_channel,
        "representative_channel_selection": (
            "middle entry of the deterministic evenly spaced channel selection"
        ),
        "standardized_post_bn_histogram": histogram,
        "normality": normality,
        "moment_comparisons": moment_comparisons,
        "self_kernels": self_kernels,
        "pair_kernels": pair_records,
        "shuffle_witness": shuffle_witness,
        "eq76_projection": eq76,
    }


def _expected_check_ranges() -> dict[str, dict[str, Any]]:
    return {
        "layer_count": {"minimum": 4.0, "maximum": 4.0, "units": "layers"},
        "all_values_finite": {"minimum": 1.0, "maximum": 1.0, "units": "boolean"},
        "observation_bound_violations": {"minimum": 0.0, "maximum": 0.0, "units": "count"},
        "self_decomposition_max_scaled_residual": {
            "minimum": 0.0,
            "maximum": 1.0,
            "units": "scaled_error",
        },
        "formula_fixture_max_scaled_error": {
            "minimum": 0.0,
            "maximum": 1.0,
            "units": "scaled_error",
        },
        "common_map_failure_count": {"minimum": 0.0, "maximum": 0.0, "units": "count"},
        "shuffle_marginal_max_scaled_error": {
            "minimum": 0.0,
            "maximum": 1.0,
            "units": "scaled_error",
        },
        "shuffle_witness_failure_count": {
            "minimum": 0.0,
            "maximum": 0.0,
            "units": "count",
        },
        "gaussian_hypothesis_pass_gate_count": {
            "minimum": 0.0,
            "maximum": 0.0,
            "units": "count",
        },
    }


def _walk_finite(value: Any) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, np.integer)):
        return True
    if isinstance(value, (float, np.floating)):
        return math.isfinite(float(value))
    if isinstance(value, Mapping):
        return all(_walk_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_walk_finite(item) for item in value)
    return False


def build_checks(
    layers: Sequence[Mapping[str, Any]],
    formula_audit: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    maximum_observations = int(config["max_examples"]) * int(
        config["spatial_positions_per_example"]
    )
    observation_violations = sum(
        int(layer["observation_count"] > maximum_observations) for layer in layers
    )
    decomposition_scaled = []
    hypothesis_gates = 0
    for layer in layers:
        for record in layer["moment_comparisons"]:
            hypothesis_gates += int(
                bool(record["moment_discrepancies_are_quality_gates"])
            )
        for record in layer["self_kernels"]:
            decomposition_scaled.append(
                abs(float(record["decomposition_residual"]))
                / float(record["decomposition_tolerance"])
            )
            hypothesis_gates += int(bool(record["hypothesis_fit_is_quality_gate"]))
        for record in layer["pair_kernels"]:
            hypothesis_gates += int(bool(record["hypothesis_fit_is_quality_gate"]))
    shuffle_scaled = []
    shuffle_failures = 0
    for layer in layers:
        witness = layer["shuffle_witness"]
        reference_scale = max(
            abs(float(witness["original_relu_pair_kernel"])),
            abs(float(witness["original_covariance"])),
            1.0,
        )
        tolerance = TOLERANCES["shuffle_marginal_atol"] + TOLERANCES[
            "shuffle_marginal_rtol"
        ] * reference_scale
        shuffle_scaled.append(float(witness["max_marginal_abs_difference"]) / tolerance)
        shuffle_failures += int(
            float(witness["dependence_scaled_change"])
            <= TOLERANCES["shuffle_change_floor"]
        )
    values = {
        "layer_count": float(len(layers)),
        "all_values_finite": float(_walk_finite(layers) and _walk_finite(formula_audit)),
        "observation_bound_violations": float(observation_violations),
        "self_decomposition_max_scaled_residual": max(decomposition_scaled, default=math.inf),
        "formula_fixture_max_scaled_error": float(formula_audit["max_scaled_error"]),
        "common_map_failure_count": float(
            sum(
                not bool(layer["eq76_projection"]["empirical_common_map_passed"])
                for layer in layers
            )
        ),
        "shuffle_marginal_max_scaled_error": max(shuffle_scaled, default=math.inf),
        "shuffle_witness_failure_count": float(shuffle_failures),
        "gaussian_hypothesis_pass_gate_count": float(hypothesis_gates),
    }
    expected = _expected_check_ranges()
    return {
        name: {
            "value": float(value),
            "minimum": float(expected[name]["minimum"]),
            "maximum": float(expected[name]["maximum"]),
            "units": expected[name]["units"],
            "passed": math.isfinite(float(value))
            and float(expected[name]["minimum"]) <= float(value) <= float(expected[name]["maximum"]),
        }
        for name, value in values.items()
    }


def enforce_checks(checks: Mapping[str, Mapping[str, Any]]) -> None:
    expected = _expected_check_ranges()
    if set(checks) != set(expected):
        _fail("check registry keys differ from the declared registry")
    failures = []
    for name, declared in expected.items():
        record = checks[name]
        if set(record) != {"value", "minimum", "maximum", "units", "passed"}:
            failures.append(f"{name}: malformed record")
            continue
        value = float(record["value"])
        passed = (
            math.isfinite(value)
            and float(declared["minimum"]) <= value <= float(declared["maximum"])
            and record["minimum"] == declared["minimum"]
            and record["maximum"] == declared["maximum"]
            and record["units"] == declared["units"]
        )
        if record["passed"] is not passed or not passed:
            failures.append(f"{name}: {value}")
    if failures:
        _fail("audit quality gates failed: " + "; ".join(failures))


def _configuration(
    *,
    seed: int,
    max_examples: int,
    batch_size: int,
    spatial_positions: int,
    channels_per_layer: int,
    pairs_per_layer: int,
    bvn_order: int,
) -> dict[str, Any]:
    positive = {
        "max_examples": max_examples,
        "batch_size": batch_size,
        "spatial_positions": spatial_positions,
        "channels_per_layer": channels_per_layer,
        "pairs_per_layer": pairs_per_layer,
    }
    if not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    if any(not isinstance(value, int) or value <= 0 for value in positive.values()):
        raise ValueError("sample and selection bounds must be positive integers")
    if max_examples > 10_000:
        raise ValueError("max_examples exceeds the CIFAR-10 test-set bound")
    if spatial_positions > 1024 or channels_per_layer > 512 or pairs_per_layer > 120:
        raise ValueError("a deterministic sampling bound exceeds its declared maximum")
    if bvn_order < 32 or bvn_order > 512:
        raise ValueError("bvn_order must lie in [32, 512]")
    return {
        "seed": seed,
        "max_examples": max_examples,
        "batch_size": batch_size,
        "spatial_positions_per_example": spatial_positions,
        "channels_per_layer": channels_per_layer,
        "pairs_per_layer": pairs_per_layer,
        "bivariate_cdf_quadrature_order": bvn_order,
        "device": "cpu",
        "data_loader_workers": 0,
        "example_selection": "sorted seeded near-equal class-stratified sample without replacement",
        "channel_selection": "deterministic evenly spaced indices",
        "spatial_selection": "deterministic evenly spaced output positions",
        "pair_selection": "deterministic ordered channel pairs",
        "tolerances": dict(TOLERANCES),
    }


def _canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_json(payload: Any) -> str:
    return _sha256_bytes(_canonical_json_bytes(payload))


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.name


def _sanitize_invocation_token(token: str) -> str:
    """Remove host-specific absolute prefixes from a recorded command token."""
    prefix = ""
    value = token
    if "=" in token:
        candidate_prefix, candidate_value = token.split("=", 1)
        if candidate_value.startswith("/"):
            prefix = candidate_prefix + "="
            value = candidate_value
    if not value.startswith("/"):
        return token
    return prefix + _display_path(Path(value))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, allow_nan=False, ensure_ascii=True, indent=2, sort_keys=True)
        handle.write("\n")


def summary_rows(summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def add(
        section: str,
        layer: str,
        case_id: str,
        metric: str,
        value: Any,
        *,
        channel_i: Any = "",
        channel_j: Any = "",
        reference: Any = "",
        sample_count: Any = "",
        origin: str,
        evidence: str,
        gate: bool = False,
        passed: Any = "",
        notes: str = "",
    ) -> None:
        numeric_value = float(value)
        if reference == "":
            absolute = relative = ""
        else:
            numeric_reference = float(reference)
            absolute = abs(numeric_value - numeric_reference)
            relative = absolute / max(abs(numeric_reference), np.finfo(np.float64).tiny)
        rows.append(
            {
                "section": section,
                "layer": layer,
                "case_id": case_id,
                "channel_i": channel_i,
                "channel_j": channel_j,
                "metric": metric,
                "value": numeric_value,
                "reference_value": reference,
                "abs_error": absolute,
                "rel_error": relative,
                "sample_count": sample_count,
                "method_origin": origin,
                "evidence_label": evidence,
                "is_quality_gate": str(gate).lower(),
                "passed": passed,
                "notes": notes,
            }
        )

    for layer in summary["layers"]:
        for record in layer["normality"]:
            for metric in (
                "mean",
                "variance",
                "skewness",
                "excess_kurtosis",
                "ks_distance_fitted_normal",
                "nonpositive_fraction",
            ):
                add(
                    "normality",
                    layer["layer"],
                    f"{record['domain']}_ch{record['channel']}",
                    metric,
                    record[metric],
                    channel_i=record["channel"],
                    sample_count=record["sample_count"],
                    origin="independent_checkpoint_measurement",
                    evidence="verified_fact",
                    notes="Gaussian-fit diagnostic, not a pass threshold",
                )
        for record in layer["moment_comparisons"]:
            comparisons = (
                (
                    "pre_bn_mean",
                    record["pre_bn_empirical_mean"],
                    record["pre_bn_running_mean"],
                ),
                (
                    "pre_bn_variance",
                    record["pre_bn_empirical_variance"],
                    record["pre_bn_running_variance"],
                ),
                (
                    "post_bn_mean",
                    record["post_bn_empirical_mean"],
                    record["post_bn_expected_mean_beta"],
                ),
                (
                    "post_bn_variance",
                    record["post_bn_empirical_variance"],
                    record["post_bn_finite_epsilon_expected_variance"],
                ),
            )
            for metric, value, reference in comparisons:
                add(
                    "moment_comparison",
                    layer["layer"],
                    f"ch{record['channel']}",
                    metric,
                    value,
                    channel_i=record["channel"],
                    reference=reference,
                    sample_count=record["sample_count"],
                    origin="independent_checkpoint_measurement",
                    evidence="verified_fact",
                    notes="Checkpoint discrepancy diagnostic, not a pass threshold",
                )
        for record in layer["self_kernels"]:
            add(
                "self_kernel",
                layer["layer"],
                f"ch{record['channel']}",
                "empirical_vs_gaussian_empirical_moments",
                record["empirical_relu_second_moment"],
                channel_i=record["channel"],
                reference=record["gaussian_empirical_moments"],
                sample_count=record["sample_count"],
                origin="paper_formula_with_empirical_moments",
                evidence="independent_inference",
                notes="Hypothesis-fit diagnostic, not a pass threshold",
            )
            add(
                "self_kernel_decomposition",
                layer["layer"],
                f"ch{record['channel']}",
                "decomposition_residual",
                record["decomposition_residual"],
                channel_i=record["channel"],
                reference=0.0,
                sample_count=record["sample_count"],
                origin="paper_formula_with_empirical_moments",
                evidence="verified_fact",
                gate=True,
                passed=str(record["decomposition_passed"]).lower(),
            )
        for record in layer["pair_kernels"]:
            add(
                "pair_kernel",
                layer["layer"],
                f"ch{record['channel_i']}_ch{record['channel_j']}",
                "empirical_vs_gaussian_exact_biased",
                record["empirical_relu_pair_kernel"],
                channel_i=record["channel_i"],
                channel_j=record["channel_j"],
                reference=record["gaussian_exact_biased_pair_kernel"],
                sample_count=record["sample_count"],
                origin="paper_formula_with_empirical_moments",
                evidence="independent_inference",
                notes="Hypothesis-fit diagnostic, not a pass threshold",
            )
        witness = layer["shuffle_witness"]
        add(
            "shuffle_witness",
            layer["layer"],
            f"ch{witness['channel_i']}_ch{witness['channel_j']}",
            "dependence_scaled_change",
            witness["dependence_scaled_change"],
            channel_i=witness["channel_i"],
            channel_j=witness["channel_j"],
            sample_count=witness["sample_count"],
            origin="deterministic_dependence_counterexample",
            evidence="counterexample",
            gate=True,
            passed=str(
                witness["dependence_scaled_change"]
                > TOLERANCES["shuffle_change_floor"]
            ).lower(),
            notes="Marginals are preserved by permutation",
        )
        projection = layer["eq76_projection"]
        add(
            "eq76_projection",
            layer["layer"],
            "stored_running_mean",
            "relative_projection_residual",
            projection["stored_running_mean_projection"]["relative_residual"],
            origin="checkpoint_condition_measurement",
            evidence="verified_fact",
            notes="Measured condition; no width-alone causal claim",
        )
        add(
            "eq76_projection",
            layer["layer"],
            "same_sample_common_map",
            "relative_reconstruction_residual",
            projection["empirical_common_map_relative_residual"],
            origin="checkpoint_condition_measurement",
            evidence="verified_fact",
            gate=True,
            passed=str(projection["empirical_common_map_passed"]).lower(),
        )
    return rows


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def layer_summary_rows(summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Flatten the key per-layer checkpoint diagnostics for tables and visuals."""
    rows = []
    for layer in summary["layers"]:
        post_normality = [
            record for record in layer["normality"] if record["domain"] == "post_bn"
        ]
        moment_records = layer["moment_comparisons"]
        self_records = layer["self_kernels"]
        projection = layer["eq76_projection"]
        witness = layer["shuffle_witness"]
        rows.append(
            {
                "layer": layer["layer"],
                "module_path": layer["module_path"],
                "observation_count": layer["observation_count"],
                "selected_channel_count": len(layer["selected_channels"]),
                "representative_channel": layer["representative_channel"],
                "median_abs_skewness": float(
                    np.median([abs(record["skewness"]) for record in post_normality])
                ),
                "max_abs_skewness": max(abs(record["skewness"]) for record in post_normality),
                "median_abs_excess_kurtosis": float(
                    np.median(
                        [abs(record["excess_kurtosis"]) for record in post_normality]
                    )
                ),
                "max_abs_excess_kurtosis": max(
                    abs(record["excess_kurtosis"]) for record in post_normality
                ),
                "median_ks_distance": float(
                    np.median(
                        [record["ks_distance_fitted_normal"] for record in post_normality]
                    )
                ),
                "max_ks_distance": max(
                    record["ks_distance_fitted_normal"] for record in post_normality
                ),
                "median_pre_bn_mean_absolute_difference": float(
                    np.median(
                        [
                            record["pre_bn_mean_absolute_difference"]
                            for record in moment_records
                        ]
                    )
                ),
                "max_pre_bn_mean_absolute_difference": max(
                    record["pre_bn_mean_absolute_difference"]
                    for record in moment_records
                ),
                "median_pre_bn_variance_symmetric_relative_error": float(
                    np.median(
                        [
                            record["pre_bn_variance_symmetric_relative_error"]
                            for record in moment_records
                        ]
                    )
                ),
                "max_pre_bn_variance_symmetric_relative_error": max(
                    record["pre_bn_variance_symmetric_relative_error"]
                    for record in moment_records
                ),
                "median_post_bn_mean_absolute_difference": float(
                    np.median(
                        [
                            record["post_bn_mean_absolute_difference"]
                            for record in moment_records
                        ]
                    )
                ),
                "max_post_bn_mean_absolute_difference": max(
                    record["post_bn_mean_absolute_difference"]
                    for record in moment_records
                ),
                "median_post_bn_variance_symmetric_relative_error": float(
                    np.median(
                        [
                            record["post_bn_variance_symmetric_relative_error"]
                            for record in moment_records
                        ]
                    )
                ),
                "max_post_bn_variance_symmetric_relative_error": max(
                    record["post_bn_variance_symmetric_relative_error"]
                    for record in moment_records
                ),
                "median_self_kernel_absolute_error": float(
                    np.median(
                        [record["gaussian_fit_absolute_error"] for record in self_records]
                    )
                ),
                "max_self_kernel_absolute_error": max(
                    record["gaussian_fit_absolute_error"] for record in self_records
                ),
                "median_self_kernel_symmetric_relative_error": float(
                    np.median(
                        [
                            record["gaussian_fit_symmetric_relative_error"]
                            for record in self_records
                        ]
                    )
                ),
                "max_self_kernel_symmetric_relative_error": max(
                    record["gaussian_fit_symmetric_relative_error"]
                    for record in self_records
                ),
                "stored_mean_projection_relative_residual": projection[
                    "stored_running_mean_projection"
                ]["relative_residual"],
                "empirical_common_map_relative_residual": projection[
                    "empirical_common_map_relative_residual"
                ],
                "shuffle_dependence_scaled_change": witness["dependence_scaled_change"],
                "gaussian_fit_metrics_are_quality_gates": False,
            }
        )
    return rows


def pair_summary_rows(summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for layer in summary["layers"]:
        for record in layer["pair_kernels"]:
            rows.append(
                {
                    "layer": layer["layer"],
                    "module_path": layer["module_path"],
                    "channel_i": record["channel_i"],
                    "channel_j": record["channel_j"],
                    "sample_count": record["sample_count"],
                    "empirical_mean_i": record["empirical_mean_i"],
                    "empirical_mean_j": record["empirical_mean_j"],
                    "empirical_std_i": record["empirical_std_i"],
                    "empirical_std_j": record["empirical_std_j"],
                    "empirical_correlation": record["empirical_correlation"],
                    "empirical_relu_pair_kernel": record[
                        "empirical_relu_pair_kernel"
                    ],
                    "gaussian_exact_biased_pair_kernel": record[
                        "gaussian_exact_biased_pair_kernel"
                    ],
                    "gaussian_fit_absolute_error": record[
                        "gaussian_fit_absolute_error"
                    ],
                    "gaussian_fit_symmetric_relative_error": record[
                        "gaussian_fit_symmetric_relative_error"
                    ],
                    "gaussian_fit_is_quality_gate": False,
                }
            )
    return rows


def histogram_rows(summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for layer in summary["layers"]:
        histogram = layer["standardized_post_bn_histogram"]
        for record in histogram["rows"]:
            rows.append(
                {
                    "layer": layer["layer"],
                    "module_path": layer["module_path"],
                    "domain": "post_bn_affine_pre_relu",
                    "channel": layer["representative_channel"],
                    "selection_rule": layer["representative_channel_selection"],
                    "sample_count": histogram["sample_count"],
                    "standardization_mean": histogram["standardization_mean"],
                    "standardization_std": histogram["standardization_std"],
                    "lower_tail_fraction": histogram["lower_tail_fraction"],
                    "upper_tail_fraction": histogram["upper_tail_fraction"],
                    "bin_index": record["bin_index"],
                    "bin_lower": record["bin_lower"],
                    "bin_upper": record["bin_upper"],
                    "bin_center": record["bin_center"],
                    "empirical_density": record["empirical_density"],
                    "standard_normal_density": record["standard_normal_density"],
                }
            )
    return rows


def _write_table_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise AuditFailure(f"cannot write an empty table: {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _git_value(*args: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(ROOT), *args],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"
    return result.stdout.strip()


def _source_hashes() -> list[dict[str, Any]]:
    records = []
    for relative in SOURCE_RELATIVE_PATHS:
        path = ROOT / relative
        if not path.is_file():
            raise AuditFailure(f"audit source is missing: {relative}")
        records.append(
            {"path": relative, "sha256": _sha256_path(path), "bytes": path.stat().st_size}
        )
    return records


def _code_identity() -> dict[str, Any]:
    commit = _git_value("rev-parse", "HEAD")
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        return {
            "source_snapshot_commit": None,
            "source_snapshot_state": "source_archive_with_individually_hashed_sources",
            "source_files_match_commit": False,
        }
    status = _git_value("status", "--porcelain", "--", *SOURCE_RELATIVE_PATHS)
    exact = status == ""
    return {
        "source_snapshot_commit": commit,
        "source_snapshot_state": (
            "exact_commit" if exact else "working_tree_with_individually_hashed_sources"
        ),
        "source_files_match_commit": exact,
    }


def _paper_identity(paper_pdf: Path | None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = json.loads(PAPER_MANIFEST_PATH.read_text(encoding="utf-8"))
    if manifest.get("pdf_sha256") != PAPER_SHA256 or manifest.get("version") != "v1":
        raise AuditFailure("paper manifest is not pinned to the required v1 PDF")
    local = {"filename": None, "sha256": None, "status": "not_supplied"}
    inputs = [
        {
            "kind": "paper_manifest",
            "path": "paper/manifest.json",
            "sha256": _sha256_path(PAPER_MANIFEST_PATH),
            "bytes": PAPER_MANIFEST_PATH.stat().st_size,
        }
    ]
    if paper_pdf is not None:
        if not paper_pdf.is_file() or _sha256_path(paper_pdf) != PAPER_SHA256:
            raise AuditFailure("explicit paper PDF does not match the required checksum")
        local = {"filename": paper_pdf.name, "sha256": PAPER_SHA256, "status": "verified"}
        inputs.append(
            {
                "kind": "paper_pdf",
                "path": paper_pdf.name,
                "sha256": PAPER_SHA256,
                "bytes": paper_pdf.stat().st_size,
            }
        )
    return (
        {
            "arxiv_id": manifest["arxiv_id"],
            "version": manifest["version"],
            "pdf_sha256": PAPER_SHA256,
            "anchors": [
                "Section 4, PDF pages 6-7",
                "PDF page 52, Eqs. 76-78",
                "PDF page 55, Eq. 83",
                "PDF page 52, Footnote 11",
            ],
            "local_pdf": local,
        },
        inputs,
    )


def _environment(provenance: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "numpy_version": np.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        **dict(provenance["runtime"]),
    }


def _require(condition: bool, message: str) -> None:
    if not condition:
        _fail(message)


def _require_close(
    actual: Any,
    expected: Any,
    message: str,
    *,
    rtol: float = 2e-12,
    atol: float = 2e-14,
) -> None:
    if isinstance(actual, bool) or isinstance(expected, bool):
        _fail(message)
    try:
        actual_value = float(actual)
        expected_value = float(expected)
    except (TypeError, ValueError, OverflowError) as error:
        raise AuditFailure(message) from error
    if not (
        math.isfinite(actual_value)
        and math.isfinite(expected_value)
        and math.isclose(actual_value, expected_value, rel_tol=rtol, abs_tol=atol)
    ):
        _fail(message)


def _require_integer(value: Any, message: str, *, minimum: int = 0) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, np.integer))
        or int(value) < minimum
    ):
        _fail(message)
    return int(value)


def _require_sha256(value: Any, message: str) -> None:
    _require(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value),
        message,
    )


def _validate_projection_record(
    record: Any,
    *,
    rows: int,
    columns: int,
    label: str,
) -> None:
    _require(isinstance(record, Mapping), f"{label} projection record is missing")
    _require_integer(record.get("rows"), f"{label} projection rows differ")
    _require_integer(record.get("columns"), f"{label} projection columns differ")
    _require(
        record["rows"] == rows and record["columns"] == columns,
        f"{label} projection dimensions differ",
    )
    rank = _require_integer(record.get("rank"), f"{label} projection rank is invalid")
    _require(rank <= min(rows, columns), f"{label} projection rank exceeds its dimensions")
    target_norm = float(record["target_norm"])
    residual_norm = float(record["residual_norm"])
    max_abs_residual = float(record["max_abs_residual"])
    minimum_singular = float(record["minimum_singular_value"])
    maximum_singular = float(record["maximum_singular_value"])
    _require(
        min(target_norm, residual_norm, max_abs_residual, minimum_singular) >= 0.0
        and maximum_singular >= minimum_singular,
        f"{label} projection contains invalid norms or singular values",
    )
    _require(
        max_abs_residual <= residual_norm + 2e-12 * max(1.0, residual_norm),
        f"{label} projection maximum residual exceeds its norm",
    )
    _require_close(
        record["relative_residual"],
        residual_norm / max(target_norm, np.finfo(np.float64).tiny),
        f"{label} projection residual ratio differs",
    )


def _validate_formula_audit(formula: Any, order: int) -> None:
    _require(isinstance(formula, Mapping), "formula audit is missing")
    expected = fixed_formula_audit(order)
    cases = formula.get("cases")
    _require(isinstance(cases, list), "formula-audit cases are missing")
    _require(len(cases) == len(expected["cases"]), "formula-audit case count differs")
    for record, reference in zip(cases, expected["cases"], strict=True):
        _require(isinstance(record, Mapping), "formula-audit case is malformed")
        _require(
            record.get("case_id") == reference["case_id"],
            "formula-audit case registry differs",
        )
        for field in ("value", "reference", "scaled_error"):
            _require_close(
                record.get(field),
                reference[field],
                f"formula-audit {record.get('case_id')} {field} differs",
                rtol=3e-12,
                atol=3e-14,
            )
        _require(
            record.get("passed") is reference["passed"],
            "formula-audit case pass flag differs",
        )
    _require_close(
        formula.get("max_scaled_error"),
        expected["max_scaled_error"],
        "formula-audit maximum scaled error differs",
        rtol=3e-12,
        atol=3e-14,
    )
    _require(
        formula.get("all_cases_passed") is expected["all_cases_passed"],
        "formula-audit aggregate pass flag differs",
    )


def _validate_layer_record(layer: Any, config: Mapping[str, Any], name: str) -> None:
    _require(isinstance(layer, Mapping), f"{name} layer record is malformed")
    _require(layer.get("layer") == name, f"{name} layer identity differs")
    _require(layer.get("module_path") == LAYER_MODULE_PATHS[name], f"{name} module path differs")
    observation_count = _require_integer(
        layer.get("observation_count"), f"{name} observation count is invalid", minimum=1
    )
    maximum_observations = int(config["max_examples"]) * int(
        config["spatial_positions_per_example"]
    )
    _require(observation_count <= maximum_observations, f"{name} observation bound is exceeded")

    eq76 = layer.get("eq76_projection")
    _require(isinstance(eq76, Mapping), f"{name} Eq. 76 record is missing")
    _require(eq76.get("layer") == name, f"{name} Eq. 76 layer identity differs")
    rows = _require_integer(eq76.get("rows"), f"{name} Eq. 76 rows are invalid", minimum=1)
    columns = _require_integer(
        eq76.get("columns"), f"{name} Eq. 76 columns are invalid", minimum=1
    )
    expected_regime = "expansion" if rows > columns else "non_expansion"
    _require(eq76.get("width_regime") == expected_regime, f"{name} Eq. 76 width regime differs")
    _require(
        eq76.get("causal_width_claim_made") is False,
        f"{name} makes an unsupported width-alone claim",
    )

    selected = layer.get("selected_channels")
    _require(isinstance(selected, list), f"{name} selected-channel registry is missing")
    expected_selected = select_evenly(rows, int(config["channels_per_layer"]))
    _require(selected == expected_selected, f"{name} selected-channel registry differs")
    representative = expected_selected[len(expected_selected) // 2]
    _require(
        layer.get("representative_channel") == representative,
        f"{name} representative channel differs",
    )
    _require(
        layer.get("representative_channel_selection")
        == "middle entry of the deterministic evenly spaced channel selection",
        f"{name} representative-channel selection rule differs",
    )

    normality = layer.get("normality")
    _require(isinstance(normality, list), f"{name} normality registry is missing")
    expected_normality_registry = [
        (channel, domain)
        for channel in expected_selected
        for domain in ("pre_bn", "post_bn")
    ]
    actual_normality_registry = [
        (record.get("channel"), record.get("domain"))
        for record in normality
        if isinstance(record, Mapping)
    ]
    _require(
        len(actual_normality_registry) == len(normality)
        and actual_normality_registry == expected_normality_registry,
        f"{name} normality channel/domain registry differs",
    )
    normality_by_key: dict[tuple[int, str], Mapping[str, Any]] = {}
    for record in normality:
        channel = int(record["channel"])
        domain = str(record["domain"])
        _require(record.get("layer") == name, f"{name} normality layer identity differs")
        _require(
            record.get("sample_count") == observation_count,
            f"{name} normality sample registry differs",
        )
        variance = float(record["variance"])
        std = float(record["std"])
        _require(variance >= 0.0 and std >= 0.0, f"{name} normality variance is invalid")
        _require_close(std * std, variance, f"{name} normality variance/std identity differs")
        _require(
            0.0 <= float(record["ks_distance_fitted_normal"]) <= 1.0
            and 0.0 <= float(record["nonpositive_fraction"]) <= 1.0
            and float(record["relu_second_moment"]) >= 0.0,
            f"{name} normality bounded metric is invalid",
        )
        normality_by_key[(channel, domain)] = record

    moments = layer.get("moment_comparisons")
    _require(isinstance(moments, list), f"{name} moment registry is missing")
    _require(
        [record.get("channel") for record in moments if isinstance(record, Mapping)]
        == expected_selected
        and len(moments) == len(expected_selected),
        f"{name} moment channel registry differs",
    )
    moments_by_channel: dict[int, Mapping[str, Any]] = {}
    moment_derived = (
        ("pre_bn_mean", "pre_bn_empirical_mean", "pre_bn_running_mean"),
        ("pre_bn_variance", "pre_bn_empirical_variance", "pre_bn_running_variance"),
        ("post_bn_mean", "post_bn_empirical_mean", "post_bn_expected_mean_beta"),
        (
            "post_bn_variance",
            "post_bn_empirical_variance",
            "post_bn_finite_epsilon_expected_variance",
        ),
    )
    for record in moments:
        channel = int(record["channel"])
        pre_diag = normality_by_key[(channel, "pre_bn")]
        post_diag = normality_by_key[(channel, "post_bn")]
        _require(record.get("layer") == name, f"{name} moment layer identity differs")
        _require(
            record.get("sample_count") == observation_count,
            f"{name} moment sample registry differs",
        )
        _require_close(
            record["pre_bn_empirical_mean"],
            pre_diag["mean"],
            f"{name} pre-BN mean link differs",
        )
        _require_close(
            record["pre_bn_empirical_variance"],
            pre_diag["variance"],
            f"{name} pre-BN variance link differs",
        )
        _require_close(
            record["post_bn_empirical_mean"],
            post_diag["mean"],
            f"{name} post-BN mean link differs",
        )
        _require_close(
            record["post_bn_empirical_variance"],
            post_diag["variance"],
            f"{name} post-BN variance link differs",
        )
        _require(
            float(record["pre_bn_running_variance"]) >= 0.0
            and float(record["post_bn_finite_epsilon_expected_variance"]) >= 0.0,
            f"{name} stored variance is invalid",
        )
        for prefix, value_field, reference_field in moment_derived:
            value = float(record[value_field])
            reference = float(record[reference_field])
            _require_close(
                record[f"{prefix}_absolute_difference"],
                abs(value - reference),
                f"{name} {prefix} absolute difference differs",
            )
            _require_close(
                record[f"{prefix}_symmetric_relative_error"],
                symmetric_relative_error(value, reference),
                f"{name} {prefix} symmetric error differs",
            )
        _require(
            record.get("moment_discrepancies_are_quality_gates") is False,
            f"{name} moment discrepancy was converted into a pass gate",
        )
        moments_by_channel[channel] = record

    self_kernels = layer.get("self_kernels")
    _require(isinstance(self_kernels, list), f"{name} self-kernel registry is missing")
    _require(
        [record.get("channel") for record in self_kernels if isinstance(record, Mapping)]
        == expected_selected
        and len(self_kernels) == len(expected_selected),
        f"{name} self-kernel channel registry differs",
    )
    for record in self_kernels:
        channel = int(record["channel"])
        post_diag = normality_by_key[(channel, "post_bn")]
        moment = moments_by_channel[channel]
        _require(record.get("layer") == name, f"{name} self-kernel layer identity differs")
        _require(
            record.get("sample_count") == observation_count,
            f"{name} self-kernel sample registry differs",
        )
        empirical = float(record["empirical_relu_second_moment"])
        fitted = relu_gaussian_self_kernel(float(post_diag["mean"]), float(post_diag["std"]))
        finite_state = relu_gaussian_self_kernel(
            float(moment["post_bn_expected_mean_beta"]),
            math.sqrt(float(moment["post_bn_finite_epsilon_expected_variance"])),
        )
        negligible = float(record["gaussian_bn_negligible_epsilon"])
        _require_close(
            empirical,
            post_diag["relu_second_moment"],
            f"{name} self-kernel empirical link differs",
        )
        _require_close(
            record["gaussian_empirical_moments"],
            fitted,
            f"{name} moment-matched self kernel differs",
        )
        _require_close(
            record["gaussian_bn_finite_epsilon"],
            finite_state,
            f"{name} finite-epsilon self kernel differs",
        )
        _require(negligible >= 0.0, f"{name} negligible-epsilon self kernel is invalid")
        _require_close(
            record["gaussian_fit_absolute_error"],
            abs(empirical - fitted),
            f"{name} self-kernel absolute error differs",
        )
        _require_close(
            record["gaussian_fit_symmetric_relative_error"],
            symmetric_relative_error(empirical, fitted),
            f"{name} self-kernel symmetric error differs",
        )
        shape = empirical - fitted
        moment_state = fitted - finite_state
        epsilon_error = finite_state - negligible
        total = empirical - negligible
        residual = total - (shape + moment_state + epsilon_error)
        tolerance = TOLERANCES["decomposition_atol"] + TOLERANCES[
            "decomposition_rtol"
        ] * abs(total)
        for field, expected in (
            ("shape_error", shape),
            ("moment_state_error", moment_state),
            ("finite_epsilon_error", epsilon_error),
            ("total_error", total),
            ("decomposition_residual", residual),
            ("decomposition_tolerance", tolerance),
        ):
            _require_close(record[field], expected, f"{name} self-kernel {field} differs")
        _require(
            record.get("decomposition_passed") is (abs(residual) <= tolerance),
            f"{name} self-kernel decomposition pass flag differs",
        )
        _require(
            record.get("hypothesis_fit_is_quality_gate") is False,
            f"{name} self-kernel hypothesis error was converted into a pass gate",
        )

    pairs = layer.get("pair_kernels")
    _require(isinstance(pairs, list), f"{name} pair-kernel registry is missing")
    expected_pairs = select_pairs(expected_selected, int(config["pairs_per_layer"]))
    actual_pairs = [
        (record.get("channel_i"), record.get("channel_j"))
        for record in pairs
        if isinstance(record, Mapping)
    ]
    _require(
        len(actual_pairs) == len(pairs) and actual_pairs == expected_pairs,
        f"{name} pair-kernel registry differs",
    )
    pairs_by_key: dict[tuple[int, int], Mapping[str, Any]] = {}
    for record in pairs:
        channel_i = int(record["channel_i"])
        channel_j = int(record["channel_j"])
        first_diag = normality_by_key[(channel_i, "post_bn")]
        second_diag = normality_by_key[(channel_j, "post_bn")]
        _require(record.get("layer") == name, f"{name} pair-kernel layer identity differs")
        _require(
            record.get("sample_count") == observation_count,
            f"{name} pair-kernel sample registry differs",
        )
        for field, expected in (
            ("empirical_mean_i", first_diag["mean"]),
            ("empirical_mean_j", second_diag["mean"]),
            ("empirical_std_i", first_diag["std"]),
            ("empirical_std_j", second_diag["std"]),
        ):
            _require_close(record[field], expected, f"{name} pair-kernel {field} link differs")
        std_product = float(record["empirical_std_i"]) * float(record["empirical_std_j"])
        covariance = float(record["empirical_covariance"])
        _require(
            abs(covariance) <= std_product + 2e-12 * max(1.0, std_product),
            f"{name} pair covariance violates its moment bound",
        )
        expected_correlation = (
            0.0
            if std_product <= TOLERANCES["variance_floor"]
            else float(np.clip(covariance / std_product, -1.0, 1.0))
        )
        _require_close(
            record["empirical_correlation"],
            expected_correlation,
            f"{name} pair correlation differs",
        )
        gaussian = relu_gaussian_pair_kernel(
            float(record["empirical_mean_i"]),
            float(record["empirical_mean_j"]),
            float(record["empirical_std_i"]),
            float(record["empirical_std_j"]),
            float(record["empirical_correlation"]),
            order=int(config["bivariate_cdf_quadrature_order"]),
        )
        empirical = float(record["empirical_relu_pair_kernel"])
        _require(empirical >= 0.0, f"{name} empirical pair kernel is invalid")
        _require_close(
            record["gaussian_exact_biased_pair_kernel"],
            gaussian,
            f"{name} exact biased pair kernel differs",
            rtol=3e-12,
            atol=3e-14,
        )
        fit_error = empirical - gaussian
        for field, expected in (
            ("gaussian_fit_error", fit_error),
            ("gaussian_fit_absolute_error", abs(fit_error)),
            (
                "gaussian_fit_symmetric_relative_error",
                symmetric_relative_error(empirical, gaussian),
            ),
            (
                "gaussian_fit_relative_error",
                fit_error / max(abs(gaussian), np.finfo(np.float64).tiny),
            ),
        ):
            _require_close(record[field], expected, f"{name} pair-kernel {field} differs")
        _require(
            record.get("hypothesis_fit_is_quality_gate") is False,
            f"{name} pair-kernel hypothesis error was converted into a pass gate",
        )
        pairs_by_key[(channel_i, channel_j)] = record

    witness = layer.get("shuffle_witness")
    _require(isinstance(witness, Mapping), f"{name} shuffle witness is missing")
    witness_key = (witness.get("channel_i"), witness.get("channel_j"))
    _require(witness_key in pairs_by_key, f"{name} shuffle witness pair is not registered")
    original_pair = pairs_by_key[witness_key]
    _require(witness.get("layer") == name, f"{name} shuffle layer identity differs")
    _require(
        witness.get("sample_count") == observation_count,
        f"{name} shuffle sample registry differs",
    )
    _require_sha256(
        witness.get("permutation_sha256"),
        f"{name} shuffle permutation digest is invalid",
    )
    _require_close(
        witness["original_covariance"],
        original_pair["empirical_covariance"],
        f"{name} shuffle covariance link differs",
    )
    _require_close(
        witness["original_relu_pair_kernel"],
        original_pair["empirical_relu_pair_kernel"],
        f"{name} shuffle pair-kernel link differs",
    )
    shuffled_kernel = float(witness["shuffled_relu_pair_kernel"])
    _require(
        float(witness["max_marginal_abs_difference"]) >= 0.0 and shuffled_kernel >= 0.0,
        f"{name} shuffle witness contains an invalid nonnegative metric",
    )
    dependence_change = max(
        abs(float(witness["shuffled_covariance"]) - float(witness["original_covariance"])),
        abs(shuffled_kernel - float(witness["original_relu_pair_kernel"])),
    )
    witness_scale = max(
        abs(float(witness["original_covariance"])),
        abs(float(witness["original_relu_pair_kernel"])),
        float(original_pair["empirical_std_i"]) * float(original_pair["empirical_std_j"]),
        np.finfo(np.float64).tiny,
    )
    _require(
        abs(float(witness["shuffled_covariance"]))
        <= float(original_pair["empirical_std_i"])
        * float(original_pair["empirical_std_j"])
        + 2e-12
        * max(
            1.0,
            float(original_pair["empirical_std_i"])
            * float(original_pair["empirical_std_j"]),
        ),
        f"{name} shuffled covariance violates its marginal-moment bound",
    )
    _require_close(
        witness["dependence_abs_change"],
        dependence_change,
        f"{name} shuffle dependence change differs",
    )
    _require_close(
        witness["dependence_scaled_change"],
        dependence_change / witness_scale,
        f"{name} shuffle dependence scale differs",
    )

    stored_projection = eq76.get("stored_running_mean_projection")
    empirical_projection = eq76.get("empirical_pre_mean_projection")
    _validate_projection_record(
        stored_projection, rows=rows, columns=columns, label=f"{name} stored-mean"
    )
    _validate_projection_record(
        empirical_projection, rows=rows, columns=columns, label=f"{name} empirical-mean"
    )
    _require(
        stored_projection["rank"] == empirical_projection["rank"],
        f"{name} Eq. 76 projection ranks differ for the same matrix",
    )
    for field in ("minimum_singular_value", "maximum_singular_value"):
        _require_close(
            stored_projection[field],
            empirical_projection[field],
            f"{name} Eq. 76 {field} differs for the same matrix",
        )
    common_residual = float(eq76["empirical_common_map_residual_norm"])
    common_scale = float(empirical_projection["target_norm"])
    _require(common_residual >= 0.0, f"{name} common-map residual is invalid")
    _require_close(
        eq76["empirical_common_map_relative_residual"],
        common_residual / max(common_scale, np.finfo(np.float64).tiny),
        f"{name} common-map residual ratio differs",
    )
    common_tolerance = TOLERANCES["common_map_atol"] + TOLERANCES[
        "common_map_rtol"
    ] * common_scale
    _require_close(
        eq76["empirical_common_map_tolerance"],
        common_tolerance,
        f"{name} common-map tolerance differs",
    )
    _require(
        eq76.get("empirical_common_map_passed") is (common_residual <= common_tolerance),
        f"{name} common-map pass flag differs",
    )
    _require(
        float(eq76["eq76_reconstructed_post_mean_max_abs_delta_from_beta"]) >= 0.0,
        f"{name} Eq. 76 reconstructed-mean delta is invalid",
    )
    _require(
        float(empirical_projection["residual_norm"])
        <= common_residual + 2e-12 * max(1.0, common_residual),
        f"{name} least-squares residual exceeds the common-map candidate residual",
    )

    histogram = layer.get("standardized_post_bn_histogram")
    _require(isinstance(histogram, Mapping), f"{name} histogram is missing")
    representative_diag = normality_by_key[(representative, "post_bn")]
    _require(
        histogram.get("sample_count") == observation_count,
        f"{name} histogram sample registry differs",
    )
    _require_close(
        histogram["standardization_mean"],
        representative_diag["mean"],
        f"{name} histogram mean link differs",
    )
    _require_close(
        histogram["standardization_std"],
        representative_diag["std"],
        f"{name} histogram std link differs",
    )
    bin_count = _require_integer(
        histogram.get("bin_count"),
        f"{name} histogram bin count is invalid",
        minimum=1,
    )
    _require(bin_count == 40, f"{name} histogram bin count differs from the fixed audit grid")
    z_limit = float(histogram["z_limit"])
    _require_close(z_limit, 5.0, f"{name} histogram z limit differs")
    histogram_records = histogram.get("rows")
    _require(
        isinstance(histogram_records, list) and len(histogram_records) == bin_count,
        f"{name} histogram row count differs",
    )
    width = 2.0 * z_limit / bin_count
    empirical_mass = 0.0
    for index, record in enumerate(histogram_records):
        _require(isinstance(record, Mapping), f"{name} histogram row is malformed")
        _require(record.get("bin_index") == index, f"{name} histogram row registry differs")
        lower = -z_limit + index * width
        upper = lower + width
        _require_close(record["bin_lower"], lower, f"{name} histogram lower grid edge differs")
        _require_close(record["bin_upper"], upper, f"{name} histogram upper grid edge differs")
        _require_close(
            record["bin_center"],
            0.5 * (lower + upper),
            f"{name} histogram center differs",
        )
        density = float(record["empirical_density"])
        _require(density >= 0.0, f"{name} histogram empirical density is invalid")
        count = density * observation_count * width
        _require_close(
            count,
            round(count),
            f"{name} histogram density is not a sample count",
            rtol=0.0,
            atol=2e-8,
        )
        normal_density = (_normal_cdf(upper) - _normal_cdf(lower)) / width
        _require_close(
            record["standard_normal_density"],
            normal_density,
            f"{name} histogram normal density differs",
        )
        empirical_mass += density * width
    lower_tail = float(histogram["lower_tail_fraction"])
    upper_tail = float(histogram["upper_tail_fraction"])
    _require(
        0.0 <= lower_tail <= 1.0 and 0.0 <= upper_tail <= 1.0,
        f"{name} histogram tail fraction is invalid",
    )
    for tail, side in ((lower_tail, "lower"), (upper_tail, "upper")):
        tail_count = tail * observation_count
        _require_close(
            tail_count,
            round(tail_count),
            f"{name} histogram {side}-tail count differs",
            rtol=0.0,
            atol=2e-8,
        )
    _require_close(
        empirical_mass + lower_tail + upper_tail,
        1.0,
        f"{name} histogram mass does not sum to one",
        rtol=0.0,
        atol=5e-12,
    )


def _validate_summary_payload(payload: Mapping[str, Any]) -> None:
    if not isinstance(payload, Mapping):
        _fail("summary root must be an object")
    if not _walk_finite(payload):
        _fail("summary contains a non-finite number or unsupported value")
    if (
        payload.get("schema_version") != AUDIT_SCHEMA_VERSION
        or payload.get("audit_id") != AUDIT_ID
    ):
        _fail("summary schema or audit ID differs")
    if payload.get("claim_ids") != list(CLAIM_IDS):
        _fail("summary claim IDs differ")
    if payload.get("paper", {}).get("pdf_sha256") != PAPER_SHA256:
        _fail("summary paper checksum differs")
    provenance = payload.get("input_provenance")
    if not isinstance(provenance, Mapping):
        _fail("summary input provenance is missing")
    checkpoint_provenance = provenance.get("checkpoint", {})
    if (
        checkpoint_provenance.get("historical_training_provenance", {}).get("status")
        != "incomplete"
    ):
        _fail("historical training provenance must be explicitly incomplete")
    if not isinstance(checkpoint_provenance.get("state_schema_sha256"), str):
        _fail("checkpoint state-schema digest is missing")
    if checkpoint_provenance.get("strict_state_dict_load") is not True:
        _fail("checkpoint was not recorded as a strict state-dictionary load")
    _require_sha256(checkpoint_provenance.get("sha256"), "checkpoint digest is missing")
    _require_sha256(
        checkpoint_provenance.get("state_schema_sha256"),
        "checkpoint state-schema digest is missing",
    )
    dataset_provenance = provenance.get("dataset", {})
    raw_files = dataset_provenance.get("raw_files", [])
    raw_roles = [
        record.get("logical_role")
        for record in raw_files
        if isinstance(record, Mapping)
    ]
    if (
        len(raw_roles) != len(raw_files)
        or set(raw_roles) != {"cifar10_test_batch", "cifar10_batches_meta"}
        or len(raw_roles) != 2
    ):
        _fail("CIFAR-10 raw-file provenance is incomplete")
    if dataset_provenance.get("preprocessing") != {
        "transform": "ToTensor",
        "normalization": "none",
        "augmentation": "none",
    }:
        _fail("dataset preprocessing record differs")
    scope = payload.get("reproduction_scope")
    if (
        not isinstance(scope, Mapping)
        or scope.get("reproduction_scope") != "architecture_adaptation"
        or scope.get("analysis_role") != "diagnostic_audit"
    ):
        _fail("summary reproduction scope differs")
    for flag in (
        "paper_experiment_reproduced",
        "hypothesis_errors_are_pass_gates",
        "width_alone_causal_claim",
    ):
        if scope.get(flag) is not False:
            _fail(f"summary reproduction-scope flag must remain false: {flag}")
    config = payload.get("config")
    if not isinstance(config, Mapping) or _sha256_json(config) != payload.get("config_sha256"):
        _fail("summary configuration hash differs")
    expected_config = _configuration(
        seed=config["seed"],
        max_examples=config["max_examples"],
        batch_size=config["batch_size"],
        spatial_positions=config["spatial_positions_per_example"],
        channels_per_layer=config["channels_per_layer"],
        pairs_per_layer=config["pairs_per_layer"],
        bvn_order=config["bivariate_cdf_quadrature_order"],
    )
    if config != expected_config or config.get("tolerances") != TOLERANCES:
        _fail("summary tolerances differ")
    if payload.get("method_origins") != METHOD_ORIGINS:
        _fail("summary method-origin registry differs")
    _require(dataset_provenance.get("name") == "CIFAR-10", "dataset identity differs")
    _require(dataset_provenance.get("split") == "test", "dataset split differs")
    dataset_length = _require_integer(
        dataset_provenance.get("dataset_length"), "dataset length is invalid", minimum=1
    )
    selected_count = _require_integer(
        dataset_provenance.get("selected_example_count"),
        "selected-example count is invalid",
        minimum=1,
    )
    _require(
        selected_count == int(config["max_examples"]) <= dataset_length,
        "selected-example count differs from the bounded configuration",
    )
    _require(
        dataset_provenance.get("selection_seed") == config["seed"]
        and dataset_provenance.get("selection_rule") == config["example_selection"],
        "dataset selection registry differs",
    )
    _require_sha256(
        dataset_provenance.get("selected_examples_sha256"),
        "selected-example digest is missing",
    )
    _require_sha256(
        dataset_provenance.get("selected_indices_sha256"),
        "selected-index digest is missing",
    )
    layers = payload.get("layers")
    if not isinstance(layers, list) or [
        layer.get("layer") for layer in layers
    ] != list(LAYER_NAMES):
        _fail("summary layer registry differs")
    for name, layer in zip(LAYER_NAMES, layers, strict=True):
        _validate_layer_record(layer, config, name)
    formula = payload.get("formula_audit")
    _validate_formula_audit(formula, int(config["bivariate_cdf_quadrature_order"]))
    recomputed_checks = build_checks(layers, formula, config)
    if payload.get("checks") != recomputed_checks:
        _fail("summary check registry does not match detailed records")
    enforce_checks(recomputed_checks)
    if payload.get("passed") is not True or payload.get("status") != "passed":
        _fail("summary is not marked passed")


def validate_summary(payload: Mapping[str, Any]) -> None:
    """Reapply all invariant gates and reject internally forged diagnostics."""
    try:
        _validate_summary_payload(payload)
    except AuditFailure:
        raise
    except (
        KeyError,
        IndexError,
        TypeError,
        ValueError,
        OverflowError,
        FloatingPointError,
    ) as error:
        raise AuditFailure(f"summary is malformed: {error}") from error


def run_audit(
    *,
    checkpoint_path: Path,
    data_dir: Path,
    result_dir: Path = DEFAULT_RESULT_DIR,
    manifest_path: Path | None = None,
    seed: int = DEFAULT_SEED,
    max_examples: int = DEFAULT_MAX_EXAMPLES,
    batch_size: int = DEFAULT_BATCH_SIZE,
    spatial_positions: int = DEFAULT_SPATIAL_POSITIONS,
    channels_per_layer: int = DEFAULT_CHANNELS_PER_LAYER,
    pairs_per_layer: int = DEFAULT_PAIRS_PER_LAYER,
    bvn_order: int = DEFAULT_BVN_ORDER,
    paper_pdf: Path | None = None,
    invocation: Sequence[str] | None = None,
) -> dict[str, str]:
    config = _configuration(
        seed=seed,
        max_examples=max_examples,
        batch_size=batch_size,
        spatial_positions=spatial_positions,
        channels_per_layer=channels_per_layer,
        pairs_per_layer=pairs_per_layer,
        bvn_order=bvn_order,
    )
    result_dir = result_dir.resolve()
    manifest_path = result_dir / "manifest.json" if manifest_path is None else manifest_path.resolve()
    if manifest_path in {result_dir / "summary.json", result_dir / "metrics.csv"}:
        raise ValueError("manifest path must not replace a result file")
    paper, paper_inputs = _paper_identity(paper_pdf)
    captured, provenance = capture_checkpoint_data(
        checkpoint_path=checkpoint_path.resolve(),
        data_dir=data_dir.resolve(),
        seed=seed,
        max_examples=max_examples,
        batch_size=batch_size,
        spatial_positions=spatial_positions,
        channels_per_layer=channels_per_layer,
    )
    layers = [
        analyze_layer(
            record,
            pairs_per_layer=pairs_per_layer,
            bvn_order=bvn_order,
            seed=seed,
        )
        for record in captured
    ]
    formula_audit = fixed_formula_audit(bvn_order)
    checks = build_checks(layers, formula_audit, config)
    enforce_checks(checks)
    reproduction_scope = {
        "claim": (
            "Independent bounded audit of BN marginal state, Gaussian surrogate fit, "
            "dependence non-identification, and the Eq. 76 projection condition."
        ),
        "reproduction_scope": "architecture_adaptation",
        "analysis_role": "diagnostic_audit",
        "paper_experiment_reproduced": False,
        "model_route": "adapted VGG-8 with a 10-class CIFAR-10 head",
        "dataset_route": "CIFAR-10 test split, not the paper ImageNet experiment",
        "hypothesis_errors_are_pass_gates": False,
        "width_alone_causal_claim": False,
        "limitations": [
            "One checkpoint cannot establish behavior across architectures or training runs.",
            "Spatial positions and examples are deterministically bounded samples.",
            "Empirical pair covariance is used only to isolate Gaussian shape error.",
            "No pruning, compression, fine-tuning, or accuracy comparison is performed.",
        ],
    }
    summary: dict[str, Any] = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "audit_id": AUDIT_ID,
        "status": "passed",
        "passed": True,
        "claim_ids": list(CLAIM_IDS),
        "paper": paper,
        "config": config,
        "config_sha256": _sha256_json(config),
        "method_origins": METHOD_ORIGINS,
        "reproduction_scope": reproduction_scope,
        "input_provenance": {
            "checkpoint": provenance["checkpoint"],
            "dataset": provenance["dataset"],
        },
        "formula_audit": formula_audit,
        "checks": checks,
        "layers": layers,
    }
    validate_summary(summary)
    summary_path = result_dir / "summary.json"
    metrics_path = result_dir / "metrics.csv"
    layer_summary_path = result_dir / "layer_summary.csv"
    pair_summary_path = result_dir / "pair_summary.csv"
    histograms_path = result_dir / "histograms.csv"
    _write_json(summary_path, summary)
    _write_csv(metrics_path, summary_rows(summary))
    _write_table_csv(layer_summary_path, layer_summary_rows(summary))
    _write_table_csv(pair_summary_path, pair_summary_rows(summary))
    _write_table_csv(histograms_path, histogram_rows(summary))
    outputs = [
        {
            "path": _display_path(path),
            "sha256": _sha256_path(path),
            "bytes": path.stat().st_size,
            "media_type": media_type,
        }
        for path, media_type in (
            (metrics_path, "text/csv"),
            (layer_summary_path, "text/csv"),
            (pair_summary_path, "text/csv"),
            (histograms_path, "text/csv"),
            (summary_path, "application/json"),
        )
    ]
    dirty = _git_value("status", "--porcelain", "--untracked-files=all")
    manifest = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "audit_id": AUDIT_ID,
        "status": "passed",
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "claim_ids": list(CLAIM_IDS),
        "paper": paper,
        "config": config,
        "config_sha256": summary["config_sha256"],
        "seeds": {
            "example_selection": seed,
            "shuffle_witness": "seed plus a fixed per-layer offset",
        },
        "code": _code_identity(),
        "environment": _environment(provenance),
        "repository": {
            "git_commit": _git_value("rev-parse", "HEAD"),
            "git_tree_dirty": dirty not in ("", "unavailable"),
        },
        "method_origins": METHOD_ORIGINS,
        "reproduction_scope": reproduction_scope,
        "expected_metric_ranges": _expected_check_ranges(),
        "observed_checks": checks,
        "source_hashes": _source_hashes(),
        "input_hashes": [
            *paper_inputs,
            {
                "kind": "checkpoint",
                "path": provenance["checkpoint"]["filename"],
                "sha256": provenance["checkpoint"]["sha256"],
                "bytes": provenance["checkpoint"]["bytes"],
                "state_schema_sha256": provenance["checkpoint"][
                    "state_schema_sha256"
                ],
                "strict_state_dict_load": provenance["checkpoint"][
                    "strict_state_dict_load"
                ],
                "historical_training_provenance_status": provenance[
                    "checkpoint"
                ]["historical_training_provenance"]["status"],
            },
            {
                "kind": "selected_dataset_examples",
                "path": "CIFAR-10/test/selected-examples",
                "sha256": provenance["dataset"]["selected_examples_sha256"],
                "selected_indices_sha256": provenance["dataset"]["selected_indices_sha256"],
                "example_count": provenance["dataset"]["selected_example_count"],
                "preprocessing": provenance["dataset"]["preprocessing"],
            },
            *[
                {
                    "kind": record["logical_role"],
                    "path": f"CIFAR-10/cifar-10-batches-py/{record['filename']}",
                    "sha256": record["sha256"],
                    "bytes": record["bytes"],
                }
                for record in provenance["dataset"]["raw_files"]
            ],
        ],
        "outputs": outputs,
        "invocation": [
            _sanitize_invocation_token(str(token)) for token in (invocation or sys.argv)
        ],
        "working_directory": _display_path(Path.cwd()),
    }
    _write_json(manifest_path, manifest)
    return {
        "summary": str(summary_path),
        "metrics": str(metrics_path),
        "layer_summary": str(layer_summary_path),
        "pair_summary": str(pair_summary_path),
        "histograms": str(histograms_path),
        "manifest": str(manifest_path),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Measure BN marginals and Gaussian-surrogate diagnostics on the adapted "
            "VGG-8/CIFAR-10 route. This is not a paper experiment reproduction."
        )
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--manifest-path", type=Path)
    parser.add_argument("--paper-pdf", type=Path)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-examples", type=int, default=DEFAULT_MAX_EXAMPLES)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--spatial-positions", type=int, default=DEFAULT_SPATIAL_POSITIONS)
    parser.add_argument("--channels-per-layer", type=int, default=DEFAULT_CHANNELS_PER_LAYER)
    parser.add_argument("--pairs-per-layer", type=int, default=DEFAULT_PAIRS_PER_LAYER)
    parser.add_argument("--bvn-order", type=int, default=DEFAULT_BVN_ORDER)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        outputs = run_audit(
            checkpoint_path=args.checkpoint,
            data_dir=args.data_dir,
            result_dir=args.result_dir,
            manifest_path=args.manifest_path,
            seed=args.seed,
            max_examples=args.max_examples,
            batch_size=args.batch_size,
            spatial_positions=args.spatial_positions,
            channels_per_layer=args.channels_per_layer,
            pairs_per_layer=args.pairs_per_layer,
            bvn_order=args.bvn_order,
            paper_pdf=args.paper_pdf,
            invocation=(sys.executable, str(Path(__file__).resolve()), *(argv or sys.argv[1:])),
        )
    except (AuditFailure, ValueError, FloatingPointError, OSError) as error:
        print(f"Part 3 audit failed: {error}", file=sys.stderr)
        return 1
    print("Part 3 audit passed")
    for name, path in outputs.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
