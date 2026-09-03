"""Generate a categorical mixture of stationary Gaussian-process curves.

Each mixture component uses the locally periodic kernel

    k_m(tau) = variance * exp(-tau**2 / (2 * lengthscale_m**2))
               * cos(frequency_m * tau),

with angular frequencies 2*pi/T_m.  The default periods are 1, 0.5, and
0.25, matching the synthetic example in the accompanying spectral-kernel
note.

For every curve i, the generator first draws a component label

    z_i ~ Categorical(mixture_weights),

then performs an exact finite-dimensional GP draw on the requested grid:

    K_m[j, k] = k_m(x_j - x_k),
    K_m + jitter * I = L_m L_m.T,
    f_i = L_m xi_i,                 xi_i ~ N(0, I),
    y_i = f_i + epsilon_i,          epsilon_i ~ N(0, noise_std**2 I).

The Cholesky factor is reused by all curves assigned to a component.  Random
Fourier features are not used to generate this dataset; they are only an
approximation used by one of the downstream fitted models.

Example
-------
python generate_gp_mixture.py --n-curves 300 --output gp_mixture.npz
"""

from __future__ import annotations # This package is used for forward references in type hints, allowing the use of types that are defined later in the code.

import argparse
from pathlib import Path

import numpy as np
from numpy.typing import NDArray #NDArray is a type hint for numpy arrays, allowing for more precise type checking and code clarity.


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def locally_periodic_kernel(
    x: FloatArray,
    frequency: float,
    lengthscale: float,
    variance: float = 1.0,
) -> FloatArray:
    """Return the covariance matrix of a stationary locally periodic kernel."""
    if lengthscale <= 0:
        raise ValueError("lengthscale must be positive")
    if variance <= 0:
        raise ValueError("variance must be positive")

    tau = x[:, None] - x[None, :]
    envelope = np.exp(-(tau**2) / (2.0 * lengthscale**2))
    return variance * envelope * np.cos(frequency * tau)


def generate_gp_mixture(
    x: FloatArray,
    n_curves: int,
    periods: FloatArray,
    lengthscales: FloatArray,
    mixture_weights: FloatArray,
    variance: float = 1.0,
    noise_std: float = 0.05,
    jitter: float = 1e-8,
    seed: int = 7,
) -> tuple[FloatArray, IntArray]:
    """Sample curves and component labels from a categorical GP mixture.

    Returns
    -------
    curves:
        Array with shape ``(n_curves, n_points)``.
    labels:
        Integer component assignments with shape ``(n_curves,)``.
    """
    periods = np.asarray(periods, dtype=np.float64)
    lengthscales = np.asarray(lengthscales, dtype=np.float64)
    mixture_weights = np.asarray(mixture_weights, dtype=np.float64)

    if x.ndim != 1 or x.size < 2:
        raise ValueError("x must be a one-dimensional array with at least two points")
    if n_curves <= 0:
        raise ValueError("n_curves must be positive")
    if periods.ndim != 1 or np.any(periods <= 0):
        raise ValueError("periods must be a one-dimensional array of positive values")
    if lengthscales.shape != periods.shape or np.any(lengthscales <= 0):
        raise ValueError("provide one positive lengthscale per period")
    if mixture_weights.shape != periods.shape or np.any(mixture_weights < 0):
        raise ValueError("provide one non-negative mixture weight per period")
    if not np.isclose(mixture_weights.sum(), 1.0):
        raise ValueError("mixture weights must sum to one")
    if noise_std < 0 or jitter <= 0:
        raise ValueError("noise_std must be non-negative and jitter must be positive")

    rng = np.random.default_rng(seed)
    labels = rng.choice(periods.size, size=n_curves, p=mixture_weights)
    frequencies = 2.0 * np.pi / periods
    curves = np.empty((n_curves, x.size), dtype=np.float64)

    # Cholesky factorize each component covariance once, then sample all curves assigned
    # to that component in a single matrix multiplication.
    for component, (frequency, lengthscale) in enumerate(
        zip(frequencies, lengthscales)
    ):
        # We first check if any curves are assigned to this component.  If not, we skip the Cholesky factorization.
        indices = np.flatnonzero(labels == component)
        if indices.size == 0:
            continue

        covariance = locally_periodic_kernel(
            x=x,
            frequency=float(frequency),
            lengthscale=float(lengthscale),
            variance=variance,
        )
        cholesky = np.linalg.cholesky(covariance + jitter * np.eye(x.size))
        standard_normal = rng.standard_normal((indices.size, x.size))
        curves[indices] = standard_normal @ cholesky.T

    if noise_std > 0:
        curves += noise_std * rng.standard_normal(curves.shape)

    return curves, labels.astype(np.int64, copy=False)


def _positive_floats(values: list[str], name: str) -> FloatArray:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 1 or result.size == 0 or np.any(result <= 0):
        raise ValueError(f"{name} must contain positive numbers")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__) #__doc__ is a special variable in Python that contains the docstring of the module, class, or function. In this case, it provides a description for the command-line interface of the script.
    parser.add_argument("--output", type=Path, default=Path("gp_mixture.npz"))
    parser.add_argument("--n-curves", type=int, default=300)
    parser.add_argument("--n-points", type=int, default=200)
    parser.add_argument("--x-min", type=float, default=0.0)
    parser.add_argument("--x-max", type=float, default=2.0)
    parser.add_argument(
        "--periods", nargs="+", default=["1.0", "0.5", "0.25"],
        help="component periods (default: 1.0 0.5 0.25)",
    )
    parser.add_argument(
        "--lengthscales", nargs="+", default=["0.6", "0.6", "0.6"],
        help="one envelope lengthscale per component",
    )
    parser.add_argument(
        "--weights", nargs="+", default=None,
        help="categorical weights; defaults to a uniform mixture",
    )
    parser.add_argument("--variance", type=float, default=1.0)
    parser.add_argument(
        "--noise-std",
        type=float,
        default=0.05,
        help="standard deviation of independent observation noise (default: 0.05)",
    )
    parser.add_argument("--jitter", type=float, default=1e-8)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    periods = _positive_floats(args.periods, "periods")
    lengthscales = _positive_floats(args.lengthscales, "lengthscales")
    if lengthscales.shape != periods.shape:
        raise ValueError("--lengthscales must have the same length as --periods")

    if args.weights is None:
        weights = np.full(periods.size, 1.0 / periods.size)
    else:
        weights = np.asarray(args.weights, dtype=np.float64)
        if weights.shape != periods.shape or np.any(weights < 0):
            raise ValueError("--weights must provide one non-negative value per period")
        if weights.sum() <= 0:
            raise ValueError("at least one mixture weight must be positive")
        weights = weights / weights.sum()

    if args.n_points < 2 or args.x_max <= args.x_min:
        raise ValueError("use at least two points and require x-max > x-min")

    x = np.linspace(args.x_min, args.x_max, args.n_points, dtype=np.float64)
    curves, labels = generate_gp_mixture(
        x=x,
        n_curves=args.n_curves,
        periods=periods,
        lengthscales=lengthscales,
        mixture_weights=weights,
        variance=args.variance,
        noise_std=args.noise_std,
        jitter=args.jitter,
        seed=args.seed,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    frequencies = 2.0 * np.pi / periods
    np.savez_compressed(
        args.output,
        x=x,
        curves=curves,
        labels=labels,
        periods=periods,
        frequencies=frequencies,
        lengthscales=lengthscales,
        mixture_weights=weights,
        variance=np.asarray(args.variance),
        noise_std=np.asarray(args.noise_std),
        seed=np.asarray(args.seed),
    )

    counts = np.bincount(labels, minlength=periods.size)
    print(f"Saved {curves.shape[0]} curves x {curves.shape[1]} points to {args.output}")
    for component, (period, frequency, count) in enumerate(
        zip(periods, frequencies, counts)
    ):
        print(
            f"  component {component}: period={period:g}, "
            f"frequency={frequency:.6g}, curves={count}"
        )


if __name__ == "__main__":
    main()
