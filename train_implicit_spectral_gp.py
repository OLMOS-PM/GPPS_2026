"""GPyTorch utilities for an implicit neural spectral RFF kernel.

An MLP maps Gaussian noise z in R^K to a frequency vector in R^D.  The
spectral density need not be evaluated: fresh pathwise-differentiable samples
are used to construct random Fourier features at every optimization step.
Rademacher sign flips explicitly symmetrize the samples, and an L2 frequency
penalty discourages spurious high-frequency mass.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import gpytorch
import numpy as np
import torch
from torch import nn


Tensor = torch.Tensor


class ImplicitFrequencyGenerator(nn.Module):
    """Dense neural sampler g_theta: R^K -> R^D."""

    def __init__(
        self,
        noise_dimension: int = 8,
        output_dimension: int = 1,
        hidden_features: Sequence[int] = (64, 64, 64),
    ) -> None:
        super().__init__()
        if noise_dimension <= 0 or output_dimension <= 0:
            raise ValueError("noise_dimension and output_dimension must be positive")
        if not hidden_features or any(width <= 0 for width in hidden_features):
            raise ValueError("hidden_features must contain positive widths")

        dimensions = [noise_dimension, *hidden_features, output_dimension]
        layers: list[nn.Module] = []
        for input_width, output_width in zip(dimensions[:-2], dimensions[1:-1]):
            layers.extend((nn.Linear(input_width, output_width), nn.SiLU()))
        layers.append(nn.Linear(dimensions[-2], dimensions[-1]))
        self.network = nn.Sequential(*layers)
        self.noise_dimension = noise_dimension
        self.output_dimension = output_dimension

    def forward(self, noise: Tensor) -> Tensor:
        return self.network(noise)


class SymmetrizedImplicitSpectralKernel(gpytorch.kernels.Kernel):
    """Stationary RFF kernel whose frequencies are emitted by a dense MLP."""

    is_stationary = True

    def __init__(
        self,
        *,
        input_dimension: int = 1,
        noise_dimension: int = 8,
        num_frequencies: int = 256,
        hidden_features: Sequence[int] = (64, 64, 64),
        initial_frequency_scale: float = 25.0,
        seed: int = 43,
    ) -> None:
        super().__init__()
        if input_dimension <= 0 or num_frequencies <= 0:
            raise ValueError("input_dimension and num_frequencies must be positive")
        if initial_frequency_scale <= 0:
            raise ValueError("initial_frequency_scale must be positive")

        self.input_dimension = input_dimension
        self.noise_dimension = noise_dimension
        self.num_frequencies = num_frequencies
        self.generator = ImplicitFrequencyGenerator(
            noise_dimension=noise_dimension,
            output_dimension=input_dimension,
            hidden_features=hidden_features,
        )
        self.raw_frequency_scale = nn.Parameter(
            torch.tensor(float(np.log(initial_frequency_scale)))
        )

        with torch.random.fork_rng():
            torch.manual_seed(seed)
            base_samples = torch.randn(num_frequencies, noise_dimension)
            signs = 2 * torch.randint(0, 2, (num_frequencies, 1)) - 1
        self.register_buffer("base_samples", base_samples)
        self.register_buffer("signs", signs.to(dtype=base_samples.dtype))

    @property
    def frequency_scale(self) -> Tensor:
        return torch.exp(self.raw_frequency_scale)

    def unsigned_frequencies(self, base_samples: Optional[Tensor] = None) -> Tensor:
        if base_samples is None:
            base_samples = self.base_samples
        return self.frequency_scale * self.generator(base_samples)

    def frequencies(self) -> Tensor:
        """Return the current sign-symmetrized frequency vectors [L, D]."""
        return self.signs * self.unsigned_frequencies()

    @torch.no_grad()
    def resample_training_frequencies(
        self,
        generator: Optional[torch.Generator] = None,
    ) -> None:
        """Draw fresh Gaussian inputs and Rademacher signs in place."""
        self.base_samples.copy_(
            torch.randn(
                self.base_samples.shape,
                dtype=self.base_samples.dtype,
                device=self.base_samples.device,
                generator=generator,
            )
        )
        fresh_signs = 2 * torch.randint(
            0,
            2,
            self.signs.shape,
            device=self.signs.device,
            generator=generator,
        ) - 1
        self.signs.copy_(fresh_signs.to(dtype=self.signs.dtype))

    def frequency_l2(self) -> Tensor:
        """Mean squared Euclidean norm of the sampled frequency vectors."""
        return self.frequencies().square().sum(dim=-1).mean()

    def _features(self, x: Tensor) -> Tensor:
        if x.shape[-1] != self.input_dimension:
            raise ValueError(
                f"expected inputs with final dimension {self.input_dimension}"
            )
        projection = x @ self.frequencies().transpose(-1, -2)
        return torch.cat((projection.cos(), projection.sin()), dim=-1) / np.sqrt(
            self.num_frequencies
        )

    def forward(
        self,
        x1: Tensor,
        x2: Tensor,
        diag: bool = False,
        **params: object,
    ) -> Tensor:
        features_1 = self._features(x1)
        features_2 = self._features(x2)
        if diag:
            return (features_1 * features_2).sum(dim=-1)
        return features_1 @ features_2.transpose(-1, -2)

    def draw_frequencies(self, num_samples: int, seed: int = 0) -> Tensor:
        """Draw fresh symmetrized frequency vectors for diagnostics."""
        if num_samples <= 0:
            raise ValueError("num_samples must be positive")
        parameter = next(self.parameters())
        random_generator = torch.Generator(device=parameter.device)
        random_generator.manual_seed(seed)
        base = torch.randn(
            num_samples,
            self.noise_dimension,
            generator=random_generator,
            dtype=parameter.dtype,
            device=parameter.device,
        )
        signs = 2 * torch.randint(
            0,
            2,
            (num_samples, 1),
            generator=random_generator,
            device=parameter.device,
        ) - 1
        return signs.to(dtype=parameter.dtype) * self.unsigned_frequencies(base)


class ImplicitSpectralGP(gpytorch.models.ExactGP):
    """Exact GP using a low-rank implicit spectral RFF covariance."""

    def __init__(
        self,
        train_x: Tensor,
        train_curves: Tensor,
        likelihood: gpytorch.likelihoods.GaussianLikelihood,
        kernel: SymmetrizedImplicitSpectralKernel,
    ) -> None:
        super().__init__(train_x, train_curves, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(kernel)

    def forward(self, x: Tensor) -> gpytorch.distributions.MultivariateNormal:
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x), self.covar_module(x)
        )


def as_tensors(
    x: np.ndarray,
    curves: np.ndarray,
    dtype: torch.dtype = torch.float32,
) -> tuple[Tensor, Tensor]:
    x_array = np.asarray(x)
    curves_array = np.asarray(curves)
    if x_array.ndim == 1:
        x_array = x_array[:, None]
    if x_array.ndim != 2:
        raise ValueError("x must have shape (N,) or (N, D)")
    if curves_array.ndim != 2 or curves_array.shape[1] != x_array.shape[0]:
        raise ValueError("curves must have shape (n_curves, N)")
    return torch.as_tensor(x_array, dtype=dtype), torch.as_tensor(
        curves_array, dtype=dtype
    )


def build_implicit_spectral_gp(
    train_x: Tensor,
    train_curves: Tensor,
    *,
    noise_dimension: int = 8,
    num_frequencies: int = 256,
    hidden_features: Sequence[int] = (64, 64, 64),
    initial_frequency_scale: float = 25.0,
    initial_outputscale: float = 1.0,
    initial_noise: float = 0.005,
    seed: int = 43,
) -> tuple[ImplicitSpectralGP, gpytorch.likelihoods.GaussianLikelihood]:
    if train_x.ndim != 2:
        raise ValueError("train_x must have shape (N, D)")
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    likelihood.initialize(noise=initial_noise)
    kernel = SymmetrizedImplicitSpectralKernel(
        input_dimension=train_x.shape[-1],
        noise_dimension=noise_dimension,
        num_frequencies=num_frequencies,
        hidden_features=hidden_features,
        initial_frequency_scale=initial_frequency_scale,
        seed=seed,
    )
    model = ImplicitSpectralGP(train_x, train_curves, likelihood, kernel)
    model.covar_module.initialize(outputscale=initial_outputscale)
    return model, likelihood


@dataclass(frozen=True)
class ImplicitTrainingHistory:
    objective: list[float]
    negative_mll: list[float]
    frequency_penalty: list[float]
    frequency_rms: list[float]


def train_implicit_spectral_gp(
    model: ImplicitSpectralGP,
    likelihood: gpytorch.likelihoods.GaussianLikelihood,
    train_x: Tensor,
    train_curves: Tensor,
    *,
    num_steps: int = 500,
    learning_rate: float = 0.003,
    frequency_penalty_weight: float = 1e-4,
    resampling_seed: int = 2027,
    print_every: int = 100,
) -> ImplicitTrainingHistory:
    """Train using fresh implicit spectral samples at every iteration."""
    if num_steps <= 0 or learning_rate <= 0:
        raise ValueError("num_steps and learning_rate must be positive")
    if frequency_penalty_weight < 0:
        raise ValueError("frequency_penalty_weight must be non-negative")

    model.train()
    likelihood.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    kernel = model.covar_module.base_kernel
    random_generator = torch.Generator(device=kernel.base_samples.device)
    random_generator.manual_seed(resampling_seed)
    objective_history: list[float] = []
    nll_history: list[float] = []
    penalty_history: list[float] = []
    rms_history: list[float] = []

    for step in range(1, num_steps + 1):
        kernel.resample_training_frequencies(random_generator)
        optimizer.zero_grad()
        negative_mll = -mll(model(train_x), train_curves).mean()
        frequency_penalty = kernel.frequency_l2()
        objective = negative_mll + frequency_penalty_weight * frequency_penalty
        objective.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()

        objective_history.append(float(objective.detach().cpu()))
        nll_history.append(float(negative_mll.detach().cpu()))
        penalty_history.append(float(frequency_penalty.detach().cpu()))
        rms_history.append(float(frequency_penalty.detach().sqrt().cpu()))
        if print_every > 0 and (step == 1 or step % print_every == 0):
            print(
                f"step {step:4d}/{num_steps} | objective={objective_history[-1]:.4f} "
                f"| NLL={nll_history[-1]:.4f} | freq RMS={rms_history[-1]:.3f} "
                f"| noise={float(likelihood.noise.detach().cpu()):.5f}"
            )

    return ImplicitTrainingHistory(
        objective=objective_history,
        negative_mll=nll_history,
        frequency_penalty=penalty_history,
        frequency_rms=rms_history,
    )


@dataclass(frozen=True)
class Prediction:
    mean: np.ndarray
    lower: np.ndarray
    upper: np.ndarray


def predict_sparse_signal(
    model: ImplicitSpectralGP,
    likelihood: gpytorch.likelihoods.GaussianLikelihood,
    observed_x: np.ndarray,
    observed_y: np.ndarray,
    prediction_x: np.ndarray,
) -> Prediction:
    parameter = next(model.parameters())
    observed_x_array = np.asarray(observed_x)
    prediction_x_array = np.asarray(prediction_x)
    if observed_x_array.ndim == 1:
        observed_x_array = observed_x_array[:, None]
    if prediction_x_array.ndim == 1:
        prediction_x_array = prediction_x_array[:, None]
    x_obs = torch.as_tensor(
        observed_x_array, dtype=parameter.dtype, device=parameter.device
    )
    y_obs = torch.as_tensor(observed_y, dtype=parameter.dtype, device=parameter.device)
    x_pred = torch.as_tensor(
        prediction_x_array, dtype=parameter.dtype, device=parameter.device
    )
    original_inputs = model.train_inputs
    original_targets = model.train_targets
    model.eval()
    likelihood.eval()
    try:
        model.set_train_data(inputs=x_obs, targets=y_obs, strict=False)
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            posterior = likelihood(model(x_pred))
            lower, upper = posterior.confidence_region()
    finally:
        model.set_train_data(
            inputs=original_inputs, targets=original_targets, strict=False
        )
    return Prediction(
        mean=posterior.mean.detach().cpu().numpy(),
        lower=lower.detach().cpu().numpy(),
        upper=upper.detach().cpu().numpy(),
    )
