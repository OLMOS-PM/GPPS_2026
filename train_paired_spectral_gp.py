"""Paired-frequency RFF kernels induced by Theorem 2 of the NG-SM paper.

The module provides differentiable Gaussian, Gaussian-mixture, and implicit
neural joint spectral samplers for frequency pairs (w1, w2).  Mixture
components are marginalized by concatenating feature blocks weighted by the
square root of their probabilities; no discrete component is sampled.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import gpytorch
import numpy as np
import torch
from gpytorch.constraints import GreaterThan
from torch import nn


Tensor = torch.Tensor


def _inverse_softplus(value: Tensor) -> Tensor:
    return torch.log(torch.expm1(value))


class PairedGaussianRFFKernel(gpytorch.kernels.Kernel):
    """Theorem-2 RFF kernel with a marginalized Gaussian mixture spectrum."""

    is_stationary = False

    def __init__(
        self,
        initial_means: Tensor,
        initial_stds: Tensor,
        *,
        num_pairs_per_component: int = 80,
        correlated: bool = True,
        initial_correlations: Tensor | float = 0.0,
        initial_weights: Tensor | None = None,
        seed: int = 101,
    ) -> None:
        super().__init__()
        means = torch.as_tensor(initial_means, dtype=torch.get_default_dtype())
        stds = torch.as_tensor(initial_stds, dtype=means.dtype)
        if means.ndim != 2 or means.shape[1] != 2:
            raise ValueError("initial_means must have shape (Q, 2)")
        if stds.shape != means.shape or torch.any(stds <= 0):
            raise ValueError("initial_stds must be positive with shape (Q, 2)")
        if num_pairs_per_component <= 0:
            raise ValueError("num_pairs_per_component must be positive")

        self.num_components = means.shape[0]
        self.num_pairs_per_component = num_pairs_per_component
        self.correlated = correlated
        self.means = nn.Parameter(means.clone())
        self.raw_stds = nn.Parameter(_inverse_softplus(stds.clone()))

        correlations = torch.as_tensor(initial_correlations, dtype=means.dtype)
        if correlations.ndim == 0:
            correlations = correlations.expand(self.num_components).clone()
        if correlations.shape != (self.num_components,) or torch.any(correlations.abs() >= 0.99):
            raise ValueError("initial_correlations must have shape (Q,) and magnitude < 0.99")
        raw_correlations = torch.atanh(correlations / 0.99)
        self.raw_correlations = nn.Parameter(
            raw_correlations, requires_grad=correlated
        )

        if initial_weights is None:
            weights = torch.full(
                (self.num_components,), 1.0 / self.num_components, dtype=means.dtype
            )
        else:
            weights = torch.as_tensor(initial_weights, dtype=means.dtype)
            if weights.shape != (self.num_components,) or torch.any(weights <= 0):
                raise ValueError("initial_weights must be positive with shape (Q,)")
            weights = weights / weights.sum()
        self.raw_weight_logits = nn.Parameter(
            weights.log(), requires_grad=self.num_components > 1
        )

        with torch.random.fork_rng():
            torch.manual_seed(seed)
            epsilon = torch.randn(
                self.num_components, num_pairs_per_component, 2, dtype=means.dtype
            )
        self.register_buffer("epsilon", epsilon)

    @property
    def stds(self) -> Tensor:
        return torch.nn.functional.softplus(self.raw_stds) + 1e-4

    @property
    def correlations(self) -> Tensor:
        if not self.correlated:
            return torch.zeros_like(self.raw_correlations)
        return 0.99 * torch.tanh(self.raw_correlations)

    @property
    def mixture_weights(self) -> Tensor:
        return torch.softmax(self.raw_weight_logits, dim=0)

    def frequency_pairs(self) -> tuple[Tensor, Tensor]:
        """Differentiable samples with the requested 2x2 covariance matrices."""
        eps_1, eps_2 = self.epsilon[..., 0], self.epsilon[..., 1]
        mu_1, mu_2 = self.means[:, 0, None], self.means[:, 1, None]
        std_1, std_2 = self.stds[:, 0, None], self.stds[:, 1, None]
        rho = self.correlations[:, None]
        w_1 = mu_1 + std_1 * eps_1
        w_2 = mu_2 + std_2 * (
            rho * eps_1 + torch.sqrt(torch.clamp(1.0 - rho.square(), min=1e-6)) * eps_2
        )
        return w_1, w_2

    def _features(self, x: Tensor) -> Tensor:
        if x.shape[-1] != 1:
            raise ValueError("PairedGaussianRFFKernel requires a one-dimensional input")
        w_1, w_2 = self.frequency_pairs()  # [Q, M]
        values = x.squeeze(-1)[..., :, None, None]
        projection_1 = values * w_1[None, :, :]
        projection_2 = values * w_2[None, :, :]
        scale = torch.sqrt(
            self.mixture_weights / (4.0 * self.num_pairs_per_component)
        )[None, :, None]
        cosine = scale * (projection_1.cos() + projection_2.cos())
        sine = scale * (projection_1.sin() + projection_2.sin())
        return torch.cat((cosine.flatten(start_dim=-2), sine.flatten(start_dim=-2)), dim=-1)

    def forward(self, x1: Tensor, x2: Tensor, diag: bool = False, **params: object) -> Tensor:
        features_1 = self._features(x1)
        features_2 = self._features(x2)
        if diag:
            return (features_1 * features_2).sum(dim=-1)
        return features_1 @ features_2.transpose(-1, -2)


class PairedImplicitRFFKernel(gpytorch.kernels.Kernel):
    """Theorem-2 kernel with an MLP sampler for the joint spectral density."""

    is_stationary = False

    def __init__(
        self,
        *,
        noise_dimension: int = 8,
        num_pairs: int = 240,
        hidden_features: Sequence[int] = (32, 32),
        initial_frequency_scale: float = 5.0,
        seed: int = 151,
    ) -> None:
        super().__init__()
        dimensions = [noise_dimension, *hidden_features, 2]
        layers: list[nn.Module] = []
        for width_in, width_out in zip(dimensions[:-2], dimensions[1:-1]):
            layers.extend((nn.Linear(width_in, width_out), nn.SiLU()))
        layers.append(nn.Linear(dimensions[-2], dimensions[-1]))
        self.generator = nn.Sequential(*layers)
        self.raw_frequency_scale = nn.Parameter(
            torch.tensor(float(np.log(initial_frequency_scale)))
        )
        self.num_pairs = num_pairs
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            base_samples = torch.randn(num_pairs, noise_dimension)
        self.register_buffer("base_samples", base_samples)

    @property
    def frequency_scale(self) -> Tensor:
        return self.raw_frequency_scale.exp()

    def frequency_pairs(self) -> tuple[Tensor, Tensor]:
        pairs = self.frequency_scale * self.generator(self.base_samples)
        return pairs[:, 0], pairs[:, 1]

    def frequency_l2(self) -> Tensor:
        w_1, w_2 = self.frequency_pairs()
        return 0.5 * (w_1.square().mean() + w_2.square().mean())

    def _features(self, x: Tensor) -> Tensor:
        if x.shape[-1] != 1:
            raise ValueError("PairedImplicitRFFKernel requires a one-dimensional input")
        w_1, w_2 = self.frequency_pairs()
        projection_1 = x * w_1[None, :]
        projection_2 = x * w_2[None, :]
        scale = 1.0 / np.sqrt(4.0 * self.num_pairs)
        return scale * torch.cat(
            (projection_1.cos() + projection_2.cos(), projection_1.sin() + projection_2.sin()),
            dim=-1,
        )

    def forward(self, x1: Tensor, x2: Tensor, diag: bool = False, **params: object) -> Tensor:
        features_1 = self._features(x1)
        features_2 = self._features(x2)
        if diag:
            return (features_1 * features_2).sum(dim=-1)
        return features_1 @ features_2.transpose(-1, -2)


class PairedSpectralGP(gpytorch.models.ExactGP):
    def __init__(self, train_x: Tensor, train_curves: Tensor, likelihood, kernel) -> None:
        super().__init__(train_x, train_curves, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(kernel)

    def forward(self, x: Tensor) -> gpytorch.distributions.MultivariateNormal:
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x), self.covar_module(x)
        )


def build_paired_spectral_gp(
    train_x: Tensor,
    train_curves: Tensor,
    kernel: gpytorch.kernels.Kernel,
    *,
    initial_outputscale: float = 1.0,
    initial_noise: float = 1e-3,
) -> tuple[PairedSpectralGP, gpytorch.likelihoods.GaussianLikelihood]:
    likelihood = gpytorch.likelihoods.GaussianLikelihood(
        noise_constraint=GreaterThan(1e-6)
    )
    likelihood.initialize(noise=initial_noise)
    model = PairedSpectralGP(train_x, train_curves, likelihood, kernel)
    model.covar_module.initialize(outputscale=initial_outputscale)
    return model.to(train_x.device), likelihood.to(train_x.device)


@dataclass(frozen=True)
class PairedTrainingHistory:
    loss: list[float]
    frequency_penalty: list[float]


def train_paired_spectral_gp(
    model: PairedSpectralGP,
    likelihood: gpytorch.likelihoods.GaussianLikelihood,
    train_x: Tensor,
    train_curves: Tensor,
    *,
    num_steps: int = 250,
    learning_rate: float = 0.01,
    frequency_penalty_weight: float = 0.0,
    print_every: int = 50,
) -> PairedTrainingHistory:
    model.train()
    likelihood.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    losses: list[float] = []
    penalties: list[float] = []
    kernel = model.covar_module.base_kernel

    for step in range(1, num_steps + 1):
        optimizer.zero_grad()
        negative_mll = -mll(model(train_x), train_curves).mean()
        penalty = (
            kernel.frequency_l2()
            if hasattr(kernel, "frequency_l2")
            else torch.zeros((), dtype=train_x.dtype, device=train_x.device)
        )
        objective = negative_mll + frequency_penalty_weight * penalty
        objective.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
        losses.append(float(negative_mll.detach().cpu()))
        penalties.append(float(penalty.detach().cpu()))
        if print_every and (step == 1 or step % print_every == 0):
            print(
                f"step {step:4d}/{num_steps} | NLL={losses[-1]:.4f} | "
                f"noise={float(likelihood.noise.detach().cpu()):.5f}"
            )
    return PairedTrainingHistory(loss=losses, frequency_penalty=penalties)


def mean_negative_log_likelihood(
    model: PairedSpectralGP,
    likelihood: gpytorch.likelihoods.GaussianLikelihood,
    x: Tensor,
    curves: Tensor,
) -> float:
    model.train()
    likelihood.train()
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    with torch.no_grad():
        value = -mll(model(x), curves).mean()
    return float(value.detach().cpu())
