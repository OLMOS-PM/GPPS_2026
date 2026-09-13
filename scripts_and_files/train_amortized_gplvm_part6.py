"""Non-stationary paired-spectral priors for the Part 6 amortized RFF-GPLVM."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .train_amortized_gplvm_v2 import PairedSpectralPrior

Tensor = torch.Tensor


def _joint_spectral_features(
    x: Tensor,
    frequencies_1: Tensor,
    frequencies_2: Tensor,
    scale: Tensor | float,
) -> Tensor:
    """Theorem-2 real features for paired frequencies (w1, w2)."""
    projection_1 = x @ frequencies_1.T
    projection_2 = x @ frequencies_2.T
    normalization = torch.as_tensor(
        scale / 4.0, dtype=x.dtype, device=x.device
    ).sqrt()
    return normalization * torch.cat(
        (
            projection_1.cos() + projection_2.cos(),
            projection_1.sin() + projection_2.sin(),
        ),
        dim=-1,
    )


class CorrelatedGaussianJointSpectralPrior(PairedSpectralPrior):
    """One joint Gaussian over two Q-dimensional frequency vectors."""

    def __init__(
        self,
        latent_dim: int,
        *,
        initial_scale: float = 1.0,
        initial_correlation: float = 0.7,
    ) -> None:
        super().__init__()
        if not -0.99 < initial_correlation < 0.99:
            raise ValueError("initial_correlation must have magnitude below 0.99")
        self.means = nn.Parameter(torch.zeros(2, latent_dim))
        initial = torch.full((2, latent_dim), float(initial_scale))
        self.raw_scales = nn.Parameter(torch.log(torch.expm1(initial)))
        correlation = torch.full((latent_dim,), float(initial_correlation))
        self.raw_correlations = nn.Parameter(torch.atanh(correlation / 0.99))

    @property
    def scales(self) -> Tensor:
        return F.softplus(self.raw_scales) + 1e-4

    @property
    def correlations(self) -> Tensor:
        return 0.99 * torch.tanh(self.raw_correlations)

    def sample_frequency_pairs(
        self, num_frequencies: int, device: torch.device
    ) -> tuple[Tensor, Tensor]:
        epsilon_1 = torch.randn(num_frequencies, self.means.shape[-1], device=device)
        epsilon_2 = torch.randn_like(epsilon_1)
        rho = self.correlations
        frequency_1 = self.means[0] + self.scales[0] * epsilon_1
        frequency_2 = self.means[1] + self.scales[1] * (
            rho * epsilon_1
            + torch.sqrt(torch.clamp(1.0 - rho.square(), min=1e-6)) * epsilon_2
        )
        return frequency_1, frequency_2

    def features(self, x: Tensor, num_frequencies: int) -> Tensor:
        frequency_1, frequency_2 = self.sample_frequency_pairs(
            num_frequencies, x.device
        )
        return _joint_spectral_features(
            x, frequency_1, frequency_2, 1.0 / num_frequencies
        )

    def summary(self) -> dict[str, np.ndarray]:
        return {
            "means": self.means.detach().cpu().numpy(),
            "scales": self.scales.detach().cpu().numpy(),
            "correlations": self.correlations.detach().cpu().numpy(),
        }


class CorrelatedGaussianMixtureJointSpectralPrior(PairedSpectralPrior):
    """Marginalized mixture of joint Gaussian frequency-pair distributions."""

    def __init__(self, latent_dim: int, num_components: int = 3) -> None:
        super().__init__()
        if num_components < 1:
            raise ValueError("num_components must be positive")
        means = torch.zeros(num_components, 2, latent_dim)
        initial_modes = torch.linspace(0.5, 3.0, num_components)
        means[:, 0, 0] = initial_modes
        means[:, 1, 0] = initial_modes
        self.means = nn.Parameter(means)
        self.raw_scales = nn.Parameter(torch.full_like(means, -0.4))
        initial_rho = torch.full((num_components, latent_dim), 0.7)
        self.raw_correlations = nn.Parameter(torch.atanh(initial_rho / 0.99))
        self.logits = nn.Parameter(torch.zeros(num_components))

    @property
    def scales(self) -> Tensor:
        return F.softplus(self.raw_scales) + 1e-4

    @property
    def correlations(self) -> Tensor:
        return 0.99 * torch.tanh(self.raw_correlations)

    @property
    def weights(self) -> Tensor:
        return self.logits.softmax(dim=0)

    def _component_pairs(
        self, component: int, count: int, device: torch.device
    ) -> tuple[Tensor, Tensor]:
        epsilon_1 = torch.randn(count, self.means.shape[-1], device=device)
        epsilon_2 = torch.randn_like(epsilon_1)
        rho = self.correlations[component]
        frequency_1 = (
            self.means[component, 0]
            + self.scales[component, 0] * epsilon_1
        )
        frequency_2 = self.means[component, 1] + self.scales[component, 1] * (
            rho * epsilon_1
            + torch.sqrt(torch.clamp(1.0 - rho.square(), min=1e-6)) * epsilon_2
        )
        return frequency_1, frequency_2

    def features(self, x: Tensor, num_frequencies: int) -> Tensor:
        num_components = len(self.logits)
        if num_frequencies < num_components:
            raise ValueError("num_frequencies must be at least num_components")
        base, remainder = divmod(num_frequencies, num_components)
        counts = [base + (component < remainder) for component in range(num_components)]
        blocks = []
        for component, count in enumerate(counts):
            frequency_1, frequency_2 = self._component_pairs(
                component, count, x.device
            )
            blocks.append(
                _joint_spectral_features(
                    x,
                    frequency_1,
                    frequency_2,
                    self.weights[component] / count,
                )
            )
        return torch.cat(blocks, dim=-1)

    def summary(self) -> dict[str, np.ndarray]:
        return {
            "weights": self.weights.detach().cpu().numpy(),
            "means": self.means.detach().cpu().numpy(),
            "scales": self.scales.detach().cpu().numpy(),
            "correlations": self.correlations.detach().cpu().numpy(),
        }


class ImplicitJointSpectralPrior(PairedSpectralPrior):
    """An MLP pushforward distribution over (w1, w2) in R^(2Q)."""

    def __init__(
        self,
        latent_dim: int,
        noise_dim: int = 8,
        hidden_dims: tuple[int, ...] = (64, 64, 64, 64),
        initial_frequency_scale: float = 1.0,
    ) -> None:
        super().__init__()
        if initial_frequency_scale <= 0:
            raise ValueError("initial_frequency_scale must be positive")
        dimensions = [noise_dim, *hidden_dims, 2 * latent_dim]
        layers: list[nn.Module] = []
        for input_dim, output_dim in zip(dimensions[:-2], dimensions[1:-1]):
            layers.extend((nn.Linear(input_dim, output_dim), nn.SiLU()))
        layers.append(nn.Linear(dimensions[-2], dimensions[-1]))
        self.network = nn.Sequential(*layers)
        self.latent_dim = latent_dim
        self.noise_dim = noise_dim
        self.raw_frequency_scale = nn.Parameter(
            torch.tensor(float(np.log(initial_frequency_scale)))
        )

    @property
    def frequency_scale(self) -> Tensor:
        return self.raw_frequency_scale.exp()

    def sample_frequency_pairs(
        self, num_frequencies: int, device: torch.device
    ) -> tuple[Tensor, Tensor]:
        noise = torch.randn(num_frequencies, self.noise_dim, device=device)
        pairs = self.frequency_scale * self.network(noise)
        return pairs[:, : self.latent_dim], pairs[:, self.latent_dim :]

    def features(self, x: Tensor, num_frequencies: int) -> Tensor:
        frequency_1, frequency_2 = self.sample_frequency_pairs(
            num_frequencies, x.device
        )
        return _joint_spectral_features(
            x, frequency_1, frequency_2, 1.0 / num_frequencies
        )

    def summary(self) -> dict[str, float]:
        return {"frequency_scale": float(self.frequency_scale.detach().cpu())}
