"""Paired-feature amortized RFF-GPLVM models used by the Part 4 v2 notebook."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from train_amortized_gplvm import AmortizedGaussianEncoder


Tensor = torch.Tensor


def _paired_features(x: Tensor, frequencies: Tensor, scale: Tensor | float) -> Tensor:
    """Return paired cosine/sine features for one spectral component."""
    projection = x @ frequencies.T
    normalization = torch.as_tensor(scale, dtype=x.dtype, device=x.device).sqrt()
    return normalization * torch.cat((projection.cos(), projection.sin()), dim=-1)


class PairedSpectralPrior(nn.Module):
    """Interface for spectra that directly construct paired real RFFs."""

    def features(self, x: Tensor, num_frequencies: int) -> Tensor:
        raise NotImplementedError

    def summary(self) -> dict[str, np.ndarray | float]:
        raise NotImplementedError


class PairedRBFSpectralPrior(PairedSpectralPrior):
    """ARD Gaussian spectrum corresponding to an RBF kernel."""

    def __init__(self, latent_dim: int, initial_lengthscale: float = 1.0):
        super().__init__()
        initial = torch.full((latent_dim,), float(initial_lengthscale))
        self.raw_lengthscale = nn.Parameter(torch.log(torch.expm1(initial)))

    @property
    def lengthscale(self) -> Tensor:
        return F.softplus(self.raw_lengthscale) + 1e-4

    def features(self, x: Tensor, num_frequencies: int) -> Tensor:
        epsilon = torch.randn(num_frequencies, x.shape[-1], device=x.device)
        frequencies = epsilon / self.lengthscale
        return _paired_features(x, frequencies, 1.0 / num_frequencies)

    def summary(self) -> dict[str, np.ndarray]:
        return {"lengthscale": self.lengthscale.detach().cpu().numpy()}


class PairedLocallyPeriodicMixturePrior(PairedSpectralPrior):
    """Stratified RFF approximation to M locally periodic kernel components."""

    def __init__(self, latent_dim: int, num_components: int = 3):
        super().__init__()
        if num_components < 1:
            raise ValueError("num_components must be positive")
        means = torch.zeros(num_components, latent_dim)
        means[:, 0] = torch.linspace(0.5, 3.0, num_components)
        self.component_means = nn.Parameter(means)
        self.raw_scales = nn.Parameter(torch.full_like(means, -0.4))
        self.logits = nn.Parameter(torch.zeros(num_components))

    @property
    def scales(self) -> Tensor:
        return F.softplus(self.raw_scales) + 1e-4

    @property
    def weights(self) -> Tensor:
        return self.logits.softmax(dim=0)

    def features(self, x: Tensor, num_frequencies: int) -> Tensor:
        if num_frequencies < len(self.logits):
            raise ValueError("num_frequencies must be at least num_components")
        base, remainder = divmod(num_frequencies, len(self.logits))
        counts = [base + (q < remainder) for q in range(len(self.logits))]
        blocks = []
        for q, count in enumerate(counts):
            epsilon = torch.randn(count, x.shape[-1], device=x.device)
            frequencies = self.component_means[q] + self.scales[q] * epsilon
            blocks.append(_paired_features(x, frequencies, self.weights[q] / count))
        return torch.cat(blocks, dim=-1)

    def summary(self) -> dict[str, np.ndarray]:
        return {
            "weights": self.weights.detach().cpu().numpy(),
            "means": self.component_means.detach().cpu().numpy(),
            "scales": self.scales.detach().cpu().numpy(),
        }


class ImplicitNeuralSpectralPrior(PairedSpectralPrior):
    """Implicit frequency law obtained by pushing Gaussian noise through an MLP."""

    def __init__(
        self,
        latent_dim: int,
        noise_dim: int = 8,
        hidden_dims: tuple[int, ...] = (64, 64, 64),
    ):
        super().__init__()
        dimensions = [noise_dim, *hidden_dims, latent_dim]
        layers: list[nn.Module] = []
        for input_dim, output_dim in zip(dimensions[:-2], dimensions[1:-1]):
            layers.extend((nn.Linear(input_dim, output_dim), nn.SiLU()))
        layers.append(nn.Linear(dimensions[-2], dimensions[-1]))
        self.network = nn.Sequential(*layers)
        self.noise_dim = noise_dim
        self.raw_frequency_scale = nn.Parameter(torch.tensor(0.0))

    @property
    def frequency_scale(self) -> Tensor:
        return F.softplus(self.raw_frequency_scale) + 1e-4

    def features(self, x: Tensor, num_frequencies: int) -> Tensor:
        noise = torch.randn(num_frequencies, self.noise_dim, device=x.device)
        frequencies = self.frequency_scale * self.network(noise)
        return _paired_features(x, frequencies, 1.0 / num_frequencies)

    def summary(self) -> dict[str, float]:
        return {"frequency_scale": float(self.frequency_scale.detach().cpu())}


class PairedRFFGPLVM(nn.Module):
    """Amortized GPLVM with q(W)=p(W) and paired sine/cosine RFFs."""

    def __init__(
        self,
        observed_dim: int,
        latent_dim: int,
        num_frequencies: int,
        spectral_prior: PairedSpectralPrior,
    ):
        super().__init__()
        self.observed_dim = observed_dim
        self.latent_dim = latent_dim
        self.num_frequencies = num_frequencies
        self.encoder = AmortizedGaussianEncoder(observed_dim, latent_dim)
        self.spectral_prior = spectral_prior
        self.raw_outputscale = nn.Parameter(torch.tensor(0.0))
        self.raw_noise = nn.Parameter(torch.tensor(-2.0))

    @property
    def outputscale(self) -> Tensor:
        return F.softplus(self.raw_outputscale) + 1e-4

    @property
    def noise(self) -> Tensor:
        return F.softplus(self.raw_noise) + 1e-4

    def encode(self, y: Tensor) -> tuple[Tensor, Tensor]:
        return self.encoder(y)

    def rff_features(self, x: Tensor) -> Tensor:
        return self.outputscale.sqrt() * self.spectral_prior.features(
            x, self.num_frequencies
        )

    def gp_log_likelihood(self, y: Tensor, x: Tensor) -> Tensor:
        features = self.rff_features(x)
        noise = self.noise
        feature_dim = features.shape[1]
        identity = torch.eye(feature_dim, dtype=features.dtype, device=features.device)
        small_system = identity + features.T @ features / noise
        cholesky = torch.linalg.cholesky(small_system + 1e-5 * identity)
        feature_targets = features.T @ y
        solved = torch.cholesky_solve(feature_targets, cholesky)
        quadratic = y.square().sum() / noise
        quadratic -= (feature_targets * solved).sum() / noise.square()
        logdet = x.shape[0] * noise.log() + 2.0 * cholesky.diagonal().log().sum()
        normalizer = x.shape[0] * np.log(2.0 * np.pi)
        return -0.5 * (quadratic + self.observed_dim * (logdet + normalizer))

    @staticmethod
    def latent_kl(mean: Tensor, log_variance: Tensor) -> Tensor:
        return 0.5 * (
            mean.square() + log_variance.exp() - 1.0 - log_variance
        ).sum()

    def negative_elbo(self, y: Tensor, beta: float = 1.0) -> tuple[Tensor, dict]:
        mean, log_variance = self.encode(y)
        x = mean + torch.exp(0.5 * log_variance) * torch.randn_like(mean)
        expected_log_likelihood = self.gp_log_likelihood(y, x)
        kl_x = self.latent_kl(mean, log_variance)
        loss = -(expected_log_likelihood - beta * kl_x) / y.numel()
        return loss, {
            "loss": float(loss.detach()),
            "log_likelihood_per_pixel": float(expected_log_likelihood.detach() / y.numel()),
            "kl_x_per_example": float(kl_x.detach() / y.shape[0]),
            "kl_w": 0.0,
        }


@dataclass
class RFFPosterior:
    weight_mean: Tensor
    cholesky: Tensor
    training_features: Tensor
    feature_seed: int


def _seeded_features(model: PairedRFFGPLVM, x: Tensor, seed: int) -> Tensor:
    devices = [x.device] if x.is_cuda else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        return model.rff_features(x)


@torch.no_grad()
def fit_rff_posterior(
    model: PairedRFFGPLVM,
    centered_images: Tensor,
    training_latents: Tensor,
    *,
    feature_seed: int = 91,
) -> RFFPosterior:
    """Condition feature weights on images using one coherent draw of W."""
    training_features = _seeded_features(model, training_latents, feature_seed)
    feature_dim = training_features.shape[1]
    identity = torch.eye(
        feature_dim, dtype=training_features.dtype, device=training_features.device
    )
    cholesky = torch.linalg.cholesky(
        identity + training_features.T @ training_features / model.noise + 1e-5 * identity
    )
    weight_mean = torch.cholesky_solve(
        training_features.T @ centered_images / model.noise, cholesky
    )
    return RFFPosterior(weight_mean, cholesky, training_features, feature_seed)


@torch.no_grad()
def decode_posterior_mean(
    model: PairedRFFGPLVM, query_latents: Tensor, posterior: RFFPosterior
) -> Tensor:
    """Decode latent queries using the posterior mean of the feature weights."""
    return _seeded_features(model, query_latents, posterior.feature_seed) @ posterior.weight_mean


@torch.no_grad()
def decode_posterior_sample(
    model: PairedRFFGPLVM, query_latents: Tensor, posterior: RFFPosterior
) -> Tensor:
    """Draw coherent functions from the feature-weight posterior."""
    epsilon = torch.randn_like(posterior.weight_mean)
    weight_sample = posterior.weight_mean + torch.linalg.solve_triangular(
        posterior.cholesky.T, epsilon, upper=True
    )
    return _seeded_features(model, query_latents, posterior.feature_seed) @ weight_sample
