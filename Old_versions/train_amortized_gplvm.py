"""Amortized variational GPLVM utilities for the Part 5 MNIST notebook.

The implementation keeps the GP likelihood exact for a modest MNIST subset.
Random Fourier frequencies are reparameterized samples from the kernel's
spectral prior; their variational posterior is deliberately tied to that prior.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isqrt

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


Tensor = torch.Tensor


class AmortizedGaussianEncoder(nn.Module):
    """Diagonal Gaussian q_phi(x_n | y_n) shared by all observations."""

    def __init__(self, observed_dim: int, latent_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(observed_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.mean_head = nn.Linear(hidden_dim, latent_dim)
        self.log_variance_head = nn.Linear(hidden_dim, latent_dim)

    def forward(self, y: Tensor) -> tuple[Tensor, Tensor]:
        hidden = self.backbone(y)
        mean = self.mean_head(hidden)
        log_variance = self.log_variance_head(hidden).clamp(-8.0, 5.0)
        return mean, log_variance


class AmortizedCNNEncoder(nn.Module):
    """Small convolutional diagonal-Gaussian q_phi(x_n | y_n) for square images."""

    def __init__(self, observed_dim: int, latent_dim: int):
        super().__init__()
        image_side = isqrt(observed_dim)
        if image_side * image_side != observed_dim:
            raise ValueError("the CNN encoder requires square image observations")
        self.image_side = image_side
        first_conv_side = (image_side + 1) // 2
        second_conv_side = (first_conv_side + 1) // 2
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * second_conv_side * second_conv_side, 64), nn.ReLU(),
        )
        self.mean_head = nn.Linear(64, latent_dim)
        self.log_variance_head = nn.Linear(64, latent_dim)

    def forward(self, y: Tensor) -> tuple[Tensor, Tensor]:
        features = self.backbone(y.reshape(-1, 1, self.image_side, self.image_side))
        mean = self.mean_head(features)
        log_variance = self.log_variance_head(features).clamp(-8.0, 5.0)
        return mean, log_variance


class SpectralPrior(nn.Module):
    """Base class for reparameterized stationary spectral priors."""

    def sample(self, num_frequencies: int, latent_dim: int) -> Tensor:
        raise NotImplementedError

    def summary(self) -> dict[str, np.ndarray]:
        raise NotImplementedError

    def kernel_matrix(self, x_left: Tensor, x_right: Tensor) -> Tensor:
        """Exact stationary kernel implied by the spectral distribution."""
        raise NotImplementedError


class RBFSpectralPrior(SpectralPrior):
    """p(w)=N(0, diag(lengthscale^-2)) for an ARD RBF kernel."""

    def __init__(self, latent_dim: int, initial_lengthscale: float = 1.0):
        super().__init__()
        initial = torch.full((latent_dim,), float(initial_lengthscale))
        self.raw_lengthscale = nn.Parameter(torch.log(torch.expm1(initial)))

    @property
    def lengthscale(self) -> Tensor:
        return F.softplus(self.raw_lengthscale) + 1e-4

    def sample(self, num_frequencies: int, latent_dim: int) -> Tensor:
        if latent_dim != self.lengthscale.numel():
            raise ValueError("latent_dim does not match the RBF prior")
        epsilon = torch.randn(
            num_frequencies, latent_dim, device=self.lengthscale.device
        )
        return epsilon / self.lengthscale

    def summary(self) -> dict[str, np.ndarray]:
        return {"lengthscale": self.lengthscale.detach().cpu().numpy()}

    def kernel_matrix(self, x_left: Tensor, x_right: Tensor) -> Tensor:
        delta = x_left[:, None, :] - x_right[None, :, :]
        squared_scaled_distance = (delta / self.lengthscale).square().sum(dim=-1)
        return torch.exp(-0.5 * squared_scaled_distance)


class GaussianMixtureSpectralPrior(SpectralPrior):
    """Learnable symmetric mixture of diagonal Gaussian pairs.

    Each component is 0.5 N(+mu_q, diag(s_q^2)) +
    0.5 N(-mu_q, diag(s_q^2)).  The random sign makes the spectral law
    symmetric and therefore produces a real stationary kernel.
    """

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

    def sample(self, num_frequencies: int, latent_dim: int) -> Tensor:
        if latent_dim != self.component_means.shape[1]:
            raise ValueError("latent_dim does not match the mixture prior")
        # Straight-through Gumbel soft assignments let the mixture weights train.
        assignment = F.gumbel_softmax(
            self.logits.expand(num_frequencies, -1), tau=0.5, hard=True
        )
        means = assignment @ self.component_means
        scales = assignment @ self.scales
        signs = torch.where(
            torch.rand(num_frequencies, 1, device=means.device) < 0.5,
            -torch.ones(1, device=means.device),
            torch.ones(1, device=means.device),
        )
        return signs * means + scales * torch.randn_like(means)

    def summary(self) -> dict[str, np.ndarray]:
        return {
            "weights": self.weights.detach().cpu().numpy(),
            "means": self.component_means.detach().cpu().numpy(),
            "scales": self.scales.detach().cpu().numpy(),
        }

    def kernel_matrix(self, x_left: Tensor, x_right: Tensor) -> Tensor:
        delta = x_left[:, None, :] - x_right[None, :, :]
        scaled_squared_distance = (
            delta.square().unsqueeze(-2) * self.scales.square()[None, None, :, :]
        ).sum(dim=-1)
        phase = delta @ self.component_means.T
        component_kernels = torch.cos(phase) * torch.exp(-0.5 * scaled_squared_distance)
        return (component_kernels * self.weights[None, None, :]).sum(dim=-1)


# Backward-compatible name for existing notebooks using the original three-component default.
ThreeGaussianSpectralPrior = GaussianMixtureSpectralPrior


class VariationalGaussianSpectralPosterior(SpectralPrior):
    """Per-frequency diagonal Gaussian q(W), with a fixed Gaussian prior.

    Unlike the tied spectral families above, this module stores variational
    means and variances for every one of the L frequency vectors and therefore
    contributes KL[q(W)||p(W)] to the ELBO.
    """

    def __init__(self, num_frequencies: int, prior_scale: Tensor):
        super().__init__()
        prior_scale = torch.as_tensor(prior_scale).detach().clone()
        if prior_scale.ndim != 1 or torch.any(prior_scale <= 0):
            raise ValueError("prior_scale must be a positive vector")
        self.num_frequencies = num_frequencies
        self.register_buffer("prior_scale", prior_scale)
        self.means = nn.Parameter(torch.zeros(num_frequencies, prior_scale.numel()))
        initial_raw_scale = torch.log(torch.expm1(prior_scale))
        self.raw_scales = nn.Parameter(initial_raw_scale.expand_as(self.means).clone())

    @property
    def scales(self) -> Tensor:
        return F.softplus(self.raw_scales) + 1e-4

    def sample(self, num_frequencies: int, latent_dim: int) -> Tensor:
        if num_frequencies != self.num_frequencies or latent_dim != self.means.shape[1]:
            raise ValueError("requested frequency shape does not match q(W)")
        return self.means + self.scales * torch.randn_like(self.means)

    def kl_to_prior(self) -> Tensor:
        variance_ratio = self.scales.square() / self.prior_scale.square()
        squared_mean = self.means.square() / self.prior_scale.square()
        return 0.5 * (
            variance_ratio + squared_mean - 1.0
            + 2.0 * (self.prior_scale.log() - self.scales.log())
        ).sum()

    def summary(self) -> dict[str, np.ndarray]:
        return {
            "mean_norm": np.asarray(float(self.means.detach().norm().cpu())),
            "average_scale": np.asarray(float(self.scales.detach().mean().cpu())),
            "prior_average_scale": np.asarray(float(self.prior_scale.mean().cpu())),
        }


class AmortizedRFFGPLVM(nn.Module):
    """Bayesian GPLVM with an amortized q(X) and configurable q(W)."""

    def __init__(
        self,
        observed_dim: int,
        latent_dim: int = 2,
        num_frequencies: int = 200,
        spectral_prior: SpectralPrior | None = None,
        encoder_type: str = "mlp",
    ):
        super().__init__()
        self.observed_dim = observed_dim
        self.latent_dim = latent_dim
        self.num_frequencies = num_frequencies
        if encoder_type == "mlp":
            self.encoder = AmortizedGaussianEncoder(observed_dim, latent_dim)
        elif encoder_type == "cnn":
            self.encoder = AmortizedCNNEncoder(observed_dim, latent_dim)
        else:
            raise ValueError("encoder_type must be 'mlp' or 'cnn'")
        self.spectral_prior = spectral_prior or RBFSpectralPrior(latent_dim)
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

    def sample_latents(self, mean: Tensor, log_variance: Tensor) -> Tensor:
        return mean + torch.exp(0.5 * log_variance) * torch.randn_like(mean)

    def sample_feature_parameters(self) -> tuple[Tensor, Tensor]:
        """Draw one coherent set of frequencies and phases from q(W)=p(W)."""
        frequencies = self.spectral_prior.sample(
            self.num_frequencies, self.latent_dim
        )
        phases = 2.0 * torch.pi * torch.rand(
            self.num_frequencies, device=frequencies.device
        )
        return frequencies, phases

    def rff_features(
        self,
        x: Tensor,
        feature_parameters: tuple[Tensor, Tensor] | None = None,
    ) -> Tensor:
        """Return the scaled N x L feature matrix without forming an N x N kernel."""
        frequencies, phases = (
            feature_parameters
            if feature_parameters is not None
            else self.sample_feature_parameters()
        )
        features = (2.0 / self.num_frequencies) ** 0.5 * torch.cos(
            x @ frequencies.T + phases
        )
        return self.outputscale.sqrt() * features

    def gp_log_likelihood(self, y: Tensor, x: Tensor) -> Tensor:
        """Exact RFF marginal likelihood using only L x L linear algebra.

        For C = F F^T + noise I, the determinant lemma and Woodbury identity
        reduce both log|C| and Y^T C^-1 Y to F^T F and F^T Y.  The N x N
        covariance is never materialized.
        """
        features = self.rff_features(x)  # N x L
        noise = self.noise
        feature_gram = features.T @ features
        small_system = torch.eye(
            self.num_frequencies, dtype=features.dtype, device=features.device
        ) + feature_gram / noise
        cholesky = torch.linalg.cholesky(
            small_system + 1e-5 * torch.eye(
                self.num_frequencies, dtype=features.dtype, device=features.device
            )
        )
        feature_targets = features.T @ y  # L x D
        solved = torch.cholesky_solve(feature_targets, cholesky)
        quadratic = y.square().sum() / noise
        quadratic = quadratic - (feature_targets * solved).sum() / noise.square()
        logdet = x.shape[0] * noise.log() + 2.0 * cholesky.diagonal().log().sum()
        normalizer = x.shape[0] * np.log(2.0 * np.pi)
        return -0.5 * (quadratic + self.observed_dim * (logdet + normalizer))

    @staticmethod
    def latent_kl(mean: Tensor, log_variance: Tensor) -> Tensor:
        """KL[q_phi(X|Y) || N(0,I)], summed over examples and dimensions."""
        return 0.5 * (
            mean.square() + log_variance.exp() - 1.0 - log_variance
        ).sum()

    def negative_elbo(self, y: Tensor, beta: float = 1.0) -> tuple[Tensor, dict]:
        mean, log_variance = self.encode(y)
        x = self.sample_latents(mean, log_variance)
        expected_log_likelihood = self.gp_log_likelihood(y, x)
        kl_x = self.latent_kl(mean, log_variance)
        kl_w = (
            self.spectral_prior.kl_to_prior()
            if hasattr(self.spectral_prior, "kl_to_prior")
            else torch.zeros((), device=y.device)
        )
        loss = -(expected_log_likelihood - beta * kl_x - kl_w) / y.numel()
        stats = {
            "loss": float(loss.detach()),
            "log_likelihood_per_pixel": float(expected_log_likelihood.detach() / y.numel()),
            "kl_x_per_example": float(kl_x.detach() / y.shape[0]),
            "kl_w": float(kl_w.detach()),
        }
        return loss, stats


@dataclass
class TrainingHistory:
    loss: list[float]
    log_likelihood_per_pixel: list[float]
    kl_x_per_example: list[float]
    kl_w: list[float]


def train_model(
    model: AmortizedRFFGPLVM,
    images: Tensor,
    *,
    epochs: int = 1000,
    learning_rate: float = 2e-3,
    beta_warmup_epochs: int = 200,
    print_every: int = 100,
) -> TrainingHistory:
    """Optimize a one-sample Monte Carlo estimate of the variational ELBO."""
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history = TrainingHistory([], [], [], [])
    for epoch in range(1, epochs + 1):
        beta = min(1.0, epoch / max(1, beta_warmup_epochs))
        optimizer.zero_grad(set_to_none=True)
        loss, stats = model.negative_elbo(images, beta=beta)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
        for key in ("loss", "log_likelihood_per_pixel", "kl_x_per_example", "kl_w"):
            getattr(history, key).append(stats[key])
        if epoch == 1 or epoch % print_every == 0 or epoch == epochs:
            print(
                f"epoch {epoch:4d} | loss {stats['loss']:.4f} | "
                f"log p/pixel {stats['log_likelihood_per_pixel']:.4f} | "
                f"KL(X)/example {stats['kl_x_per_example']:.3f} | "
                f"KL(W) {stats['kl_w']:.3f}"
            )
    return history


@torch.no_grad()
def latent_means(
    model: AmortizedRFFGPLVM, images: Tensor, batch_size: int = 2048
) -> np.ndarray:
    """Encode a large dataset in bounded-memory chunks (not ELBO mini-batches)."""
    model.eval()
    means = []
    for start in range(0, images.shape[0], batch_size):
        mean, _ = model.encode(images[start : start + batch_size])
        means.append(mean.cpu())
    return torch.cat(means).numpy()


@torch.no_grad()
def posterior_mean_reconstructions(
    model: AmortizedRFFGPLVM,
    centered_training_images: Tensor,
    training_latents: Tensor,
    query_latents: Tensor,
) -> Tensor:
    """Exact-kernel GP posterior means without sampling Fourier frequencies.

    This is intended for reconstruction displays at inferred latents. It uses
    the analytic kernel corresponding to a tied spectral prior, rather than a
    new finite random-feature draw.
    """
    model.eval()
    kernel = model.outputscale * model.spectral_prior.kernel_matrix(
        training_latents, training_latents
    )
    covariance = kernel + model.noise * torch.eye(
        training_latents.shape[0], dtype=kernel.dtype, device=kernel.device
    )
    cholesky = torch.linalg.cholesky(
        covariance + 1e-5 * torch.eye(
            training_latents.shape[0], dtype=kernel.dtype, device=kernel.device
        )
    )
    alpha = torch.cholesky_solve(centered_training_images, cholesky)
    cross_kernel = model.outputscale * model.spectral_prior.kernel_matrix(
        query_latents, training_latents
    )
    return cross_kernel @ alpha


@torch.no_grad()
def posterior_mean_images(
    model: AmortizedRFFGPLVM,
    centered_training_images: Tensor,
    training_latents: Tensor,
    query_latents: Tensor,
    *,
    seed: int = 91,
) -> Tensor:
    """Return conditional posterior-mean images using Woodbury feature algebra.

    A single draw of W and the phases is shared by training and query features.
    The returned values remain centered; add the training pixel mean before
    displaying them.
    """
    model.eval()
    devices = [] if query_latents.device.type == "cpu" else [query_latents.device]
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        feature_parameters = model.sample_feature_parameters()
        train_features = model.rff_features(training_latents, feature_parameters)
        query_features = model.rff_features(query_latents, feature_parameters)
        noise = model.noise
        small_system = torch.eye(
            model.num_frequencies,
            dtype=train_features.dtype,
            device=train_features.device,
        ) + train_features.T @ train_features / noise
        cholesky = torch.linalg.cholesky(
            small_system
            + 1e-5
            * torch.eye(
                model.num_frequencies,
                dtype=train_features.dtype,
                device=train_features.device,
            )
        )
        feature_targets = train_features.T @ centered_training_images
        posterior_weight_mean = torch.cholesky_solve(
            feature_targets / noise, cholesky
        )
        return query_features @ posterior_weight_mean
