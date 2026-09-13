"""GPyTorch utilities for a symmetrized RealNVP spectral RFF kernel.

The flow acts on a two-dimensional standard Gaussian.  Its first output
coordinate, multiplied by a learnable positive scale, defines an implicit
one-dimensional spectral distribution.  Random signs symmetrize the frequency
samples.  During training, fresh base samples and signs are drawn at every
optimization step.  The resulting frequencies form paired sine/cosine random
Fourier features and are trained through the GP marginal likelihood.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import gpytorch
import numpy as np
import torch
from torch import nn


Tensor = torch.Tensor


class AffineCoupling(nn.Module):
    """One RealNVP affine coupling transform with a fixed binary mask."""

    def __init__(
        self,
        dimension: int,
        hidden_features: int,
        mask: Tensor,
        max_log_scale: float = 1.5,
    ) -> None:
        super().__init__()
        if mask.shape != (dimension,):
            raise ValueError("mask must have shape (dimension,)")
        self.register_buffer("mask", mask.to(dtype=torch.get_default_dtype()))
        self.max_log_scale = max_log_scale
        self.network = nn.Sequential(
            nn.Linear(dimension, hidden_features),
            nn.Tanh(),
            nn.Linear(hidden_features, hidden_features),
            nn.Tanh(),
            nn.Linear(hidden_features, 2 * dimension),
        )
        # Start at the identity so optimization begins from a broad Gaussian
        # spectrum rather than an arbitrary randomly distorted distribution.
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        masked_x = self.mask * x
        log_scale, shift = self.network(masked_x).chunk(2, dim=-1)
        transform_mask = 1.0 - self.mask
        log_scale = self.max_log_scale * torch.tanh(log_scale) * transform_mask
        shift = shift * transform_mask
        y = masked_x + transform_mask * (x * torch.exp(log_scale) + shift)
        return y, log_scale.sum(dim=-1)


class RealNVP(nn.Module):
    """A small alternating-mask RealNVP flow."""

    def __init__(
        self,
        dimension: int = 2,
        num_layers: int = 6,
        hidden_features: int = 64,
    ) -> None:
        super().__init__()
        if dimension < 2:
            raise ValueError("RealNVP coupling requires dimension >= 2")
        layers = []
        for layer in range(num_layers):
            mask = torch.tensor(
                [(index + layer) % 2 for index in range(dimension)],
                dtype=torch.get_default_dtype(),
            )
            layers.append(AffineCoupling(dimension, hidden_features, mask))
        self.layers = nn.ModuleList(layers)

    def forward(self, base_samples: Tensor) -> tuple[Tensor, Tensor]:
        values = base_samples
        log_abs_det = torch.zeros(
            base_samples.shape[:-1],
            dtype=base_samples.dtype,
            device=base_samples.device,
        )
        for layer in self.layers:
            values, layer_log_det = layer(values)
            log_abs_det = log_abs_det + layer_log_det
        return values, log_abs_det


class SymmetrizedRealNVPSpectralKernel(gpytorch.kernels.Kernel):
    """Stationary RFF kernel with frequencies sampled by a RealNVP flow."""

    is_stationary = True

    def __init__(
        self,
        *,
        num_frequencies: int = 192,
        flow_dimension: int = 2,
        num_flow_layers: int = 6,
        hidden_features: int = 64,
        initial_frequency_scale: float = 10.0,
        seed: int = 37,
    ) -> None:
        super().__init__()
        if num_frequencies <= 0:
            raise ValueError("num_frequencies must be positive")
        if initial_frequency_scale <= 0:
            raise ValueError("initial_frequency_scale must be positive")

        self.num_frequencies = num_frequencies
        self.flow_dimension = flow_dimension
        self.flow = RealNVP(
            dimension=flow_dimension,
            num_layers=num_flow_layers,
            hidden_features=hidden_features,
        )
        self.raw_frequency_scale = nn.Parameter(
            torch.tensor(float(np.log(initial_frequency_scale)))
        )

        with torch.random.fork_rng():
            torch.manual_seed(seed)
            base_samples = torch.randn(num_frequencies, flow_dimension)
            signs = 2 * torch.randint(0, 2, (num_frequencies,)) - 1
        self.register_buffer("base_samples", base_samples)
        self.register_buffer("signs", signs.to(dtype=base_samples.dtype))

    @property
    def frequency_scale(self) -> Tensor:
        return torch.exp(self.raw_frequency_scale)

    def unsigned_frequencies(self, base_samples: Optional[Tensor] = None) -> Tensor:
        if base_samples is None:
            base_samples = self.base_samples
        flow_values, _ = self.flow(base_samples)
        return self.frequency_scale * flow_values[..., 0]

    def frequencies(self) -> Tensor:
        """Return the current sign-symmetrized Monte Carlo frequencies."""
        return self.signs * self.unsigned_frequencies()

    @torch.no_grad()
    def resample_training_frequencies(
        self,
        generator: Optional[torch.Generator] = None,
    ) -> None:
        """Draw fresh base samples and Rademacher signs in place.

        The random draws do not require gradients.  Their transformation by
        ``self.flow`` remains pathwise differentiable with respect to every
        flow and frequency-scale parameter.
        """
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
        """Mean squared norm of the current training frequencies."""
        return self.frequencies().square().mean()

    def _features(self, x: Tensor) -> Tensor:
        if x.shape[-1] != 1:
            raise ValueError("this spectral kernel supports one-dimensional inputs")
        projection = x.squeeze(-1).unsqueeze(-1) * self.frequencies()
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
        """Draw fresh symmetrized frequencies for visualization."""
        parameter = next(self.parameters())
        generator = torch.Generator(device=parameter.device)
        generator.manual_seed(seed)
        base = torch.randn(
            num_samples,
            self.flow_dimension,
            generator=generator,
            dtype=parameter.dtype,
            device=parameter.device,
        )
        signs = 2 * torch.randint(
            0,
            2,
            (num_samples,),
            generator=generator,
            device=parameter.device,
        ) - 1
        return signs.to(dtype=parameter.dtype) * self.unsigned_frequencies(base)


class NFSpectralGP(gpytorch.models.ExactGP):
    """Exact GP using a low-rank NF spectral RFF covariance."""

    def __init__(
        self,
        train_x: Tensor,
        train_curves: Tensor,
        likelihood: gpytorch.likelihoods.GaussianLikelihood,
        kernel: SymmetrizedRealNVPSpectralKernel,
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
    if x_array.ndim != 1:
        raise ValueError("x must be one-dimensional")
    if curves_array.ndim != 2 or curves_array.shape[1] != x_array.size:
        raise ValueError("curves must have shape (n_curves, len(x))")
    return (
        torch.as_tensor(x_array, dtype=dtype).unsqueeze(-1),
        torch.as_tensor(curves_array, dtype=dtype),
    )


def build_nf_spectral_gp(
    train_x: Tensor,
    train_curves: Tensor,
    *,
    num_frequencies: int = 192,
    num_flow_layers: int = 6,
    hidden_features: int = 64,
    initial_frequency_scale: float = 10.0,
    initial_outputscale: float = 1.0,
    initial_noise: float = 0.005,
    seed: int = 37,
) -> tuple[NFSpectralGP, gpytorch.likelihoods.GaussianLikelihood]:
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    likelihood.initialize(noise=initial_noise)
    kernel = SymmetrizedRealNVPSpectralKernel(
        num_frequencies=num_frequencies,
        num_flow_layers=num_flow_layers,
        hidden_features=hidden_features,
        initial_frequency_scale=initial_frequency_scale,
        seed=seed,
    )
    model = NFSpectralGP(train_x, train_curves, likelihood, kernel)
    model.covar_module.initialize(outputscale=initial_outputscale)
    return model, likelihood


@dataclass(frozen=True)
class NFTrainingHistory:
    objective: list[float]
    negative_mll: list[float]
    frequency_penalty: list[float]
    frequency_rms: list[float]


def train_nf_spectral_gp(
    model: NFSpectralGP,
    likelihood: gpytorch.likelihoods.GaussianLikelihood,
    train_x: Tensor,
    train_curves: Tensor,
    *,
    num_steps: int = 400,
    learning_rate: float = 0.01,
    frequency_penalty_weight: float = 1e-4,
    resample_frequencies_each_step: bool = True,
    resampling_seed: int = 2026,
    print_every: int = 50,
) -> NFTrainingHistory:
    """Train with negative evidence plus an L2 frequency-norm penalty.

    By default, a fresh Monte Carlo draw from the base distribution (and fresh
    symmetrizing signs) is used at every optimization step.
    """
    if num_steps <= 0 or learning_rate <= 0:
        raise ValueError("num_steps and learning_rate must be positive")
    if frequency_penalty_weight < 0:
        raise ValueError("frequency_penalty_weight must be non-negative")

    model.train()
    likelihood.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    objective_history: list[float] = []
    nll_history: list[float] = []
    penalty_history: list[float] = []
    rms_history: list[float] = []
    kernel = model.covar_module.base_kernel
    resampling_generator = torch.Generator(device=kernel.base_samples.device)
    resampling_generator.manual_seed(resampling_seed)

    for step in range(1, num_steps + 1):
        if resample_frequencies_each_step:
            kernel.resample_training_frequencies(resampling_generator)
        optimizer.zero_grad()
        prior = model(train_x)
        negative_mll = -mll(prior, train_curves).mean()
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

    return NFTrainingHistory(
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
    model: NFSpectralGP,
    likelihood: gpytorch.likelihoods.GaussianLikelihood,
    observed_x: np.ndarray,
    observed_y: np.ndarray,
    prediction_x: np.ndarray,
) -> Prediction:
    parameter = next(model.parameters())
    x_obs = torch.as_tensor(
        observed_x, dtype=parameter.dtype, device=parameter.device
    ).unsqueeze(-1)
    y_obs = torch.as_tensor(
        observed_y, dtype=parameter.dtype, device=parameter.device
    )
    x_pred = torch.as_tensor(
        prediction_x, dtype=parameter.dtype, device=parameter.device
    ).unsqueeze(-1)
    original_inputs = model.train_inputs
    original_targets = model.train_targets
    model.eval()
    likelihood.eval()
    try:
        model.set_train_data(inputs=x_obs, targets=y_obs, strict=False)
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            posterior = likelihood(model(x_pred))
            lower, upper = posterior.confidence_region()
            mean = posterior.mean
    finally:
        model.set_train_data(
            inputs=original_inputs, targets=original_targets, strict=False
        )
    return Prediction(
        mean=mean.detach().cpu().numpy(),
        lower=lower.detach().cpu().numpy(),
        upper=upper.detach().cpu().numpy(),
    )
