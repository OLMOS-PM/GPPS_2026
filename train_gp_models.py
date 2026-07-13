"""Train shared-hyperparameter RBF and RFF Gaussian processes with GPyTorch.

The rows of ``train_curves`` are treated as independent realizations observed
on the same input grid.  Their marginal log likelihoods are averaged during
training, so every training curve contributes to one shared mean, lengthscale,
output scale, and observation-noise variance.

The RFF model uses GPyTorch's RFFKernel.  Passing ``num_features=L`` draws
``L/2`` frequencies and creates a cosine and sine feature for each draw, for
exactly ``L`` random Fourier features in total.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import gpytorch
import numpy as np
import torch
from gpytorch.constraints import GreaterThan


Tensor = torch.Tensor


class SharedHyperparameterGP(gpytorch.models.ExactGP):
    """Independent GP realizations with one shared set of hyperparameters."""

    def __init__(
        self,
        train_x: Tensor,
        train_curves: Tensor,
        likelihood: gpytorch.likelihoods.GaussianLikelihood,
        base_kernel: gpytorch.kernels.Kernel,
    ) -> None:
        super().__init__(train_x, train_curves, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(base_kernel)

    def forward(self, x: Tensor) -> gpytorch.distributions.MultivariateNormal:
        mean = self.mean_module(x)
        covariance = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covariance)


@dataclass(frozen=True)
class SparseGPPrediction:
    """Posterior summary for one sparsely observed test signal."""

    x: np.ndarray
    mean: np.ndarray
    lower: np.ndarray
    upper: np.ndarray


def as_gp_tensors(
    x: np.ndarray,
    curves: np.ndarray,
    *,
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
) -> tuple[Tensor, Tensor]:
    """Convert a grid and a matrix of curves to GPyTorch-compatible tensors."""
    x_array = np.asarray(x)
    curves_array = np.asarray(curves)
    if x_array.ndim != 1:
        raise ValueError("x must be one-dimensional")
    if curves_array.ndim != 2 or curves_array.shape[1] != x_array.size:
        raise ValueError("curves must have shape (n_curves, len(x))")

    train_x = torch.as_tensor(x_array, dtype=dtype, device=device).unsqueeze(-1)
    train_curves = torch.as_tensor(curves_array, dtype=dtype, device=device)
    return train_x, train_curves


def _new_likelihood(noise_init: float) -> gpytorch.likelihoods.GaussianLikelihood:
    if noise_init <= 0:
        raise ValueError("noise_init must be positive")
    likelihood = gpytorch.likelihoods.GaussianLikelihood(
        noise_constraint=GreaterThan(1e-6)
    )
    likelihood.initialize(noise=noise_init)
    return likelihood


def build_rbf_gp(
    train_x: Tensor,
    train_curves: Tensor,
    *,
    initial_lengthscale: float = 0.2,
    initial_outputscale: float = 1.0,
    initial_noise: float = 0.01,
) -> tuple[SharedHyperparameterGP, gpytorch.likelihoods.GaussianLikelihood]:
    """Build an exact GP with an RBF covariance kernel."""
    likelihood = _new_likelihood(initial_noise)
    base_kernel = gpytorch.kernels.RBFKernel()
    base_kernel.initialize(lengthscale=initial_lengthscale)
    model = SharedHyperparameterGP(train_x, train_curves, likelihood, base_kernel)
    model.covar_module.initialize(outputscale=initial_outputscale)
    return model, likelihood


def build_rff_gp(
    train_x: Tensor,
    train_curves: Tensor,
    *,
    num_features: int = 128,
    seed: int = 19,
    initial_lengthscale: float = 0.2,
    initial_outputscale: float = 1.0,
    initial_noise: float = 0.01,
) -> tuple[SharedHyperparameterGP, gpytorch.likelihoods.GaussianLikelihood]:
    """Build an RBF GP approximated with ``num_features / 2`` frequencies."""
    if num_features <= 0 or num_features % 2 != 0:
        raise ValueError("num_features must be a positive even integer")

    likelihood = _new_likelihood(initial_noise)
    # RFFKernel generates both sin and cos features for every sampled frequency.
    # fork_rng keeps construction reproducible without changing the caller's RNG.
    devices = [train_x.device] if train_x.is_cuda else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        base_kernel = gpytorch.kernels.RFFKernel(
            num_samples=num_features // 2,
            num_dims=train_x.shape[-1],
        )
    base_kernel.initialize(lengthscale=initial_lengthscale)
    model = SharedHyperparameterGP(train_x, train_curves, likelihood, base_kernel)
    model.covar_module.initialize(outputscale=initial_outputscale)
    return model, likelihood


def train_shared_gp(
    model: SharedHyperparameterGP,
    likelihood: gpytorch.likelihoods.GaussianLikelihood,
    train_x: Tensor,
    train_curves: Tensor,
    *,
    num_steps: int = 100,
    learning_rate: float = 0.05,
    print_every: int = 10,
) -> list[float]:
    """Maximize the average marginal likelihood over all training curves."""
    if num_steps <= 0:
        raise ValueError("num_steps must be positive")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")

    model.train()
    likelihood.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    marginal_log_likelihood = gpytorch.mlls.ExactMarginalLogLikelihood(
        likelihood, model
    )
    losses: list[float] = []

    for step in range(1, num_steps + 1):
        optimizer.zero_grad()
        prior = model(train_x)
        # The MLL has one value per independent curve. Averaging learns one set
        # of shared parameters without making the gradient scale depend on the
        # number of curves in the training partition.
        loss = -marginal_log_likelihood(prior, train_curves).mean()
        loss.backward()
        optimizer.step()

        loss_value = float(loss.detach().cpu())
        losses.append(loss_value)
        if print_every > 0 and (step == 1 or step % print_every == 0):
            parameters = learned_hyperparameters(model, likelihood)
            print(
                f"step {step:4d}/{num_steps} | loss={loss_value:.4f} | "
                f"ell={parameters['lengthscale']:.4f} | "
                f"scale={parameters['outputscale']:.4f} | "
                f"noise={parameters['noise']:.5f}"
            )

    return losses


def mean_negative_log_likelihood(
    model: SharedHyperparameterGP,
    likelihood: gpytorch.likelihoods.GaussianLikelihood,
    x: Tensor,
    curves: Tensor,
) -> float:
    """Evaluate mean NLL per observation, averaged over independent curves."""
    model.train()
    likelihood.train()
    marginal_log_likelihood = gpytorch.mlls.ExactMarginalLogLikelihood(
        likelihood, model
    )
    with torch.no_grad():
        prior = model(x)
        nll = -marginal_log_likelihood(prior, curves).mean()
    return float(nll.detach().cpu())


def learned_hyperparameters(
    model: SharedHyperparameterGP,
    likelihood: gpytorch.likelihoods.GaussianLikelihood,
) -> dict[str, float]:
    """Return the learned scalar kernel and likelihood parameters."""
    return {
        "mean": float(model.mean_module.constant.detach().cpu().squeeze()),
        "lengthscale": float(
            model.covar_module.base_kernel.lengthscale.detach().cpu().squeeze()
        ),
        "outputscale": float(model.covar_module.outputscale.detach().cpu()),
        "noise": float(likelihood.noise.detach().cpu().squeeze()),
    }


def predict_sparse_signal(
    model: SharedHyperparameterGP,
    likelihood: gpytorch.likelihoods.GaussianLikelihood,
    observed_x: np.ndarray,
    observed_y: np.ndarray,
    prediction_x: np.ndarray,
) -> SparseGPPrediction:
    """Condition learned population parameters on a few points from one curve.

    The model's original population training data are restored before returning;
    only the fitted hyperparameters are used for the test signal.
    """
    observed_x = np.asarray(observed_x)
    observed_y = np.asarray(observed_y)
    prediction_x = np.asarray(prediction_x)
    if observed_x.ndim != 1 or observed_y.shape != observed_x.shape:
        raise ValueError("observed_x and observed_y must be matching 1D arrays")
    if prediction_x.ndim != 1:
        raise ValueError("prediction_x must be one-dimensional")
    if observed_x.size < 2:
        raise ValueError("at least two observations are required")

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

    return SparseGPPrediction(
        x=prediction_x.copy(),
        mean=mean.detach().cpu().numpy(),
        lower=lower.detach().cpu().numpy(),
        upper=upper.detach().cpu().numpy(),
    )
