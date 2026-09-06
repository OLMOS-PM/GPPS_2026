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
from gpytorch.constraints import GreaterThan, Positive


Tensor = torch.Tensor


class SharedHyperparameterGP(gpytorch.models.ExactGP):
    """Independent GP realizations with one shared set of hyperparameters."""
    """ This class defines a Gaussian Process (GP) model where multiple independent realizations (or curves) 
    share the same set of hyperparameters. It inherits from GPyTorch's ExactGP class, 
    which is used for exact inference in Gaussian Processes. The shared hyperparameters include the mean, lengthscale, output scale, and 
    observation noise variance, which are learned from the training data."""

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


class SymmetricSpectralMixtureRFFKernel(gpytorch.kernels.Kernel):
    """RFF kernel for a mixture of symmetric Gaussian spectral pairs.

    Component q has normalized spectral density

        0.5 N(+mean_q, spectral_variance)
        + 0.5 N(-mean_q, spectral_variance).

    Fixed standard-normal draws provide a differentiable reparameterization
    ``frequency = mean_q + sqrt(variance_q) * epsilon``. Paired cosine/sine
    features implicitly symmetrize each component. The positive means and the
    softmax mixture weights are learned by marginal-likelihood optimization.
    """

    is_stationary = True

    def __init__(
        self,
        component_means: Tensor,
        *,
        num_frequencies_per_component: int = 64,
        spectral_variance: float | Tensor = 1.0,
        learn_spectral_variance: bool = False,
        initial_weights: Optional[Tensor] = None,
        seed: int = 23,
    ) -> None:
        super().__init__()
        means = torch.as_tensor(component_means, dtype=torch.get_default_dtype())
        if means.ndim != 1 or means.numel() == 0 or torch.any(means <= 0):
            raise ValueError("component_means must be a non-empty positive 1D tensor")
        if num_frequencies_per_component <= 0:
            raise ValueError("num_frequencies_per_component must be positive")
        variances = torch.as_tensor(spectral_variance, dtype=means.dtype)
        if variances.ndim == 0:
            variances = variances.expand_as(means).clone() #reshape the variances tensor to match the shape of means if it is a scalar
        if variances.shape != means.shape or torch.any(variances <= 0):
            raise ValueError(
                "spectral_variance must be positive and scalar or match the means"
            )

        self.num_components = int(means.numel())
        self.num_frequencies_per_component = num_frequencies_per_component
        self.register_parameter(
            # Create torch parameter "raw_component_means" initialized to zeros with the same shape as "means". This parameter will be transformed to ensure positivity.
            "raw_component_means", torch.nn.Parameter(torch.zeros_like(means))
        )
        self.register_constraint("raw_component_means", Positive())
        self.component_means = means

        if initial_weights is None:
            # Default to uniform mixture weights if no initial weights are provided. Create a tensor of the same shape as "means" filled with the value 1.0 / num_components, ensuring that the sum of the weights is 1.
            weights = torch.full_like(means, 1.0 / self.num_components)
        else:
            weights = torch.as_tensor(initial_weights, dtype=means.dtype)
            if weights.shape != means.shape or torch.any(weights <= 0):
                raise ValueError("initial_weights must be positive and match the means")
            weights = weights / weights.sum()
        self.raw_mixture_logits = torch.nn.Parameter(weights.log()) # we optimize logits instead of weights to ensure positivity and sum-to-one constraint after applying softmax
        self.register_parameter(
            "raw_spectral_variances", torch.nn.Parameter(torch.zeros_like(variances))
        )
        self.register_constraint("raw_spectral_variances", Positive())
        self.spectral_variances = variances
        self.raw_spectral_variances.requires_grad_(learn_spectral_variance)

        with torch.random.fork_rng():
            torch.manual_seed(seed)
            half_count = (self.num_frequencies_per_component + 1) // 2
            """ Generate half the number of random draws from a standard normal distribution. 
            These draws will be used to create antithetic pairs (positive and negative) to ensure symmetry in the spectral mixture. 
            The use of antithetic draws helps reduce variance in the estimation of the kernel and improves convergence during training."""

            half_draws = torch.randn(half_count, dtype=means.dtype)
            # Common antithetic draws give every mixture component the same
            # finite-sample envelope and reduce spurious weight differences.
            one_component_draws = torch.cat((half_draws, -half_draws))[
                : self.num_frequencies_per_component
            ]
            standard_frequencies = one_component_draws.unsqueeze(0).repeat(
                self.num_components, 1
            )
        self.register_buffer("standard_frequencies", standard_frequencies)

    @property # This decorator indicates that the method can be accessed like an attribute
    def component_means(self) -> Tensor:
        return self.raw_component_means_constraint.transform(self.raw_component_means)

    """ For simplicity, means are constrained to be positive. The setter method allows the user to set the component 
    means while ensuring that they remain positive. It takes a tensor value as input, 
    converts it to the appropriate dtype and device, and then applies the inverse transformation of the 
    constraint to update the raw_component_means parameter."""
    @component_means.setter
    def component_means(self, value: Tensor) -> None:
        value = torch.as_tensor(
            value,
            dtype=self.raw_component_means.dtype,
            device=self.raw_component_means.device,
        )
        self.initialize(
            raw_component_means=self.raw_component_means_constraint.inverse_transform(value)
        )

    @property
    def mixture_weights(self) -> Tensor:
        return torch.softmax(self.raw_mixture_logits, dim=-1)

    @property
    def spectral_variances(self) -> Tensor:
        return self.raw_spectral_variances_constraint.transform(
            self.raw_spectral_variances
        )

    @spectral_variances.setter
    def spectral_variances(self, value: Tensor) -> None:
        value = torch.as_tensor(
            value,
            dtype=self.raw_spectral_variances.dtype,
            device=self.raw_spectral_variances.device,
        )
        self.initialize(
            raw_spectral_variances=(
                self.raw_spectral_variances_constraint.inverse_transform(value)
            )
        )

    def _features(self, x: Tensor) -> Tensor:

        # Here we check that the input tensor x has the correct shape. The kernel is designed to work with 1D inputs, so we ensure that the last dimension of x is 1. If not, we raise a ValueError indicating that the kernel only supports 1D inputs.
        if x.shape[-1] != 1:
            raise ValueError("SymmetricSpectralMixtureRFFKernel supports 1D inputs")

        # We scale and shift the standard normal draws to create the frequencies for each component of the mixture. The frequencies are computed as the sum of the component means and the product of the square root of the spectral variances and the standard normal draws. This ensures that each component has its own set of frequencies based on its mean and variance.
        frequencies = (
            self.component_means.unsqueeze(-1)
            + self.spectral_variances.sqrt().unsqueeze(-1)
            * self.standard_frequencies
        )

        # We compute the feature vector for the RFF kernel. 
        projection = x.squeeze(-1).unsqueeze(-1).unsqueeze(-1) * frequencies
        features = torch.cat((projection.cos(), projection.sin()), dim=-1)
        scale_shape = [1] * (features.ndim - 2) + [self.num_components, 1]
        scales = (
            self.mixture_weights / self.num_frequencies_per_component
        ).sqrt().view(*scale_shape)
        features = features * scales
        return features.flatten(start_dim=-2)

    def forward(
        self,
        x1: Tensor,
        x2: Tensor,
        diag: bool = False,
        **params: object,
    ) -> Tensor:
        features_1 = self._features(x1)
        features_2 = self._features(x2)
        """ k(x1, x2) = phi(x1) phi(x2)^T, where phi is the feature mapping defined by the RFF kernel. 
        This computes the covariance matrix between the inputs x1 and x2 based on their feature representations.
        If diag is True, we return only the diagonal elements of the covariance matrix, which correspond to the variances of each input point.
        """
        if diag:
            return (features_1 * features_2).sum(dim=-1)
        return features_1 @ features_2.transpose(-1, -2) 

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
    # Create a new Gaussian likelihood with a positive noise constraint and initialize its noise parameter. Raise a ValueError if the provided noise_init is not positive.
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
    train_x: Tensor, # Positional tensor of shape (n_points, 1) representing the input grid for training curves.
    train_curves: Tensor, # Positional tensor of shape (n_curves, n_points) representing the training curves observed on the input grid.
    *, #this * indicates that the following parameters are keyword-only arguments.
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
    with torch.random.fork_rng(devices=devices): #random.fork_rng creates a context in which the random number generator state is temporarily modified. This ensures that any random numbers generated within this block do not affect the global random state outside of it, allowing for reproducibility without altering the caller's random number generator state.
        torch.manual_seed(seed)
        base_kernel = gpytorch.kernels.RFFKernel(
            num_samples=num_features // 2,
            num_dims=train_x.shape[-1],
        )
    base_kernel.initialize(lengthscale=initial_lengthscale)
    model = SharedHyperparameterGP(train_x, train_curves, likelihood, base_kernel)
    model.covar_module.initialize(outputscale=initial_outputscale)
    return model, likelihood


def build_symmetric_spectral_mixture_rff_gp(
    train_x: Tensor,
    train_curves: Tensor,
    *,
    initial_component_means: Tensor,
    num_frequencies_per_component: int = 64,
    spectral_variance: float | Tensor = 1.0,
    learn_spectral_variance: bool = False,
    initial_weights: Optional[Tensor] = None,
    seed: int = 23,
    initial_outputscale: float = 1.0,
    initial_noise: float = 0.01,
) -> tuple[SharedHyperparameterGP, gpytorch.likelihoods.GaussianLikelihood]:
    """Build a GP with a learnable symmetric spectral-mixture RFF kernel."""
    likelihood = _new_likelihood(initial_noise)
    base_kernel = SymmetricSpectralMixtureRFFKernel(
        initial_component_means,
        num_frequencies_per_component=num_frequencies_per_component,
        spectral_variance=spectral_variance,
        learn_spectral_variance=learn_spectral_variance,
        initial_weights=initial_weights,
        seed=seed,
    )
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
            base_kernel = model.covar_module.base_kernel
            if isinstance(base_kernel, SymmetricSpectralMixtureRFFKernel): #if the base kernel is an instance of SymmetricSpectralMixtureRFFKernel, we extract and print the learned parameters specific to this kernel type, including the component means, mixture weights, spectral variances, and noise level.
                means = base_kernel.component_means.detach().cpu().numpy()
                weights = base_kernel.mixture_weights.detach().cpu().numpy()
                variances = base_kernel.spectral_variances.detach().cpu().numpy()
                print(
                    f"step {step:4d}/{num_steps} | loss={loss_value:.4f} | "
                    f"spectral_means={np.round(means, 3)} | "
                    f"variances={np.round(variances, 3)} | "
                    f"weights={np.round(weights, 3)} | "
                    f"noise={float(likelihood.noise.detach().cpu().squeeze()):.5f}"
                )
            else:
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


def learned_spectral_mixture_parameters(
    model: SharedHyperparameterGP,
    likelihood: gpytorch.likelihoods.GaussianLikelihood,
) -> dict[str, object]:
    """Return learned parameters from a symmetric spectral-mixture RFF GP."""
    kernel = model.covar_module.base_kernel
    if not isinstance(kernel, SymmetricSpectralMixtureRFFKernel):
        raise TypeError("model does not use SymmetricSpectralMixtureRFFKernel")
    return {
        "mean": float(model.mean_module.constant.detach().cpu().squeeze()), # GP constant mean
        "component_means": kernel.component_means.detach().cpu().numpy().copy(),
        "mixture_weights": kernel.mixture_weights.detach().cpu().numpy().copy(),
        "spectral_variances": (
            kernel.spectral_variances.detach().cpu().numpy().copy()
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
