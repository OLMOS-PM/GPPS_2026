# Gaussian Processes Summer School 2026

Teaching material on Gaussian processes, random Fourier features, spectral kernels, non-stationarity, and Gaussian process latent variable models (GPLVMs).

## Notebooks

| Part | Topic | Notebook | Rendered HTML |
|---:|---|---|---|
| 1 | GP regression and random Fourier features | [Notebook](1-gp_regression_rff_intro.ipynb) | [HTML](1-gp_regression_rff_intro.html) |
| 2 | Learnable Gaussian spectral mixtures | [Notebook](2-gp_mixture_demo_v3.ipynb) | [HTML](2-gp_mixture_demo_v3.html) |
| 3 | Implicit neural spectral distributions | [Notebook](3-gp_implicit_spectral_demov2.ipynb) | [HTML](3-gp_implicit_spectral_demov2.html) |
| 4 | Amortized variational GPLVMs | [Notebook](4-gp_gplvm_mnist_demov2.ipynb) | [HTML](4-gp_gplvm_mnist_demov2.html) |
| 5 | Non-stationary spectral kernels | [Notebook](5-gp_nonstationary_kernelsv2.ipynb) | [HTML](5-gp_nonstationary_kernelsv2.html) |
| 6 | Stationary versus non-stationary spectral GPLVMs | [Notebook](6-gp_gplvm_nonstationary_spectral.ipynb) | [HTML](6-gp_gplvm_nonstationary_spectral.html) |

Shared Python helpers and media assets are in [`scripts_and_files/`](scripts_and_files/). Run the notebooks from the repository root so their package imports and relative media paths resolve correctly.

## Reproduce the `gpps-mps` environment

The committed [`environment.yml`](environment.yml) is an exact export of the environment used to prepare these notebooks, including Python and library versions. Its Conda build identifiers correspond to macOS on Apple Silicon and the PyTorch installation supports Apple Metal Performance Shaders (MPS).

Install [Miniconda or Anaconda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html), clone this repository, and run:

```bash
conda env create --file environment.yml
conda activate gpps-mps
python -m ipykernel install --user --name gpps-mps --display-name "Python (gpps-mps)"
jupyter lab
```

Then select the **Python (gpps-mps)** kernel in Jupyter and open any numbered notebook.

Because the manifest contains exact Apple Silicon build identifiers, it may not solve unchanged on Linux, Windows, or Intel macOS. On those platforms, remove the Conda build suffixes (the final `=...` portion) while retaining the pinned package versions, then recreate the environment. Numerical results can also vary slightly across CPU, CUDA, and MPS backends.

## Data

The MNIST and Fashion-MNIST loaders first use local data when available and otherwise download the official datasets. Dataset labels are used only for downstream evaluation in the GPLVM demonstrations, not for unsupervised GPLVM training.

## Author and contact

**Pablo M. Olmos**<br>
Universidad Carlos III de Madrid<br>
[Personal website](https://olmos-pm.github.io/)<br>
[pamartin@ing.uc3m.es](mailto:pamartin@ing.uc3m.es)

## License

See [LICENSE](LICENSE).
