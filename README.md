# FiRe: Fixed-Point Iteration for Image Restoration

[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2411.18970)

[Matthieu Terris](https://matthieutrs.github.io/), [Ulugbek S. Kamilov](https://ukmlv.github.io/), [Thomas Moreau](https://tommoral.github.io/about.html).

This repository contains the official implementation of our CVPR 2025 paper "FiRe: Fixed-Point Iteration for Image Restoration". 
This work introduces a novel approach to image restoration from the viewpoint of fixed-point iteration.

![flexible](https://github.com/matthieutrs/FiRe_public/blob/main/images/summary_compressed.jpg) 

## Features

The core idea of this work is to observe that, given a restoration model $\text{R}$ associated to a degradation operator $D$,
the operator $\text{R}\circ D$ is an implicit prior of the form $x - \nabla \log p(x)$.
In turns, this allows do derive new PnP-like algorithms that can be used to solve inverse problems.

Given a measurement $y = Ax + e$, where $A$ is a linear operator, $e$ is the realisation of some random noise, and $x$ is the image to restore, define $f(x) = \frac{1}{2} \|Ax - y\|^2$ as the data fidelity term.
Then, if $\text{R}$ is a restoration model, we can define the fixed-point iteration associated to $\text{R}$ as

we can define the following algorithm:
``` math
\begin{align*}
&u_k = (1-\gamma) x_k + \gamma \text{R}(x_{k}) \\
&x_{k+1} = \text{prox}_{\lambda f}(u_k) \\
\end{align*}
```

where $\gamma$ is a step size, and $\lambda$ is a regularization parameter.

## Code
To reproduce the experiments, first download the test datasets and place them in your data folder. Next, update the `config/config.json` file to point to the correct data folder. There, there are two folders to specify:
- `ROOT_DATASET`: the folder within which the [CBSD68](https://huggingface.co/datasets/deepinv/CBSD68) and [set3c](https://huggingface.co/datasets/deepinv/set3c) datasets are located;
- `ROOT_CKPT`: the path to the folder containing the pre-trained models.

Then, you can run the following scripts to reproduce the experiments:

### Single restoration prior

```bash
python run_baselines.py --problem='gaussian_blur' --method_name='GD_swinir_2x' --dataset_name='set3c' --noise_level=0.01 --sigma_noise_max=0.01 --max_iter=60 --lambd=50 --eq=0
```

<details>
<summary><strong>Motion deblurring prior</strong> (click to expand) </summary>

```bash
python run_baselines.py --problem='gaussian_blur' --method_name='GD_restormer_motion' --dataset_name='set3c' --noise_level=0.01 --l_value=0.6 --sigma_blur_value=1. --max_iter=60 --gamma=0.5 --motion_sigma_noise=0.005 --lambd=50 --eq=1 --results_folder='results_single_prior/'
python run_baselines.py --problem='motion_blur' --method_name='GD_restormer_motion' --dataset_name='set3c' --noise_level=0.01 --l_value=0.6 --sigma_blur_value=1. --max_iter=60 --gamma=1.0 --motion_sigma_noise=0.005 --lambd=20 --eq=1 --results_folder='results_single_prior/'
python run_baselines.py --problem='SRx4' --method_name='GD_restormer_motion' --dataset_name='set3c' --noise_level=0.01 --l_value=0.6 --sigma_blur_value=1. --max_iter=60 --gamma=1.0 --motion_sigma_noise=0.005 --lambd=50 --eq=1 --results_folder='results_single_prior/'
```
</details>

<details>
<summary><strong>Gaussian deblurring prior</strong> (click to expand) </summary>

```bash
python run_baselines.py --problem='gaussian_blur' --method_name='GD_restormer_gaussian' --dataset_name='set3c' --noise_level=0.01 --sigma_blur_min=0.01 --sigma_blur_max=3.0 --max_iter=60 --gamma=1.0 --motion_sigma_noise=0.005 --lambd=20 --results_folder='results_single_prior/'
python run_baselines.py --problem='motion_blur' --method_name='GD_restormer_gaussian' --dataset_name='set3c' --noise_level=0.01 --sigma_blur_min=0.01 --sigma_blur_max=1.0 --max_iter=60 --gamma=1.0 --motion_sigma_noise=0.05 --lambd=20 --results_folder='results_single_prior/'
python run_baselines.py --problem='SRx4' --method_name='GD_restormer_gaussian' --dataset_name='set3c' --noise_level=0.01 --sigma_blur_min=0.01 --sigma_blur_max=4.0 --max_iter=60 --gamma=1.0 --motion_sigma_noise=0.02 --lambd=20 --results_folder='results_single_prior/'
```
</details>

<details>
<summary><strong>Super resolution x2 / x3 prior</strong> (click to expand) </summary>

For SRx2:
```bash
python run_baselines.py --problem='gaussian_blur' --method_name='GD_swinir_2x' --dataset_name='set3c' --noise_level=0.01 --sigma_noise_max=0.01 --max_iter=60 --lambd=50 --eq=0 --results_folder='results_single_prior/'
python run_baselines.py --problem='motion_blur' --method_name='GD_swinir_2x' --dataset_name='set3c' --noise_level=0.01 --sigma_noise_max=0.01 --max_iter=60 --lambd=100 --eq=0 --results_folder='results_single_prior/'
python run_baselines.py --problem='SRx4' --method_name='GD_swinir_2x' --dataset_name='set3c' --noise_level=0.01 --sigma_noise_max=0.01 --max_iter=60 --lambd=50 --eq=0 --results_folder='results_single_prior/'
```

For SRx3:
```bash
python run_baselines.py --problem='gaussian_blur' --method_name='GD_swinir_2x' --dataset_name='set3c' --noise_level=0.01 --sigma_noise_max=0.01 --max_iter=60 --lambd=50 --eq=0 --results_folder='results_single_prior/'
python run_baselines.py --problem='motion_blur' --method_name='GD_swinir_2x' --dataset_name='set3c' --noise_level=0.01 --sigma_noise_max=0.01 --max_iter=60 --lambd=100 --eq=0 --results_folder='results_single_prior/'
python run_baselines.py --problem='SRx4' --method_name='GD_swinir_2x' --dataset_name='set3c' --noise_level=0.01 --sigma_noise_max=0.01 --max_iter=60 --lambd=50 --eq=0 --results_folder='results_single_prior/'
```
</details>

<details>
<summary><strong>de-JPEG/denoising prior</strong> (click to expand) </summary>

```bash
srun python run_baselines.py --problem='gaussian_blur' --method_name='GD_scunet_jpeg' --dataset_name='set3c' --noise_level=0.01 --sigma_noise_min=0.0 --sigma_noise_max=0.1 --quality_min=40 --quality_max=80 --lambd=20 --max_iter=60 --results_folder='results_single_prior/'
srun python run_baselines.py --problem='motion_blur' --method_name='GD_scunet_jpeg' --dataset_name='set3c' --noise_level=0.01 --sigma_noise_min=0.0 --sigma_noise_max=0.01 --quality_min=40 --quality_max=80 --lambd=20 --max_iter=60 --results_folder='results_single_prior/'
srun python run_baselines.py --problem='SRx4' --method_name='GD_scunet_jpeg' --dataset_name='set3c' --noise_level=0.01 --sigma_noise_min=0.05 --sigma_noise_max=0.05 --quality_min=20 --quality_max=60 --lambd=50 --max_iter=80 --results_folder='results_single_prior/'
```
</details>

<details>
<summary><strong>Inpainting prior</strong> (click to expand) </summary>

```bash
python run_baselines.py --problem='gaussian_blur' --method_name='GD_lama' --dataset_name='set3c' --p_mask_min=0.6 --p_mask_max=0.6 --sigma_noise_max=0.0 --lambd=50 --gamma=0.6 --max_iter=20 --equivariant=0 --results_folder='results_single_prior/'
python run_baselines.py --problem='motion_blur' --method_name='GD_lama' --dataset_name='set3c' --p_mask_min=0.6 --p_mask_max=0.6 --sigma_noise_max=0.0 --lambd=20 --gamma=0.6 --max_iter=20 --equivariant=0 --results_folder='results_single_prior/'
python run_baselines.py --problem='SRx4' --method_name='GD_lama' --dataset_name='set3c' --p_mask_min=0.6 --p_mask_max=0.6 --sigma_noise_max=0.0 --lambd=50 --gamma=0.8 --max_iter=20 --equivariant=0 --results_folder='results_single_prior/'
```
</details>

### Combining priors
The above interpretation suggests that multiple restoration priors can be combined by simply averaging their outputs.
To do so, we can use the `--list_gamma_values` argument to specify the $\gamma$ values for each prior.
We observed that a good choice for combined priors is to use (a) the SCUNet de-JPEG prior, (b) the gaussian deblurring prior, and (c) the SR prior.

To run the combined prior experiments, you can use the following commands:

```bash
python run_baselines.py --problem='SRx4' --method_name='GD_multi_scunet_denoise_restormer_gaussian_swinir_3x' --dataset_name='set3c' --noise_level=0.01 --sigma_blur_min=0.01 --sigma_blur_max=0.1 --max_iter=30 --list_gamma_values 1.5 0.25 1.25 --motion_sigma_noise=0.005 --sigma_noise_min=0.0 --sigma_noise_max=0.1 --lambd=100 --results_folder='results_multiprior/' --equivariant=0
```

## 📝 Citation

If you find this work useful for your research, please cite our paper:

```bibtex
@inproceedings{terris2025fire,
  title={FiRe: Fixed-points of restoration priors for solving inverse problems},
  author={Terris, Matthieu and Kamilov, Ulugbek S and Moreau, Thomas},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={23185--23194},
  year={2025}
}
```
