from pathlib import Path

import time
import math
import json

import numpy as np
import torch
from torchvision.transforms import v2
import scipy.io

import deepinv as dinv
from deepinv.physics.generator import PhysicsGenerator
from deepinv.utils.tensorlist import dirac

from .inpainting import RandomMask

with open('config/config.json') as f:
    config = json.load(f)

ROOT_DATASET = config['ROOT_DATASET']
GAUSS_KERNEL_PTH = config['GAUSS_KERNEL_PTH']
MOTION_KERNEL_PTH = config['MOTION_KERNEL_PTH']

def get_blur_kernel(id=1):
    pth_blur = MOTION_KERNEL_PTH + '/blur_'+str(id)+'.mat'
    h = scipy.io.loadmat(pth_blur)
    h = torch.from_numpy(np.array(h['blur'])).unsqueeze(0).unsqueeze(0)
    return h

def get_physics(x_true, n_channels, problem_type='sr', device='cpu', sr=2, img_size=None, id_blur=1, noise_level=0.01):

    if problem_type=='inpainting':
        # mask = RandomMask(max(x_true.shape[-2], x_true.shape[-1])) if center_crop is None else RandomMask(center_crop)
        mask = RandomMask(max(x_true.shape[-2], x_true.shape[-1]))
        mask = mask[..., :x_true.shape[-2], :x_true.shape[-1]]
        mask = torch.from_numpy(mask).float().unsqueeze(0)

        physics = dinv.physics.Inpainting(
            tensor_size=(x_true.shape[1], x_true.shape[-2], x_true.shape[-1]),
            mask=mask[0],
            device=device,
            noise_model=dinv.physics.GaussianNoise(sigma=noise_level),
        )
    elif problem_type=='inpainting_small':
        # mask = RandomMask(max(x_true.shape[-2], x_true.shape[-1])) if center_crop is None else RandomMask(center_crop)
        mask = RandomMask(max(x_true.shape[-2], x_true.shape[-1]), hole_range=[0.3, 0.45])
        mask = mask[..., :x_true.shape[-2], :x_true.shape[-1]]
        mask = torch.from_numpy(mask).float().unsqueeze(0)

        physics = dinv.physics.Inpainting(
            tensor_size=(x_true.shape[1], x_true.shape[-2], x_true.shape[-1]),
            mask=mask[0],
            device=device,
            noise_model=dinv.physics.GaussianNoise(sigma=noise_level),
        )

    elif 'SRx' in problem_type:
        sr = int(problem_type[-1])
        physics = dinv.physics.Downsampling((n_channels, x_true.shape[-2], x_true.shape[-1]),
                                            filter='bicubic',
                                            factor=sr,
                                            device=device,
                                            noise_model=dinv.physics.GaussianNoise(sigma=noise_level),)
                                            # padding='constant')
    elif problem_type == 'denoising':
        physics = dinv.physics.DecomposablePhysics()
        physics.noise_model = dinv.physics.GaussianNoise(sigma=0.05)
    elif problem_type == 'gaussian_blur':
        kernel = scipy.io.loadmat(GAUSS_KERNEL_PTH)['kernel']
        kernel_torch = torch.from_numpy(kernel).unsqueeze(0).unsqueeze(0)  # add batch and channel dimensions
        n_channels = 3  # 3 for color images, 1 for gray-scale images
        physics = dinv.physics.BlurFFT(
            img_size=(n_channels, x_true.shape[-2], x_true.shape[-1]),
            filter=kernel_torch,
            device=device,
            noise_model=dinv.physics.GaussianNoise(sigma=noise_level),
        )
    elif problem_type == 'motion_blur':
        blur_filter = get_blur_kernel(id=id_blur)
        physics = dinv.physics.BlurFFT(
            img_size=(n_channels, x_true.shape[-2], x_true.shape[-1]),
            filter=blur_filter,
            device=device,
            noise_model=dinv.physics.GaussianNoise(sigma=noise_level),
        )
    elif problem_type == 'CT':
        physics = dinv.physics.Tomography(
            img_width=x_true.shape[-1],
            angles=100,
            circle=False,
            device=device,
            noise_model=dinv.physics.GaussianNoise(sigma=noise_level),
        )
    else:
        print(f"Physics {problem_type} not implemented")
        raise NotImplementedError

    return physics


class InpaintingRandomMaskGenerator(PhysicsGenerator):

    def __init__(
            self,
            mask_shape: tuple,
            num_channels: int = 1,
            device: str = "cpu",
            dtype: type = torch.float32,
            p_min=0.1,
            p_max=0.9,
    ) -> None:
        kwargs = {"mask_shape": mask_shape}
        if len(mask_shape) != 2:
            raise ValueError(
                "mask_shape must 2D. Add channels via num_channels parameter"
            )
        super().__init__(
            num_channels=num_channels,
            device=device,
            dtype=dtype,
            **kwargs,
        )
        self.p_min = p_min
        self.p_max = p_max

    def generate_mask(self, image_shape):
        # Create an all-ones tensor which will serve as the initial mask
        mask = torch.ones(image_shape)
        batch_size = mask.shape[0]

        # Generate random mask with probability p

        p = torch.rand(batch_size) * (self.p_max - self.p_min) + self.p_min

        # set to same size as mask
        p = p.view(-1, 1, 1, 1)
        p = p.expand(-1, 1, image_shape[-2], image_shape[-1])

        mask = torch.rand(image_shape) < p

        return mask.float()

    def step(self, batch_size: int = 1):
        r"""
        Generate a random motion blur PSF with parameters :math:`\sigma` and :math:`l`

        :param int batch_size: batch_size.
        :param float sigma: the standard deviation of the Gaussian Process
        :param float l: the length scale of the trajectory

        :return: dictionary with key **'filter'**: the generated PSF of shape `(batch_size, 1, psf_size[0], psf_size[1])`
        """

        batch_shape = (batch_size, self.num_channels, self.mask_shape[-2], self.mask_shape[-1])

        mask = self.generate_mask(batch_shape)

        return {
            "mask": mask.to(self.factory_kwargs["device"])
        }


class JPEGTransform(dinv.physics.Physics):
    r"""
    Applies JPEG transform to an image.
    """
    def __init__(self, quality=50, sigma=0., device='cpu', **kwargs):
        super().__init__(**kwargs)
        self.quality = quality
        self.sigma = sigma
        self.device = device

    def forward(self, x, **kwargs):
        r"""
        Computes forward operator

        .. math::

                y = N(A(x), \sigma)


        :param torch.Tensor, list[torch.Tensor] x: signal/image
        :return: (torch.Tensor) noisy measurements

        """
        return self.A(x, **kwargs)

    def A(self, x, quality=None, sigma=None, **kwargs):
        r"""
        Applies the forward operator :math:`y = A(x)`.

        Note: it would be cleaner to define this as a sensor model rather than a measurement operator but for the sake of this project it's easier like that!

        :param torch.Tensor x: input tensor
        :param torch.nn.Parameter, float mask: singular values.
        :return: (torch.Tensor) output tensor
        """
        self.update_parameters(quality=quality, sigma=sigma, **kwargs)

        y = x + self.sigma * torch.randn_like(x)

        if isinstance(self.quality, torch.Tensor):
            q = self.quality.item()
        else:
            q = self.quality

        return self.jpeg(y, q)

    def jpeg(self, x, quality):
        x_device = x.device
        x = torch.clamp(x, 0, 1)
        x_ = (x*255).to(torch.uint8)
        x_ = x_.to('cpu')
        out = v2.functional.jpeg(x_, quality)
        out = out.float()/255.
        return out.to(x_device)

    def update_parameters(self, **kwargs):
        r"""
        Updates the singular values of the operator.

        """
        for key, value in kwargs.items():
            if value is not None and hasattr(self, key):
                if 'quality' in key:
                    setattr(self, key, value)
                else:
                    setattr(self, key, torch.nn.Parameter(value, requires_grad=False).to(self.device))


class InpaintingBrushGenerator(PhysicsGenerator):

    def __init__(
            self,
            mask_shape: tuple,
            num_channels: int = 1,
            device: str = "cpu",
            dtype: type = torch.float32,
    ) -> None:
        kwargs = {"mask_shape": mask_shape}
        if len(mask_shape) != 2:
            raise ValueError(
                "mask_shape must 2D. Add channels via num_channels parameter"
            )
        super().__init__(
            num_channels=num_channels,
            device=device,
            dtype=dtype,
            **kwargs,
        )

    def generate_mask(self, image_shape):
        # Create an all-ones tensor which will serve as the initial mask
        mask = RandomMask(max(image_shape[-2], image_shape[-1]))

        mask = torch.from_numpy(mask).float().unsqueeze(0)

        return mask.float()

    def step(self, batch_size: int = 1):
        r"""
        Generate a random motion blur PSF with parameters :math:`\sigma` and :math:`l`

        :param int batch_size: batch_size.
        :param float sigma: the standard deviation of the Gaussian Process
        :param float l: the length scale of the trajectory

        :return: dictionary with key **'filter'**: the generated PSF of shape `(batch_size, 1, psf_size[0], psf_size[1])`
        """

        batch_shape = (batch_size, self.num_channels, self.mask_shape[-2], self.mask_shape[-1])

        mask = self.generate_mask(batch_shape)

        return {
            "mask": mask.to(self.factory_kwargs["device"])
        }


class JPEGRandomCompressionGenerator(PhysicsGenerator):

    def __init__(
            self,
            device: str = "cpu",
            dtype: type = torch.float32,
            quality_min=1,
            quality_max=95,
            sigma_min=0.,
            sigma_max=0.5,
    ) -> None:
        kwargs = {"quality_min": quality_min, "quality_max": quality_max,
                  "sigma_min": sigma_min, "sigma_max": sigma_max}
        super().__init__(
            device=device,
            dtype=dtype,
            **kwargs,
        )

    def step(self, batch_size: int = 1):
        r"""
        Generate a random quality factor.
        """
        # Generate a random integer in the range [quality_min, quality_max]
        if self.quality_min == self.quality_max:
            quality = self.quality_min
        else:
            quality = torch.randint(self.quality_min, self.quality_max, (batch_size,))

        # Generate a random float in the range [sigma_min, sigma_max]
        sigma = torch.rand(batch_size) * (self.sigma_max - self.sigma_min) + self.sigma_min

        return {"quality": quality, "sigma": sigma}


class RandomGaussianBlurGenerator(PhysicsGenerator):

    def __init__(
            self,
            device: str = "cpu",
            dtype: type = torch.float32,
            sigma_min=0.0,
            sigma_max=10.0,
    ) -> None:
        kwargs = {"sigma_min": sigma_min, "sigma_max": sigma_max}
        super().__init__(
            device=device,
            dtype=dtype,
            **kwargs,
        )
        self.device = device

    def step(self, batch_size: int = 1):
        r"""
        Generate a random quality factor.
        """
        # Generate a random float in the range [sigma_min, sigma_max]
        sigma_blur = torch.rand(batch_size) * (self.sigma_max - self.sigma_min) + self.sigma_min
        filter = dinv.physics.blur.gaussian_blur(sigma=(sigma_blur, sigma_blur), angle=0.0).to(self.device)

        return {"filter": filter}


class VoidBlurGenerator(PhysicsGenerator):

    def __init__(
            self,
            device: str = "cpu",
            dtype: type = torch.float32,
    ) -> None:
        super().__init__(
            device=device,
            dtype=dtype,
        )
        self.device = device

    def step(self, batch_size: int = 1):
        r"""
        Generate a random quality factor.
        """
        kernel_torch = dirac((1, 1, 3, 3)).to(self.device)
        for i in range(batch_size-1):
            kernel_torch = torch.cat((kernel_torch, dirac((1, 1, 3, 3)).to(device)), dim=0)
        return {"filter": kernel_torch}




def cubic(x):
    absx = torch.abs(x)
    absx2 = absx**2
    absx3 = absx**3
    return (1.5 * absx3 - 2.5 * absx2 + 1) * ((absx <= 1).type_as(absx)) + (
        -0.5 * absx3 + 2.5 * absx2 - 4 * absx + 2
    ) * (((absx > 1) * (absx <= 2)).type_as(absx))

def calculate_weights_indices(
    in_length, out_length, scale, kernel, kernel_width, antialiasing
):
    if (scale < 1) and (antialiasing):
        # Use a modified kernel to simultaneously interpolate and antialias- larger kernel width
        kernel_width = kernel_width / scale

    # Output-space coordinates
    x = torch.linspace(1, out_length, out_length)

    # Input-space coordinates. Calculate the inverse mapping such that 0.5
    # in output space maps to 0.5 in input space, and 0.5+scale in output
    # space maps to 1.5 in input space.
    u = x / scale + 0.5 * (1 - 1 / scale)

    # What is the left-most pixel that can be involved in the computation?
    left = torch.floor(u - kernel_width / 2)

    # What is the maximum number of pixels that can be involved in the
    # computation?  Note: it's OK to use an extra pixel here; if the
    # corresponding weights are all zero, it will be eliminated at the end
    # of this function.
    P = math.ceil(kernel_width) + 2

    # The indices of the input pixels involved in computing the k-th output
    # pixel are in row k of the indices matrix.
    indices = left.view(out_length, 1).expand(out_length, P) + torch.linspace(
        0, P - 1, P
    ).view(1, P).expand(out_length, P)

    # The weights used to compute the k-th output pixel are in row k of the
    # weights matrix.
    distance_to_center = u.view(out_length, 1).expand(out_length, P) - indices
    # apply cubic kernel
    if (scale < 1) and (antialiasing):
        weights = scale * cubic(distance_to_center * scale)
    else:
        weights = cubic(distance_to_center)
    # Normalize the weights matrix so that each row sums to 1.
    weights_sum = torch.sum(weights, 1).view(out_length, 1)
    weights = weights / weights_sum.expand(out_length, P)

    # If a column in weights is all zero, get rid of it. only consider the first and last column.
    weights_zero_tmp = torch.sum((weights == 0), 0)
    if not math.isclose(weights_zero_tmp[0], 0, rel_tol=1e-6):
        indices = indices.narrow(1, 1, P - 2)
        weights = weights.narrow(1, 1, P - 2)
    if not math.isclose(weights_zero_tmp[-1], 0, rel_tol=1e-6):
        indices = indices.narrow(1, 0, P - 2)
        weights = weights.narrow(1, 0, P - 2)
    weights = weights.contiguous()
    indices = indices.contiguous()
    sym_len_s = -indices.min() + 1
    sym_len_e = indices.max() - in_length
    indices = indices + sym_len_s - 1
    return weights, indices, int(sym_len_s), int(sym_len_e)


def imresize(img, scale, antialiasing=True):
    # Now the scale should be the same for H and W
    # input: img: pytorch tensor, CHW or HW [0,1]
    # output: CHW or HW [0,1] w/o round

    if img.is_cuda:
        img = img.cpu()
    need_unsqueeze = True if img.dim() == 2 else False
    need_squeeze = True if img.dim() == 4 else False

    if need_unsqueeze:
        img.unsqueeze_(0)
    if need_squeeze:
        img.squeeze_(0)

    in_C, in_H, in_W = img.size()
    out_C, out_H, out_W = (
        in_C,
        math.ceil(in_H * scale),
        math.ceil(in_W * scale),
    )
    kernel_width = 4
    kernel = "cubic"

    # Return the desired dimension order for performing the resize.  The
    # strategy is to perform the resize first along the dimension with the
    # smallest scale factor.
    # Now we do not support this.

    # get weights and indices
    weights_H, indices_H, sym_len_Hs, sym_len_He = calculate_weights_indices(
        in_H, out_H, scale, kernel, kernel_width, antialiasing
    )
    weights_W, indices_W, sym_len_Ws, sym_len_We = calculate_weights_indices(
        in_W, out_W, scale, kernel, kernel_width, antialiasing
    )
    weights_H, indices_H = weights_H, indices_H
    weights_W, indices_W= weights_W, indices_W


    # process H dimension
    # symmetric copying
    img_aug = torch.FloatTensor(in_C, in_H + sym_len_Hs + sym_len_He, in_W)
    img_aug.narrow(1, sym_len_Hs, in_H).copy_(img)

    sym_patch = img[:, :sym_len_Hs, :]
    inv_idx = torch.arange(sym_patch.size(1) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(1, inv_idx)
    img_aug.narrow(1, 0, sym_len_Hs).copy_(sym_patch_inv)

    sym_patch = img[:, -sym_len_He:, :]
    inv_idx = torch.arange(sym_patch.size(1) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(1, inv_idx)
    img_aug.narrow(1, sym_len_Hs + in_H, sym_len_He).copy_(sym_patch_inv)

    out_1 = torch.FloatTensor(in_C, out_H, in_W)
    kernel_width = weights_H.size(1)
    for i in range(out_H):
        idx = int(indices_H[i][0])
        for j in range(out_C):
            out_1[j, i, :] = (
                img_aug[j, idx : idx + kernel_width, :].transpose(0, 1).mv(weights_H[i])
            )

    # process W dimension
    # symmetric copying
    out_1_aug = torch.FloatTensor(in_C, out_H, in_W + sym_len_Ws + sym_len_We)
    out_1_aug.narrow(2, sym_len_Ws, in_W).copy_(out_1)

    sym_patch = out_1[:, :, :sym_len_Ws]
    inv_idx = torch.arange(sym_patch.size(2) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(2, inv_idx)
    out_1_aug.narrow(2, 0, sym_len_Ws).copy_(sym_patch_inv)

    sym_patch = out_1[:, :, -sym_len_We:]
    inv_idx = torch.arange(sym_patch.size(2) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(2, inv_idx)
    out_1_aug.narrow(2, sym_len_Ws + in_W, sym_len_We).copy_(sym_patch_inv)

    out_2 = torch.FloatTensor(in_C, out_H, out_W)
    kernel_width = weights_W.size(1)
    for i in range(out_W):
        idx = int(indices_W[i][0])
        for j in range(out_C):
            out_2[j, :, i] = out_1_aug[j, :, idx : idx + kernel_width].mv(weights_W[i])
    if need_unsqueeze:
        out_2.squeeze_()
    if need_squeeze:
        out_2.unsqueeze_(0)
    return out_2


class PhysicsRescale(dinv.physics.LinearPhysics):
    r"""
    A different implementation of the down/upsampling operator using imresize with antialiasing.
    """
    def __init__(self, img_size, factor, device='cpu', filter='bicubic', noise_model=None):
        super().__init__(img_size=img_size, device=device, noise_model=noise_model)
        self.factor = factor
        self.filter = filter
        self.device = device

    def A(self, x, **kwargs):
        out_cpu = imresize(x, 1/self.factor, antialiasing=True)
        return out_cpu.to(self.device)

    def A_adjoint(self, y, **kwargs):
        out_cpu = imresize(y, self.factor, antialiasing=True)
        return out_cpu.to(self.device)