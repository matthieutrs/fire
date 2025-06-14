import numpy as np

import deepinv as dinv
import json

import torch.nn

from simple_lama_inpainting import LamaRestorer
from restoration.gradient_descent import GradientDescentModel, RestModel, SHARP
from restoration.physics import InpaintingRandomMaskGenerator, JPEGTransform, JPEGRandomCompressionGenerator, RandomGaussianBlurGenerator, InpaintingBrushGenerator, VoidBlurGenerator, PhysicsRescale

from deepinv.optim import optim_builder, PnP, L2
from deepinv.physics.generator import RandomMaskGenerator, MotionBlurGenerator, DiffractionBlurGenerator, GeneratorMixture, SigmaGenerator


with open('config/config.json') as f:
    config = json.load(f)

ROOT_CKPT = config['ROOT_CKPT']


class LamaRestorerDeepinv(torch.nn.Module):
    def __init__(self, device='cpu'):
        super(LamaRestorerDeepinv, self).__init__()
        self.lama = LamaRestorer(model_path=ROOT_CKPT + '/big-lama.pt').to(device)


    def forward(self, y, physics):
        mask = physics.mask
        return self.lama(y, 1-mask)  # and see here too for instance


class IdentityDeepinv(torch.nn.Module):
    def __init__(self, device='cpu'):
        super(IdentityDeepinv, self).__init__()


    def forward(self, y, physics=None):
        return y


class DRUNetBlind(torch.nn.Module):
    def __init__(self, device='cpu', sigma=0.05):
        super(DRUNetBlind, self).__init__()
        self.drunet = dinv.models.DRUNet(in_channels=3, out_channels=3, device=device,
                                        pretrained=ROOT_CKPT + '/drunet_color.pth')
        self.sigma = sigma
        self.device = device

    def forward(self, y, sigma=None, physics=None):
        if sigma is None:
            sigma = self.sigma
        if physics is not None:
            sigma = physics.noise_model.sigma.float()
            print('Choosing sigma from physics: ', sigma)
        return self.drunet(y, sigma)


def get_LAMA(device='cpu'):
    return LamaRestorerDeepinv(device=device)


class BlindRestorationModel(torch.nn.Module):
    def __init__(self, model, device='cpu'):
        super(BlindRestorationModel, self).__init__()
        self.model = model
        self.device = device

    def forward(self, y, physics=None):
        return self.model(y)


def get_backbone(backbone='lama', device='cpu'):
    if 'lama' in backbone.lower():
        model = LamaRestorer(model_path=ROOT_CKPT + '/big-lama.pt').to(device)
    elif 'restormer' in backbone.lower():
        model = dinv.models.Restormer(pretrained=ROOT_CKPT + '/single_image_defocus_deblurring.pth',
                                      LayerNorm_type='WithBias').to(device)
        if 'gaussian' in backbone.lower():
            model.load_state_dict(torch.load(ROOT_CKPT + '/restormer_gaussian.pth'))
        elif 'motion' in backbone.lower():
            model.load_state_dict(torch.load(ROOT_CKPT + '/restormer_motion.pth'))
        else:
            print('WARNING: LOADING DEFAULT RESTORMER MODEL')
        model = dinv.models.ArtifactRemoval(model, pinv=False, device=device)
    elif 'swinir' in backbone.lower():
        scale = int(backbone.split('_')[-1].split('x')[0])
        model_path = ROOT_CKPT + '/SwinIR_'+str(scale)+'x.pth'
        model = dinv.models.SwinIR(upscale=scale, in_chans=3, img_size=64, window_size=8,
                                  img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                                  mlp_ratio=2, upsampler='pixelshuffle', resi_connection='1conv', pretrained=model_path)
        model = BlindRestorationModel(model)
    elif 'drunet' in backbone.lower():
        # model = dinv.models.DRUNet(in_channels=3, out_channels=3, device=device,
        #                           pretrained='/gpfsstore/rech/tbo/ubd63se/checkpoints/drunet_color.pth')
        model = DRUNetBlind(device=device)
    elif 'scunet' in backbone.lower():
        model = dinv.models.SCUNet(device=device,
                                   pretrained=ROOT_CKPT + '/scunet_color_real_psnr.pth')
        model = BlindRestorationModel(model)
    elif 'identity' in backbone.lower():
        model = IdentityDeepinv(device=device)
    else:
        raise ValueError(f'Backbone {backbone} not recognized')
    print('LOADED BACKBONE:', backbone)  # Debug
    return model.to(device)


def get_model_and_physics(model_name, img_size=(1, 64, 64), device='cpu', sigma_blur=2.0,
                          quality_min=0.2, quality_max=0.8, sigma_noise_min=0.01, sigma_noise_max=0.1,
                          p_mask_min=0.1, p_mask_max=0.9, sigma_blur_min=0.001, sigma_blur_max=0.001,
                          l_value=0.9, sigma_blur_value=0.9,
                          motion_sigma_noise=0.01,
                          root_pth='/gpfsstore/rech/tbo/ubd63se/checkpoints'):

    physics_generator = None
    pad_multiple = 1
    average_last = True

    model = get_backbone(model_name, device=device)
    if 'swinir' in model_name.lower():
        scale = int(model_name.split('_')[-1].split('x')[0])  # for example 'swinir_3x'
        # physics = dinv.physics.Downsampling(img_size=img_size, factor=scale, device=device, filter='bicubic',
        #                                     noise_model=dinv.physics.GaussianNoise(sigma=0.))
        physics = PhysicsRescale(img_size=img_size, factor=scale, device=device,
                                 filter='bicubic', noise_model=dinv.physics.GaussianNoise(sigma=0.))
        physics_generator = SigmaGenerator(device=device, sigma_min=0.03, sigma_max=0.06)
        pad_multiple = scale
        average_last = True
    elif 'restormer' in model_name.lower():
        if 'motion' in model_name.lower():
            if l_value > 0. and sigma_blur_value > 0.:
                psf_size = 31
                physics_generator = MotionBlurGenerator(
                    (psf_size, psf_size), l=l_value, sigma=sigma_blur_value, device=device,
                )
            else:
                physics_generator = VoidBlurGenerator(device=device)
            filters = physics_generator.step(batch_size=1)
            kernel_torch = filters['filter']
            physics = dinv.physics.BlurFFT(
                img_size=(3, img_size[-2], img_size[-1]),
                filter=kernel_torch,
                device=device,
                noise_model=dinv.physics.GaussianNoise(sigma=motion_sigma_noise),  # TODO HERE ADD CIRCULAR PADDING !!
            )
        else:
            if sigma_blur_max > 0.:
                physics_generator = RandomGaussianBlurGenerator(
                    sigma_min=sigma_blur_min, sigma_max=sigma_blur_max, device=device
                )
            else:
                physics_generator = VoidBlurGenerator(device=device)
            filters = physics_generator.step(batch_size=1)
            kernel_torch = filters['filter']

            physics = dinv.physics.Blur(kernel_torch,
                                        device=device,
                                        noise_model=dinv.physics.GaussianNoise(sigma=motion_sigma_noise),
                                        padding='circular')

    elif 'drunet' in model_name.lower() and not 'naive' in model_name.lower():
        physics = dinv.physics.DecomposablePhysics(noise_model=dinv.physics.GaussianNoise(sigma=sigma_noise_max))
        model.sigma = sigma_noise_max  # beware, may have been discrepancy here
        average_last = False
    elif 'drunet' in model_name.lower() and 'naive' in model_name.lower():
        physics = dinv.physics.DecomposablePhysics(noise_model=dinv.physics.GaussianNoise(sigma=0.0))
        model.sigma = sigma_noise_max  # beware, may have been discrepancy here
        average_last = False
    elif 'lama' in model_name.lower():
        physics = dinv.physics.DecomposablePhysics(noise_model=dinv.physics.GaussianNoise(sigma=sigma_noise_max))
        if 'brush' in model_name.lower():
            physics_generator = InpaintingBrushGenerator(mask_shape=(img_size[-2], img_size[-1]), device=device)
        else:
            physics_generator = InpaintingRandomMaskGenerator(mask_shape=(img_size[-2], img_size[-1]), p_min=p_mask_min,
                                                              p_max=p_mask_max, device=device)

    elif 'scunet_jpeg' in model_name.lower():
        physics = JPEGTransform(quality=100, sigma=0., device=device)
        physics_generator = JPEGRandomCompressionGenerator(device=device, quality_min=quality_min,
                                                           quality_max=quality_max,
                                                           sigma_min=sigma_noise_min, sigma_max=sigma_noise_max)

    elif 'scunet_denoise' in model_name.lower():
        physics = JPEGTransform(quality=100, sigma=0., device=device)
        physics_generator = JPEGRandomCompressionGenerator(device=device, quality_min=100, quality_max=100,
                                                           sigma_min=sigma_noise_min, sigma_max=sigma_noise_max)

    elif 'identity' in model_name.lower():
        # Identity physics
        physics = dinv.physics.DecomposablePhysics(noise_model=dinv.physics.GaussianNoise(sigma=0.))
        physics_generator = SigmaGenerator(device=device, sigma_min=0., sigma_max=0.)

    return model, physics, pad_multiple, physics_generator, average_last


def parse_model_name(model_name):
    # Split the string by underscore
    parts = model_name.split('_')

    # Remove 'GD' and 'multi' if they're present
    parts = [part for part in parts if part not in ['GD', 'multi', 'SHARP', 'DRP']]

    # Combine 'restormer' and 'gaussian' if they're present
    if 'restormer' in parts and 'gaussian' in parts:
        index = parts.index('restormer')
        parts[index] = 'restormer_gaussian'
        parts.remove('gaussian')

    # Combine 'restormer' and 'gaussian' if they're present
    if 'restormer' in parts and 'motion' in parts:
        index = parts.index('restormer')
        parts[index] = 'restormer_motion'
        parts.remove('motion')


    # Combine 'restormer' and 'gaussian' if they're present
    if 'scunet' in parts and 'jpeg' in parts:
        index = parts.index('scunet')
        parts[index] = 'scunet_jpeg'
        parts.remove('jpeg')

    if 'scunet' in parts and 'denoise' in parts:
        index = parts.index('scunet')
        parts[index] = 'scunet_denoise'
        parts.remove('denoise')

    if 'swinir' in parts and '2x' in parts:
        index = parts.index('swinir')
        parts[index] = 'swinir_2x'
        parts.remove('2x')

    if 'swinir' in parts and '3x' in parts:
        index = parts.index('swinir')
        parts[index] = 'swinir_3x'
        parts.remove('3x')

    return parts

def get_model(model_name, device='cpu', channels=3, num_blocks=2000, img_size=(1, 64, 64),
              sigma_noise_min=0.01, sigma_noise_max=0.1, quality_min=20, quality_max=80,
              p_mask_min=0.1, p_mask_max=0.9, l_value=0.9, sigma_blur_value=0.9, motion_sigma_noise=0.01,
              avg_gdsteps=1, sigma_blur_min=0.001, sigma_blur_max=0.001, max_iter=50, lambd=20., gamma=1.0,
              equivariant=True, noise_level_img=0.01, list_gamma_values=None, special_physics=None,
              special_model_name=None, special_pad_multiple=1):

    if 'GD' in model_name:
        kwargs = {'sigma_noise_min': sigma_noise_min, 'sigma_noise_max': sigma_noise_max, 'quality_min': quality_min,
                  'quality_max': quality_max, 'p_mask_min': p_mask_min, 'p_mask_max': p_mask_max,
                  'l_value': l_value, 'sigma_blur_value': sigma_blur_value, 'sigma_blur_min': sigma_blur_min,
                  'sigma_blur_max': sigma_blur_max, 'motion_sigma_noise': motion_sigma_noise}

        if not 'multi' in model_name.lower(): ## dirty hack for now
            model, physics, pad_multiple, physics_generator, average_last = get_model_and_physics(model_name,
                                                                                                  img_size=img_size,
                                                                                                  device=device,
                                                                                                  **kwargs)
            apply_adjoint = True if 'restormer' in model_name.lower() else False
            stochastic = False if 'drunet' in model_name.lower() else True

            restmodel = RestModel(model, physics, device=device, pad_multiple=pad_multiple,
                                  physics_generator=physics_generator, stochastic=stochastic,
                                  apply_adjoint=apply_adjoint, model_name=model_name)

        else:

            list_model_name = parse_model_name(model_name)

            restmodel = []
            for model_name_cur in list_model_name:
                model_cur, physics_cur, pad_multiple_cur, physics_generator_cur, average_last = get_model_and_physics(model_name_cur,
                                                                                                                        img_size=img_size,
                                                                                                                        device=device,
                                                                                                                        **kwargs)
                stochastic = False if 'drunet' in model_name_cur.lower() else True
                apply_adjoint = True if 'restormer' in model_name_cur.lower() else False

                rest_model_cur = RestModel(model_cur, physics_cur, device=device, pad_multiple=pad_multiple_cur,
                                           physics_generator=physics_generator_cur, stochastic=stochastic,
                                           apply_adjoint=apply_adjoint, model_name=model_name_cur)

                restmodel.append(rest_model_cur)

            if special_physics is not None:
                special_model = get_backbone(special_model_name, device=device)
                rest_model_cur = RestModel(special_model, special_physics, device=device, pad_multiple=special_pad_multiple,
                                           physics_generator=None, stochastic=False)

                restmodel.append(rest_model_cur)

        if not 'SHARP' in model_name:
            model = GradientDescentModel(restmodel, max_iter=max_iter, lambd=lambd, equivariant=equivariant,
                                         apply_adjoint=False, gamma=gamma, average_gradient_steps=avg_gdsteps,
                                         init_pinv=False, average_last=average_last,
                                         list_gamma_values=list_gamma_values).to(device)
        else:
            model = SHARP(restmodel, max_iter=max_iter, lambd=lambd, equivariant=equivariant,
                                         apply_adjoint=False, gamma=gamma, average_gradient_steps=avg_gdsteps,
                                         init_pinv=False, average_last=average_last,
                                         list_gamma_values=list_gamma_values).to(device)

    else:
        raise ValueError('Model not recognized')

    print('Loading ', model_name, ' successful')

    return model
