import os
from argparse import ArgumentParser

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F

import deepinv as dinv
from deepinv.loss.metric import PSNR, SSIM, LPIPS

from restoration.utils import get_data, to_image, set_seed
from restoration.physics import get_physics
from restoration.models import get_model, get_backbone


def pad_function(x, multiple=8):
    # Get current dimensions
    _, _, h, w = x.shape

    # Calculate padding needed
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple

    # Calculate padding for each side
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    # Apply padding
    x_padded = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)

    # Return padded tensor and padding values
    return x_padded, (pad_top, pad_left)

def run_experiment_restoration(problem_type='sr', dataset_name='set3c', img_size=None, results_folder='results/',
                               device='cpu', model_name='dncnn', num_test_samples=None, max_iter=100, lambd=20, gamma=1.0,
                               n_channels=3, sr=2, sigma=0.01, slice_idx=None, debug=False, num_blocks=2000,
                               sigma_noise_min=0.0, sigma_noise_max=0.0, quality_min=100, quality_max=100,
                               p_mask_min=0.1, avg_gdsteps=1, p_mask_max=0.9, l_value=0.01, sigma_blur_value=0.01,
                               motion_sigma_noise=0.01, sigma_blur_min=0.01, sigma_blur_max=0.01, equivariant=True,
                               list_gamma_values=None, return_all=False, beta=0.5):

    # keep_logs = True if 'GD' in model_name else False
    if 'GD' in model_name and not 'DRP' in model_name:
        keep_logs = True
    else:
        keep_logs = False

    # torch.manual_seed(0)  # Reproducibility
    set_seed(0)

    # padding = 4 if ('DRP' in model_name or '3x' in model_name) else None
    # padding = 4 if 'DRP' in model_name else None  # Ideally we want to get rid of that and pad directly in the forward
    # padding = 4 if 'swinir' in model_name else padding
    padding = None

    test_dataloader = get_data(dataset_name=dataset_name, img_size=img_size, n_channels=n_channels, padding=padding)

    if num_test_samples is not None:
        num_test_samples_max = min(num_test_samples, len(test_dataloader))
    else:
        num_test_samples_max = len(test_dataloader)

    psnr_fun = PSNR()
    ssim_fun = SSIM()
    lpips_fun = LPIPS(device=device)

    psnr_array = np.zeros((num_test_samples_max, 3))
    psnr_array_u = np.zeros((num_test_samples_max, 3))

    with open(results_folder + 'metrics.txt', 'w+') as f:  # 'a' mode to append to an existing file or create a new one if it doesn't exist
        f.write(f"PSNR, SSIM, LPIPS\n")

    for num_slice, batch in enumerate(test_dataloader):

        if slice_idx is not None and num_slice != slice_idx:
            continue

        print(f"Running slice {num_slice}")
        # Free memory
        torch.cuda.empty_cache()

        x_true, _ = batch
        x_true = x_true.to(device)


        if 'diffpir' in model_name.lower() and x_true.shape[-1] > 800:
            continue

        multiple = 8
        if 'drp' in model_name.lower() and not 'drp_inpainting' in model_name.lower():
            multiple = 12
        if 'diffpir' in model_name.lower() and x_true.shape[-1] < 800:
            multiple = 128
        if 'swinir_3x' in model_name.lower():
            multiple = 12

        x_true, (pad_top, pad_left) = pad_function(x_true, multiple=multiple)  # pad the image to have a size multiple of 8
        x_true = x_true.to(device)

        img_size = x_true.shape[-2:]

        print('DEBUGGING SHAPES : x_true ', x_true.shape, img_size)

        model = get_model(model_name, device=device, channels=n_channels, num_blocks=num_blocks,
                          img_size=(1, img_size[0], img_size[1]),
                          sigma_noise_min=sigma_noise_min, sigma_noise_max=sigma_noise_max,
                          quality_min=quality_min, quality_max=quality_max,
                          p_mask_min=p_mask_min, p_mask_max=p_mask_max, avg_gdsteps=avg_gdsteps,
                          l_value=l_value, sigma_blur_value=sigma_blur_value, motion_sigma_noise=motion_sigma_noise,
                          sigma_blur_min=sigma_blur_min, sigma_blur_max=sigma_blur_max,
                          max_iter=max_iter, lambd=lambd, gamma=gamma, equivariant=equivariant,
                          list_gamma_values=list_gamma_values)



        physics = get_physics(x_true, n_channels, problem_type=problem_type, device=device, sr=sr, img_size=img_size,
                              noise_level=sigma)

        if 'swinir' in model_name.lower():
            model = get_model(model_name, device=device, channels=n_channels, num_blocks=num_blocks,
                              img_size=(1, x_true.shape[-2], x_true.shape[-1]),
                              sigma_noise_min=sigma_noise_min, sigma_noise_max=sigma_noise_max,
                              quality_min=quality_min, quality_max=quality_max,
                              p_mask_min=p_mask_min, p_mask_max=p_mask_max, avg_gdsteps=avg_gdsteps,
                              l_value=l_value, sigma_blur_value=sigma_blur_value, motion_sigma_noise=motion_sigma_noise,
                              sigma_blur_min=sigma_blur_min, sigma_blur_max=sigma_blur_max,
                              max_iter=max_iter, lambd=lambd, gamma=gamma, equivariant=equivariant,
                              list_gamma_values=list_gamma_values)

        y = physics(x_true)

        print('DEBUGGING SHAPES : y ', y.shape)

        if 'GD' in model_name:
            if 'inpainting' in problem_type:
                init_model = get_backbone(special_model_name, device=device)
                x_init = init_model(y, physics=physics)
                last_model = get_backbone('scunet', device=device)
            elif 'sr' in problem_type.lower():
                last_model = None
                x_init = physics.A_dagger(y)
                print('X init with pinv : x_init ', x_init.shape)
            else:
                last_model = None
                x_init = None
        else:
            last_model = None
            x_init = None

        backproj = physics.A_adjoint(y)

        print('DEBUGGING SHAPES : backproj ', backproj.shape)

        plt.imsave(results_folder+'backproj_'+str(num_slice)+'.png', to_image(backproj)[0].cpu().numpy(), cmap='viridis')
        plt.imsave(results_folder+'obs_'+str(num_slice)+'.png', to_image(y)[0].cpu().numpy(), cmap='viridis')
        plt.imsave(results_folder+'target_'+str(num_slice)+'.png', to_image(x_true[..., pad_top:-pad_top or None, pad_left:-pad_left or None])[0].cpu().numpy(), cmap='viridis')
        # plt.imsave(results_folder+'check_init_'+str(num_slice)+'.png', to_image(x_init[..., pad_top:-pad_top or None, pad_left:-pad_left or None])[0].cpu().numpy(), cmap='viridis')

        # if not debug:
        with torch.no_grad():

            if keep_logs:
                if not return_all:
                    if 'sharp' in model_name.lower():
                        x, u, logs = model(y, physics, x_true=x_true, x_init=x_init, last_model=last_model, beta=beta)
                    else:
                        x, u, logs = model(y, physics, x_true=x_true, x_init=x_init, last_model=last_model)
                else:
                    x, u, logs, list_x_k, list_u_k, list_uin_k = model(y, physics, x_true=x_true, x_init=x_init, last_model=last_model, return_all=return_all)
                cost, log_psnr, log_psnr_u = logs['cost'], logs['psnr'], logs['psnr_u']
            else:
                x = model(y, physics)
                cost, log_psnr, log_psnr_u, u = None, None, None, None

            print('Ran exp with x.device ', x.device)

            print(x_true.shape)
            print(x.shape)

            x = x[..., pad_top:-pad_top or None, pad_left:-pad_left or None]  # undo padding
            x_true = x_true[..., pad_top:-pad_top or None, pad_left:-pad_left or None]  # undo padding
            if u is not None:
                u = u[..., pad_top:-pad_top or None, pad_left:-pad_left or None]  # undo padding

            if x_init is not None:
                x_init = x_init[..., pad_top:-pad_top or None, pad_left:-pad_left or None]

            if padding is not None:
                x = x[:, :, padding:-padding, padding:-padding]
                x_true = x_true[:, :, padding:-padding, padding:-padding]
                u = u[:, :, padding:-padding, padding:-padding]


            if return_all:
                list_x_k = [x[..., pad_top:-pad_top or None, pad_left:-pad_left or None] for x in list_x_k]
                list_u_k = [u[..., pad_top:-pad_top or None, pad_left:-pad_left or None] for u in list_u_k]
                list_uin_k = [u[..., pad_top:-pad_top or None, pad_left:-pad_left or None] for u in list_uin_k]
                if padding is not None:
                    list_x_k = [x[:, :, padding:-padding, padding:-padding] for x in list_x_k]
                    list_u_k = [u[:, :, padding:-padding, padding:-padding] for u in list_u_k]
                    list_uin_k = [u[:, :, padding:-padding, padding:-padding] for u in list_uin_k]

                for k, x_k in enumerate(list_x_k):
                    plt.imsave(results_folder + 'x_' + str(k) + '_' + str(num_slice) + '.png', to_image(x_k)[0].cpu().numpy(),
                               cmap='viridis')
                for k, u_k in enumerate(list_u_k):
                    plt.imsave(results_folder + 'u_' + str(k) + '_' + str(num_slice) + '.png', to_image(u_k)[0].cpu().numpy(),
                               cmap='viridis')
                for k, uin_k in enumerate(list_uin_k):
                    plt.imsave(results_folder + 'uin_' + str(k) + '_' + str(num_slice) + '.png', to_image(uin_k)[0].cpu().numpy(),
                               cmap='viridis')



            # print(x_true.shape)
            # print(x.shape)

            # Compute metrics
            psnr = psnr_fun(x_true, x).item()
            ssim = ssim_fun(x_true, x).item()
            lpips = lpips_fun(x_true.to('cpu'), x.to('cpu')).item()

            if u is not None:
                psnr_u = psnr_fun(x_true, u).item()
                ssim_u = ssim_fun(x_true, u).item()
                lpips_u = lpips_fun(x_true.to('cpu'), u.to('cpu')).item()

        if log_psnr is not None:
            # plot cost values and save it
            plt.figure()
            plt.plot(cost)
            plt.xlabel('Iterations')
            plt.ylabel('Cost')
            plt.title('Cost function')
            plt.savefig(results_folder+'cost_'+str(num_slice)+'.png')

            plt.figure()
            plt.plot(log_psnr)
            plt.xlabel('Iterations')
            plt.ylabel('PSNR')
            plt.title('PSNR')
            plt.savefig(results_folder+'psnr_'+str(num_slice)+'.png')

            # save the array of cost values and psnr values in txt file
            np.savetxt(results_folder+'cost_'+str(num_slice)+'.txt', cost)
            np.savetxt(results_folder+'psnr_'+str(num_slice)+'.txt', log_psnr)
            np.savetxt(results_folder+'psnr_u_'+str(num_slice)+'.txt', log_psnr_u)

        plt.imsave(results_folder+'rec_'+str(num_slice)+'.png', to_image(x)[0].cpu().numpy(), cmap='viridis')
        if u is not None:
            plt.imsave(results_folder+'u_'+str(num_slice)+'.png', to_image(u)[0].cpu().numpy(), cmap='viridis')
        if x_init is not None:
            plt.imsave(results_folder+'init_'+str(num_slice)+'.png', to_image(x_init)[0].cpu().numpy(), cmap='viridis')

        cur_metrics = np.array([psnr, ssim, lpips])

        if u is not None:
            cur_metrics_u = np.array([psnr_u, ssim_u, lpips_u])

        if slice_idx is None:
            psnr_array[num_slice, :] = cur_metrics
            if u is not None:
                psnr_array_u[num_slice, :] = cur_metrics_u
        else:
            num_slice = 0
            psnr_array[0, :] = cur_metrics
            if u is not None:
                psnr_array_u[0, :] = cur_metrics_u

        # Open the file in append mode or write mode
        with open(results_folder+'metrics.txt', 'a') as f:  # 'a' mode to append to an existing file or create a new one if it doesn't exist
            f.write(f"{psnr_array[num_slice, :]}\n")

        if u is not None:
            with open(results_folder+'metrics_u.txt', 'a') as f:  # 'a' mode to append to an existing file or create a new one if it doesn't exist
                f.write(f"{psnr_array_u[num_slice, :]}\n")

        # delete unused variables to free memory
        del model, physics, x, x_true, y, cur_metrics, cost, log_psnr, backproj

        if u is not None:
            del log_psnr_u, u, cur_metrics_u

        if num_test_samples is not None and num_slice >= num_test_samples-1:
            break

    # Compute mean and std
    metrics = np.mean(psnr_array, axis=0)
    metrics_std = np.std(psnr_array, axis=0)

    metrics_u = np.mean(psnr_array_u, axis=0)
    metrics_u_std = np.std(psnr_array_u, axis=0)

    with open(results_folder+'metrics.txt', 'a') as f:
        f.write(f"Mean: {metrics[0]:.2f} & {metrics[1]:.2f} & {metrics[2]:.2f}\n")
        f.write(f"Std: {metrics_std[0]:.2f} & {metrics_std[1]:.2f} & {metrics_std[2]:.2f}\n")

    with open(results_folder+'metrics_u.txt', 'a') as f:
        f.write(f"Mean: {metrics_u[0]:.2f} & {metrics_u[1]:.2f} & {metrics_u[2]:.2f}\n")
        f.write(f"Std: {metrics_u_std[0]:.2f} & {metrics_u_std[1]:.2f} & {metrics_u_std[2]:.2f}\n")


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--method_name', type=str, default='DPIR')
    parser.add_argument('--dataset_name', type=str, default='set3c')
    parser.add_argument('--results_folder', type=str, default='results/')
    parser.add_argument('--noise_level', type=float, default=0.01)
    parser.add_argument('--max_iter', type=int, default=60)
    parser.add_argument('--lambd', type=float, default=20)
    parser.add_argument('--gamma', type=float, default=1.0)
    parser.add_argument('--beta', type=float, default=1.0)  # beta value for SHARP algorithm
    parser.add_argument('--avg_gdsteps', type=int, default=1)
    parser.add_argument('--sigma_noise_min', type=float, default=0.0)
    parser.add_argument('--sigma_noise_max', type=float, default=0.0)
    parser.add_argument('--p_mask_min', type=float, default=0.0)
    parser.add_argument('--p_mask_max', type=float, default=0.0)
    parser.add_argument('--quality_min', type=int, default=100)
    parser.add_argument('--quality_max', type=int, default=100)
    parser.add_argument('--l_value', type=float, default=0.9)
    parser.add_argument('--sigma_blur_value', type=float, default=0.01)
    parser.add_argument('--sigma_blur_min', type=float, default=0.01)
    parser.add_argument('--sigma_blur_max', type=float, default=0.01)
    parser.add_argument('--motion_sigma_noise', type=float, default=0.01)
    parser.add_argument('--slice_idx', type=float, default=-1)
    parser.add_argument('--problem', type=str, default='noisy_inpainting')
    parser.add_argument('--compute_lip', type=int, default=0)
    parser.add_argument('--num_blocks', type=int, default=2000)
    parser.add_argument('--list_gamma_values', nargs='+', type=float, default=None)
    parser.add_argument('--equivariant', type=int, default=1)
    parser.add_argument('--return_all', type=int, default=0)
    args = parser.parse_args()

    device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"

    equivariant = True if args.equivariant == 1 else False
    equivariant_str = 'eq' if equivariant else 'noneq'

    if 'DYS' or 'SYD' in args.method_name:
        str_blocks = str(args.num_blocks)
    else:
        str_blocks = ''

    if 'lama' in args.method_name.lower():
        str_blocks = '2000'
    else:
        str_blocks = ''

    if 'GD' in args.method_name and not 'identity' in args.method_name.lower():
        str_model = args.method_name
        str_params = ''
        if 'scunet' in args.method_name.lower():
            str_params = str_params + ('_sigma_min_' + str(args.sigma_noise_min) + '_sigma_max_' + str(args.sigma_noise_max)
                         + '_quality_min_' + str(args.quality_min) + '_quality_max_' + str(args.quality_max))
        if 'lama' in args.method_name.lower():
            str_params = str_params + ('_p_min_' + str(args.p_mask_min) + '_p_max_' + str(args.p_mask_max) +
                         '_sigmanoisemax_' + str(args.sigma_noise_max))
        if 'restormer' in args.method_name.lower():
            if 'motion' in args.method_name.lower():
                str_params = str_params + ( '_lvalue_' + str(args.l_value) + '_sigmablurvalue_'
                             + str(args.sigma_blur_value) + '_motionsigmanoise_' + str(args.motion_sigma_noise))
            elif 'gaussian' in args.method_name.lower():
                str_params = str_params + ('_sigmablurmin_' + str(args.sigma_blur_min) + '_sigmablurmax_'
                             + str(args.sigma_blur_max) + '_motionsigmanoise_' + str(args.motion_sigma_noise))
        if 'drunet' in args.method_name.lower():
            str_params = str_params + (args.method_name + '_sigma_max_' + str(args.sigma_noise_max))

        str_list_gamma_values = '_'.join(map(str, args.list_gamma_values)) if args.list_gamma_values is not None else str(args.gamma)
        #
        # str_params = str_params + '_gamma_' + str_list_gamma_values

        if not 'sharp' in str_model.lower():
            str_model = (str_model + str_params + '_avg_gdsteps_' + str(args.avg_gdsteps) + '_maxiter_' + str(args.max_iter) + '_lambda_' + str(args.lambd)
                     + '_gamma_' + str_list_gamma_values + '_' + equivariant_str)
        else:
            str_model = (str_model + str_params + '_avg_gdsteps_' + str(args.avg_gdsteps) + '_maxiter_' + str(args.max_iter) + '_beta_' + str(args.beta)
                     + '_gamma_' + str_list_gamma_values + '_' + equivariant_str)
    else:
        str_model = args.method_name

    results_folder = (args.results_folder + args.problem + '_' + args.dataset_name +
                      '_' + str(args.noise_level) + '/' + str_model + '/')

    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    slice_idx = None if args.slice_idx < 0 else int(args.slice_idx)
    return_all = True if args.return_all == 1 else False

    run_experiment_restoration(dataset_name=args.dataset_name, problem_type=args.problem,
                               results_folder=results_folder,
                               device=device, model_name=args.method_name,
                               sigma=args.noise_level, slice_idx=slice_idx,
                               max_iter=args.max_iter, lambd=args.lambd, gamma=args.gamma,
                               debug=True, num_blocks=args.num_blocks,
                               sigma_noise_min=args.sigma_noise_min, sigma_noise_max=args.sigma_noise_max,
                               quality_min=args.quality_min, quality_max=args.quality_max,
                               p_mask_min=args.p_mask_min, p_mask_max=args.p_mask_max, avg_gdsteps=args.avg_gdsteps,
                               l_value=args.l_value, sigma_blur_value=args.sigma_blur_value,
                               motion_sigma_noise=args.motion_sigma_noise,
                               sigma_blur_min=args.sigma_blur_min, sigma_blur_max=args.sigma_blur_max,
                               equivariant=equivariant, list_gamma_values=args.list_gamma_values,
                               return_all=return_all, beta=args.beta)

