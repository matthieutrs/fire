import numpy as np
import torch

import deepinv as dinv
from deepinv.optim.data_fidelity import L2

class Shift(torch.nn.Module):
    r"""
    Fast integer 2D translations.

    Generates n_transf randomly shifted versions of 2D images with circular padding.

    :param n_trans: number of shifted versions generated per input image.
    :param float shift_max: maximum shift as fraction of total height/width.
    """

    def __init__(self, n_trans=1, shift_max=1.0):
        super(Shift, self).__init__()
        self.n_trans = n_trans
        self.shift_max = shift_max

    def forward(self, x):
        r"""
        Applies a random translation to the input image.

        :param torch.Tensor x: input image
        :return: torch.Tensor containing the translated images concatenated along the first dimension
        """
        H, W = x.shape[-2:]
        assert self.n_trans <= H - 1 and self.n_trans <= W - 1

        H_max, W_max = int(self.shift_max * H), int(self.shift_max * W)

        x_shift = (
            torch.arange(-H_max, H_max)[torch.randperm(2 * H_max)][: self.n_trans]
            if H_max > 0
            else torch.zeros(self.n_trans)
        )
        y_shift = (
            torch.arange(-W_max, W_max)[torch.randperm(2 * W_max)][: self.n_trans]
            if W_max > 0
            else torch.zeros(self.n_trans)
        )

        out = torch.cat(
            [torch.roll(x, [sx, sy], [-2, -1]) for sx, sy in zip(x_shift, y_shift)],
            dim=0,
        )
        return out


    def A(self, x):
        r"""
        Applies a random translation to the input image.

        :param torch.Tensor x: input image
        :return: torch.Tensor containing the translated images concatenated along the first dimension
        """
        H, W = x.shape[-2:]
        assert self.n_trans <= H - 1 and self.n_trans <= W - 1

        H_max, W_max = int(self.shift_max * H), int(self.shift_max * W)
        # H_max, W_max = 12, 12  # Attempt

        x_shift = (
            torch.arange(-H_max, H_max)[torch.randperm(2 * H_max)][: self.n_trans]
            if H_max > 0
            else torch.zeros(self.n_trans)
        )
        y_shift = (
            torch.arange(-W_max, W_max)[torch.randperm(2 * W_max)][: self.n_trans]
            if W_max > 0
            else torch.zeros(self.n_trans)
        )

        out = torch.cat(
            [torch.roll(x, [sx, sy], [-2, -1]) for sx, sy in zip(x_shift, y_shift)],
            dim=0,
        )
        return out, x_shift, y_shift

    def Ainv(self, x, x_shift, y_shift):
        r"""
        Applies a random translation to the input image.

        :param torch.Tensor x: input image
        :return: torch.Tensor containing the translated images concatenated along the first dimension
        """
        out = torch.cat(
            [torch.roll(x, [-sx, -sy], [-2, -1]) for sx, sy in zip(x_shift, y_shift)],
            dim=0,
        )
        return out


class RestModel(torch.nn.Module):
    def __init__(self, model, physics, device='cpu', pad_multiple=1, physics_generator=None, apply_adjoint=False,
                 stochastic=True, average_gradient_steps=1, model_name=None):
        super(RestModel, self).__init__()
        self.physics = physics.to(device)
        self.model = model.to(device)
        self.padding_factor = pad_multiple
        self.physics_generator = physics_generator
        self.apply_adjoint = False
        self.stochastic = stochastic
        self.average_gradient_steps = average_gradient_steps
        self.model_name = model_name

    def pad(self, x):
        r"""
        Pads the input image to a multiple of the padding factor.

        :param torch.Tensor x: input image
        :return: torch.Tensor containing the padded image
        """
        if self.padding_factor == 1:
            return x, None, None
        else:
            H, W = x.shape[-2:]
            H_pad = int(np.ceil(H / self.padding_factor) * self.padding_factor)
            W_pad = int(np.ceil(W / self.padding_factor) * self.padding_factor)
            pad_H = H_pad - H
            pad_W = W_pad - W
            return torch.nn.functional.pad(x, (0, pad_W, 0, pad_H), mode='constant', value=0), pad_H, pad_W

    def crop(self, x, pad_H, pad_W):
        r"""
        Crops the input image to its original size.

        :param torch.Tensor x: input image
        :param int pad_H: padding height
        :param int pad_W: padding width
        :return: torch.Tensor containing the cropped image
        """
        if self.padding_factor == 1:
            return x
        else:
            H, W = x.shape[-2:]
            return x[..., :H - pad_H, :W - pad_W]

    def forward(self, x):
        raise NotImplementedError()


class GradientDescentModel(torch.nn.Module):
    r"""
    Implements FiRe Gradient Descent (GD).

    :param rest_model: RestModel object containing the model and physics.
    :param max_iter: Maximum number of iterations.
    :param lambd: Regularization parameter in the proximal operator.
    :param gamma: Step-size of the prior term.
    :param equivariant: Whether to use equivariant shifts.
    :param apply_adjoint: Whether to apply the adjoint operator of the physics model.
    :param stochastic: Whether to use stochastic physics.
    :param average_last: Whether to average the last iterations.
    :param average_gradient_steps: Number of gradient steps to average.
    :param init_pinv: Whether to initialize with the pseudoinverse of the physics model.
    :param return_u: Whether to return the intermediate u values.
    :param list_gamma_values: List of gamma values for each model in case of multi-model.
    """
    def __init__(self, rest_model, max_iter=20, lambd=1., gamma=1., equivariant=True, apply_adjoint=False,
                 stochastic=True, average_last=True, average_gradient_steps=1, init_pinv=False, return_u=False,
                 list_gamma_values=None):
        super(GradientDescentModel, self).__init__()
        self.rest_model = rest_model
        self.max_iter = max_iter
        self.lambd = lambd
        self.gamma = gamma
        self.data_fidelity = L2()
        self.equivariant = equivariant
        self.apply_adjoint = apply_adjoint
        self.stochastic = stochastic
        self.average_last = average_last
        self.init_pinv = init_pinv
        if self.equivariant:
            self.shift_op = Shift(shift_max=0.1, n_trans=1)
        self.average_gradient_steps = average_gradient_steps
        self.return_u = return_u
        self.list_gamma_values = list_gamma_values
        if isinstance(rest_model, list):
            self.multi_model = True
            if self.list_gamma_values is None:
                self.list_gamma_values = [gamma for _ in range(len(rest_model))]
        else:
            self.multi_model = False

    def gradient_model(self, x, gamma=1, debug=False):
        if self.multi_model:
            return self.gradient_model_multi(x, gamma=gamma, debug=debug)
        else:
            return self.gradient_model_base(x, self.rest_model, gamma=gamma, debug=debug)


    def gradient_model_multi(self, x, gamma=1, debug=False):
        r"""
        Implements the gradient of the data fidelity term.

        :param torch.Tensor x: Image.
        :param float gamma: Step-size of the proximal operator.

        :return: (torch.Tensor) gradient.
        """
        x_out = torch.zeros_like(x)
        cost = 0.

        if debug:
            out_debug = []

        for i, model in enumerate(self.rest_model):
            if not debug:
                x_cur, cost_cur = self.gradient_model_base(x, model, gamma=self.list_gamma_values[i])
            else:
                x_cur, cost_cur, out_debug = self.gradient_model_base(x, model, gamma=self.list_gamma_values[i], debug=debug)
            x_out = x_out + x_cur
            cost = cost + cost_cur

        x_out = x_out / len(self.rest_model)
        cost = cost / len(self.rest_model)

        if not debug:
            return x_out, cost
        else:
            return x_out, cost, out_debug


    def gradient_model_base(self, x, rest_model, gamma=1, debug=False):
        r"""
        Implements the gradient of the data fidelity term.

        :param torch.Tensor x: Image.
        :param float gamma: Step-size of the proximal operator.

        :return: (torch.Tensor) gradient.
        """
        x_out = torch.zeros_like(x)
        cost = 0.

        for _ in range(rest_model.average_gradient_steps):
            degraded, pad_H, pad_W = rest_model.pad(x)
            if not rest_model.stochastic:
                degraded = rest_model.physics(degraded)  # beware this can be stochastic
            else:
                params = rest_model.physics_generator.step(x.size(0))
                degraded = rest_model.physics(degraded, **params)
            if rest_model.apply_adjoint:
                if not rest_model.stochastic:
                    degraded = rest_model.physics.A_adjoint(degraded)
                else:
                    degraded = rest_model.physics.A_adjoint(degraded, **params)
            with torch.no_grad():
                restored = rest_model.model(degraded, physics=rest_model.physics)
                restored = rest_model.crop(restored, pad_H, pad_W)

            cost = cost + torch.linalg.norm((restored - x).flatten(), ord=2)
            x_out = x_out + (1 - gamma) * x + gamma * restored

        x_out = x_out / rest_model.average_gradient_steps
        cost = cost / rest_model.average_gradient_steps

        if not debug:
            return x_out, cost
        else:
            return x_out, cost, [degraded, restored]

    def forward(self, y, physics, x_true=None, x_init=None, last_model=None, debug=False, return_all=False):
        """
        Implements the Davis-Yin Splitting algorithm for the Plug-and-Play framework.

        Args:
            gradient_model:
            physics:
            y:
            max_iter:
            lambd:

        Returns:
        """
        if x_init is None:
            if self.init_pinv:
                x_k = physics.A_dagger(y)
            else:
                x_k = physics.A_adjoint(y)
        else:
            x_k = x_init

        if return_all:
            list_x_k = []
            list_u_k = []
            list_uin_k = []

        # list of cost values
        cost_list = []
        if x_true is not None:
            psnr_list = []
            psnr_list_u = []

        with torch.no_grad():

            x_mean = x_k.clone()
            for it in range(self.max_iter):

                if self.equivariant and it < self.max_iter - 1:
                    x_k, x_shift, y_shift = self.shift_op.A(x_k)

                if not debug and not return_all:
                    u, cost = self.gradient_model(x_k, gamma=self.gamma)
                elif return_all:
                    u, cost, out_debug = self.gradient_model(x_k, gamma=self.gamma, debug=True)
                    in_model = out_debug[0]
                else:
                    u, cost, out_debug = self.gradient_model(x_k, gamma=self.gamma, debug=True)
                if self.equivariant and it < self.max_iter - 1:
                    u = self.shift_op.Ainv(u, x_shift, y_shift)

                x_k = self.data_fidelity.prox(u, y, physics, gamma=self.lambd)

                if return_all:
                    list_x_k.append(x_k)
                    list_u_k.append(u)
                    list_uin_k.append(in_model)

                if self.average_last:
                    # average the last 10 iterations
                    if it >= (self.max_iter - self.max_iter//2):
                        f = 1 / (it - self.max_iter + self.max_iter//2 + 1)
                        x_mean = (1 - f) * x_mean + f * x_k
                        x_k = x_mean.clone()

                if it == self.max_iter - 1 and last_model is not None:
                    x_k = last_model(x_k)

                cost_list.append(cost.item())
                psnr_x = dinv.metric.cal_psnr(x_k, x_true)
                psnr_list.append(psnr_x.item())

                psnr_u = dinv.metric.cal_psnr(u, x_true)
                psnr_list_u.append(psnr_u.item())

        logs = {'cost': cost_list}

        if x_true is not None:
            logs['psnr'] = psnr_list
            logs['psnr_u'] = psnr_list_u

        if debug:
            return x_k, logs, u, x_k_next, x_true, out_debug
        elif return_all:
            return x_k, u, logs, list_x_k, list_u_k, list_uin_k
        else:
            return x_k, u, logs
