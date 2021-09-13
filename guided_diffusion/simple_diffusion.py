import enum
import math

import numpy as np
import torch

def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    """
    Get a pre-defined beta schedule for the given name.

    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


class ModelMeanType(enum.Enum):
    """
    Which type of output the model predicts.
    """

    PREVIOUS_X = enum.auto()  # the model predicts x_{t-1}
    START_X = enum.auto()  # the model predicts x_0
    EPSILON = enum.auto()  # the model predicts epsilon


class ModelVarType(enum.Enum):
    """
    What is used as the model's output variance.

    The LEARNED_RANGE option has been added to allow the model to predict
    values between FIXED_SMALL and FIXED_LARGE, making its job easier.
    """

    LEARNED = enum.auto()
    FIXED_SMALL = enum.auto()
    FIXED_LARGE = enum.auto()
    LEARNED_RANGE = enum.auto()

class SimpleDiffusion(object):
    """
    Simplified diffusion model for EPSILON type models.
    TODO: make use of the learned variance.
    """
    def __init__(self,
                 device,
                 *,
                 schedule_name='linear',
                 diffusion_steps=1000,
                 model_mean_type=ModelMeanType.EPSILON,
                 model_var_type=ModelVarType.LEARNED_RANGE):
        assert model_mean_type == ModelMeanType.EPSILON
        assert model_var_type == ModelVarType.LEARNED_RANGE

        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        self.diffusion_steps = diffusion_steps
        self.device = device

        # Use float64 for accuracy.
        # Note to self: it's N(µ, σ^2)

        # Forward process:
        # q(x[t] | x[t−1]) := N(sqrt(1-βt) x[t-1],  βt)
        beta1 = get_named_beta_schedule(schedule_name, diffusion_steps).astype(np.float64)
        # Note: βt is actually beta1[t-1], because q(x[0] | x[−1]) is not defined

        # q(x[t] | x[0]) := N(sqrt(α[t]) x[0],  1 − α[t])
        # q(x[t] | x[t-1]) := N(sqrt(α[t] / α[t-1]) x[t-1],  1 - α[t] / α[t-1])
        self.alphas = np.append(1.0, np.cumprod(1.0 - beta1))

    def alpha(self, t):
        """Does geometric interpolation of alpha[t] for non-integral t."""
        assert type(t) in (int,float)
        if type(t) is int:
            return self.alphas[t]
        elif type(t) is float:
            t1 = int(t)
            t2 = min(t1 + 1, self.diffusion_steps)
            r = t - t1
            return self.alphas[t1]**(1-r) * self.alphas[t2]**r

    def split_model_out(self, model_output):
        [B, C, H, W] = model_output.shape
        assert C == 6
        eps, model_var_values = torch.split(model_output, 3, dim=1)
        return eps

    def q_sample(self, x, t1, t2, noise=None):
        assert t1 < t2
        if noise is None:
            noise = torch.randn_like(x)
        a1 = self.alpha(t1)
        a2 = self.alpha(t2)
        return np.sqrt(a2/a1) * x + np.sqrt(1 - a2/a1) * noise

    def p_xstart(self, model, x, t):
        t_in = torch.tensor(t).float().to(x.device).broadcast_to([x.shape[0]])
        eps = self.split_model_out(model(x, t_in))
        a1 = self.alpha(t)
        xstart = (x - np.sqrt(1 - a1) * eps) / np.sqrt(a1)
        return xstart

    def p_sample(self, model, x, t1, t2, cond_fn=None, noise=None, eta=1.0):
        assert t1 > t2

        t_in = torch.tensor(t1).float().to(x.device).broadcast_to([x.shape[0]])
        eps = self.split_model_out(model(x, t_in))

        a1 = self.alpha(t1)
        a2 = self.alpha(t2)

        # "Diffusion Models Beat GANs on Image Synthesis" page 7-8.
        # Conditioning for DDIM
        eps0 = eps
        if cond_fn is not None:
            eps = eps - np.sqrt(1 - a1) * cond_fn(x, t1)

        if noise is None:
            noise = torch.randn_like(x)

        # "Denoising Diffusion Implicit Models" formula 12
        xstart = (x - np.sqrt(1 - a1) * eps) / np.sqrt(a1)

        ddpm_sigma2 = (1 - a1/a2) / (1 - a1)
        if eta <= 1.0:
            sigma2 = eta * ddpm_sigma2
        else:
            sigma2 = ddpm_sigma2 + (eta-1) * (1 - ddpm_sigma2)
        adjust = np.sqrt(1 - sigma2) * eps + np.sqrt(sigma2) * noise

        # Generate the unconditional xstart for output using the original epsilon
        xstart0 = (x - np.sqrt(1 - a1) * eps0) / np.sqrt(a1)
        return np.sqrt(a2) * xstart + np.sqrt(1 - a2) * adjust, xstart0

    @torch.no_grad()
    def p_sample_loop_progressive(self, model, shape, init_image=None, schedule=None, cond_fn=None, eta=1.0, init_mask=None, progress=None):
        if schedule is None:
            schedule = reversed(range(self.diffusion_steps + 1)) # [T..0]
        schedule = list(schedule)
        timesteps = list(zip(schedule, schedule[1:]))

        if init_image is None:
            image = self.q_sample(torch.zeros(*shape, device=self.device, dtype=torch.float32), 0, schedule[0])
        else:
            image = self.q_sample(init_image.broadcast_to(shape), 0, schedule[0])



        for (t1, t2) in (progress(timesteps) if progress else timesteps):
            if t1 == t2:
                continue
            image, pred_xstart = self.p_sample(model, image, t1, t2, cond_fn=cond_fn, eta=eta)
            if init_mask is not None and t2 > 0:
                noisy_init = self.q_sample(init_image.broadcast_to(shape), 0, t2)
                image = torch.sqrt(init_mask) * noisy_init + torch.sqrt(1 - init_mask) * image
            yield {'sample': image, 'pred_xstart': pred_xstart}
