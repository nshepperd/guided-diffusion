#!/usr/bin/env python

# Imports
import math
import io
import sys
import time
import os
import shutil

from PIL import Image
import requests
import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from torchvision import transforms
from torchvision.transforms import functional as TF
from tqdm.auto import tqdm
from functools import partial

sys.path.append('./CLIP')

import clip
from guided_diffusion.script_util import create_model_and_diffusion, model_and_diffusion_defaults
from guided_diffusion.simple_diffusion import SimpleDiffusion

# Define necessary functions

def fetch(url_or_path):
    if str(url_or_path).startswith('http://') or str(url_or_path).startswith('https://'):
        r = requests.get(url_or_path)
        r.raise_for_status()
        fd = io.BytesIO()
        fd.write(r.content)
        fd.seek(0)
        return fd
    return open(url_or_path, 'rb')

def fetch_model(url):
    basename = os.path.basename(url)
    if not os.path.exists(f'models/{basename}'):
        os.makedirs('models', exist_ok=True)
        os.system(f'curl -o models/{basename} {url}')
    return f'models/{basename}'

class MakeCutouts(nn.Module):
    def __init__(self, cut_size, cutn, cut_pow=1.):
        super().__init__()
        self.cut_size = cut_size
        self.cutn = cutn
        self.cut_pow = cut_pow

    def forward(self, input):
        sideY, sideX = input.shape[2:4]
        max_size = min(sideX, sideY)
        min_size = min(sideX, sideY, self.cut_size)
        cutouts = []
        for _ in range(self.cutn):
            size = int(torch.rand([])**self.cut_pow * (max_size - min_size) + min_size)
            offsetx = torch.randint(0, sideX - size + 1, ())
            offsety = torch.randint(0, sideY - size + 1, ())
            cutout = input[:, :, offsety:offsety + size, offsetx:offsetx + size]
            cutouts.append(F.adaptive_avg_pool2d(cutout, self.cut_size))
        return torch.cat(cutouts)


def spherical_dist_loss(x, y):
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    return (x - y).norm(dim=-1).div(2).arcsin().pow(2).mul(2)


def tv_loss(input):
    """L2 total variation loss, as in Mahendran et al."""
    input = F.pad(input, (0, 1, 0, 1), 'replicate')
    x_diff = input[..., :-1, 1:] - input[..., :-1, :-1]
    y_diff = input[..., 1:, :-1] - input[..., :-1, :-1]
    return (x_diff**2 + y_diff**2).mean([1, 2, 3])

@torch.no_grad()
def txt(prompt):
  """Returns normalized embedding."""
  return norm1(clip_model.encode_text(clip.tokenize(prompt).to(device)).float())

def norm1(prompt):
  """Normalized to the unit hypersphere."""
  return prompt / torch.linalg.norm(prompt, dim=-1, keepdim=True)

# Model settings

model_config = model_and_diffusion_defaults()
model_config.update({
    'attention_resolutions': '32, 16, 8',
    'class_cond': False,
    'diffusion_steps': 1000, # ignored
    'rescale_timesteps': True, # ignored
    'timestep_respacing': '1000', # ignored
    'image_size': 256,
    'learn_sigma': True,
    'noise_schedule': 'linear',
    'num_channels': 256,
    'num_head_channels': 64,
    'num_res_blocks': 2,
    'resblock_updown': True,
    'use_fp16': True,
    'use_scale_shift_norm': True,
})

url_256x256_uncond = 'https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt'
url_512x512_uncond = 'https://the-eye.eu/public/AI/models/512x512_diffusion_unconditional_ImageNet/512x512_diffusion_uncond_finetune_008100.pt'

if model_config['image_size'] == 512:
    model_url = url_512x512_uncond
else:
    model_url = url_256x256_uncond

# Load models

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

model, _ = create_model_and_diffusion(**model_config)
model.load_state_dict(torch.load(fetch_model(model_url), map_location='cpu'))
model.requires_grad_(False).eval().to(device)
for name, param in model.named_parameters():
    if 'qkv' in name or 'norm' in name or 'proj' in name:
        param.requires_grad_()
if model_config['use_fp16']:
    model.convert_to_fp16()

clip_model = clip.load('ViT-B/16', jit=False)[0].eval().requires_grad_(False).to(device)
clip_size = clip_model.visual.input_resolution
normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                 std=[0.26862954, 0.26130258, 0.27577711])

# Use SimpleDiffusion instead of the regular one
diffusion = SimpleDiffusion(device, schedule_name=model_config['noise_schedule'])

# generates the closed interval [t_start..t_end]
def make_schedule(t_start, t_end, step_size=1):
    schedule = []
    t = t_start
    while t > t_end:
        schedule.append(t)
        t -= step_size
    schedule.append(t_end)
    return schedule

## Settings for this run:

title = "shutterstock watermark removal"
prompt = txt(title)
batch_size = 1
clip_guidance_scale = 0  # Controls how much the image should look like the prompt.
tv_scale = 150           # Controls the smoothness of the final output.
mask_scale = 1000
eta = 1.0 # 0.0: ddim. 1.0: ddpm. between 1.0 and 2.0: extra noisy
cutn = 16
n_batches = 1
init_image = 'examples/inpainting_goose.png'
skip_timesteps = 0  # This needs to be between approx. 200 and 500 when using an init image.
                      # Higher values make the output look more like the init.
schedule = make_schedule(1000 - skip_timesteps, 0, 1)
# schedule = make_schedule(1000, 500) + make_schedule(500, 250, 0.5) + make_schedule(250, 0, 0.25)
seed = 0

## Actually do the run...

def run():
    os.makedirs('progress', exist_ok=True)
    os.makedirs('samples', exist_ok=True)
    if seed is not None:
        torch.manual_seed(seed)

    text_embed = prompt

    init = None
    init_mask = None
    if init_image is not None:
        S = model_config['image_size']
        init = Image.open(fetch(init_image)).convert('RGBA')
        init = init.resize((S, S), Image.BILINEAR)
        init = TF.to_tensor(init).to(device)
        init_mask = init[3] # alpha channel
        init_mask = (init_mask>0.5).to(torch.float32)
        init = init[:3].unsqueeze(0).mul(2).sub(1) # RGB

    make_cutouts = MakeCutouts(clip_size, cutn)

    def mask_xstart(img):
        if init_mask is not None:
            img = init_mask.sqrt() * init + (1 - init_mask).sqrt() * img
        return img

    def cond_fn(x, t, y=None):
        with torch.enable_grad():
            x = x.detach().requires_grad_()
            n = x.shape[0]
            pred_xstart = diffusion.p_xstart(model, x, t)
            fac = np.sqrt(1 - diffusion.alpha(t))
            x_in = mask_xstart(pred_xstart) * fac + x * (1 - fac)
            if clip_guidance_scale > 0:
                clip_in = normalize(make_cutouts(x_in.add(1).div(2)))
                image_embeds = clip_model.encode_image(clip_in).float().reshape([cutn, n, -1])
                dists = spherical_dist_loss(image_embeds, text_embed.unsqueeze(0))
                losses = dists.mean(0)
            else:
                losses = torch.zeros([], device=x.device)
            tv_losses = tv_loss(x_in)
            mask_losses = (init_mask * (pred_xstart - init)).square().mean((1,2,3))
            loss = (losses.sum() * clip_guidance_scale
                    + tv_losses.sum() * tv_scale
                    + mask_losses.sum() * mask_scale / (1 - diffusion.alpha(t)))
            grad = -torch.autograd.grad(loss, x)[0]
        # Gradient clipping before returning
        magnitude = grad.square().mean().sqrt()
        return grad * magnitude.clamp(max=0.2) / magnitude

    for i in range(n_batches):
        timestring = time.strftime('%Y%m%d%H%M%S')

        samples = diffusion.p_sample_loop_progressive(
            model,
            (batch_size, 3, model_config['image_size'], model_config['image_size']),
            schedule = schedule,
            cond_fn = cond_fn,
            init_image = init,
            eta=eta,
            init_mask = init_mask,
            progress = tqdm
        )

        for j, sample in enumerate(samples):
            if j % 100 == 0 or j == len(schedule)-2:
                print()
                for k, image in enumerate(sample['pred_xstart']):
                    pil_image = TF.to_pil_image(image.add(1).div(2).clamp(0, 1))
                    filename = f'progress/progress_{i * batch_size + k:05}_{j:04}.png'
                    pil_image.save(filename)
                    tqdm.write(f'Wrote batch {i}, step {j}, output {k} to {filename}')

        for k in range(batch_size):
            filename = f'progress_{i * batch_size + k:05}.png'
            final_name = f'samples/{timestring}_{k}_{title}.png'
            shutil.copyfile(filename, final_name)

run()
