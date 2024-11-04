import argparse
import os
import random

import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
from guidance import Guidance, GuidanceConfig
from tqdm import tqdm

device = torch.device("cuda")


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


parser = argparse.ArgumentParser()
parser.add_argument("--prompt", type=str, default="a DSLR photo of a dolphin")
parser.add_argument(
    "--extra_src_prompt",
    type=str,
    default=", oversaturated, smooth, pixelated, cartoon, foggy, hazy, blurry, bad structure, noisy, malformed",
)
parser.add_argument(
    "--extra_tgt_prompt",
    type=str,
    default=", detailed high resolution, high quality, sharp",
)
parser.add_argument("--init_image_fn", type=str, default=None)
parser.add_argument(
    "--mode", type=str, default="bridge", choices=["bridge", "sds", "nfsd", "vsd"]
)
parser.add_argument("--cfg_scale", type=float, default=40)
parser.add_argument("--lr", type=float, default=0.01)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--n_steps", type=int, default=1000)
parser.add_argument("--stage_two_start_step", type=int, default=500)
args = parser.parse_args()

init_image_fn = args.init_image_fn

guidance = Guidance(
    GuidanceConfig(sd_pretrained_model_or_path="stabilityai/stable-diffusion-2-1-base"),
    use_lora=(args.mode == "vsd"),
)

if init_image_fn is not None:
    reference = torch.tensor(plt.imread(init_image_fn))[..., :3]
    reference = reference.permute(2, 0, 1)[None, ...]
    reference = reference.to(guidance.unet.device)

    reference_latent = guidance.encode_image(reference)
    im = reference_latent
else:
    # Initialize with low-magnitude noise, zeros also works
    im = torch.randn((1, 4, 64, 64), device=guidance.unet.device)

save_dir = "results/%s_gen/%s_lr%.3f_seed%d_scale%.1f" % (
    args.mode,
    args.prompt.replace(" ", "_"),
    args.lr,
    args.seed,
    args.cfg_scale,
)
os.makedirs(save_dir, exist_ok=True)
print("Save dir:", save_dir)

seed_everything(args.seed)


def decode_latent(latent):
    latent = latent.detach().to(device)
    with torch.no_grad():
        rgb = guidance.decode_latent(latent)
    rgb = rgb.float().cpu().permute(0, 2, 3, 1)
    rgb = rgb.permute(1, 0, 2, 3)
    rgb = rgb.flatten(start_dim=1, end_dim=2)
    return rgb


batch_size = 1

im.requires_grad_(True)
im.retain_grad()

im_optimizer = torch.optim.AdamW([im], lr=args.lr, betas=(0.9, 0.99), eps=1e-15)
if args.mode == "vsd":
    lora_optimizer = torch.optim.AdamW(
        [
            {"params": guidance.unet_lora.parameters(), "lr": 3e-4},
        ],
        weight_decay=0,
    )

im_opts = []

for step in tqdm(range(args.n_steps)):

    guidance.config.guidance_scale = args.cfg_scale
    if args.mode == "bridge":
        if step < args.stage_two_start_step:
            loss_dict = guidance.sds_loss(
                im=im, prompt=args.prompt, cfg_scale=args.cfg_scale, return_dict=True
            )
        else:
            loss_dict = guidance.bridge_stage_two(
                im=im, prompt=args.prompt, cfg_scale=args.cfg_scale, extra_tgt_prompts=args.extra_tgt_prompt, extra_src_prompts=args.extra_src_prompt, return_dict=True
            )

    elif args.mode == "sds":
        loss_dict = guidance.sds_loss(
            im=im, prompt=args.prompt, cfg_scale=args.cfg_scale, return_dict=True
        )
    elif args.mode == "nfsd":
        loss_dict = guidance.nfsd_loss(
            im=im, prompt=args.prompt, cfg_scale=args.cfg_scale, return_dict=True
        )
    elif args.mode == "vsd":
        loss_dict = guidance.vsd_loss(
            im=im, prompt=args.prompt, cfg_scale=7.5, return_dict=True
        )
        lora_loss = loss_dict["lora_loss"]
        lora_loss.backward()
        lora_optimizer.step()
        lora_optimizer.zero_grad()
    else:
        raise ValueError(args.mode)

    grad = loss_dict["grad"]
    src_x0 = loss_dict["src_x0"] if "src_x0" in loss_dict else grad

    im.backward(gradient=grad)
    im_optimizer.step()
    im_optimizer.zero_grad()

    if step % 10 == 0:
        decoded = decode_latent(im.detach()).cpu().numpy()
        im_opts.append(decoded)
        plt.imsave(os.path.join(save_dir, "debug_image.png"), decoded)

    if step % 100 == 0:
        imageio.mimwrite(
            os.path.join(save_dir, "debug_optimization.mp4"),
            np.stack(im_opts).astype(np.float32) * 255,
            fps=10,
            codec="libx264",
        )
