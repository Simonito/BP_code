import torch
from torch.cuda.amp import autocast
from generative.networks.schedulers import DDPMScheduler
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly
import plotly.express as px

import os
from argparse import ArgumentParser
from monai.metrics import SSIMMetric, PSNRMetric
import pandas as pd
import numpy as np
import wandb

from models.trunet_orig_newFattn import UNETR as UnetBaseline
from models.trunet_patch_newFattn import UNETR as UnetPatch


def retrieve_scheduler():
    return DDPMScheduler(num_train_timesteps=1000)


def load_models(baseline_model_path, patch_model_path, img_size: int, device: torch.device):
    # first, load the baseline model
    model_base = UnetBaseline(
        in_channels=2,
        out_channels=1,
        num_heads=16,
        patch_size=4,
        img_size=img_size,
    ).to(device)
    model_base.load_state_dict(torch.load(baseline_model_path, map_location=device))

    # secondly, load the model patch
    model_patch = UnetPatch(
        in_channels=2,
        out_channels=1,
        num_heads=16,
        patch_size=4,
        img_size=img_size,
    ).to(device)
    model_patch.load_state_dict(torch.load(patch_model_path, map_location=device))

    return model_base, model_patch


def log_table_row(index, row, wandb):
    wandb.log({
        'id': index,
        'ssim orig': row['SSIM orig'],
        'ssim patch': row['SSIM patch'],
        'psnr orig': row['PSNR orig'],
        'psnr patch': row['PSNR patch'],
        'ct': row['CT'],
        'ct orig': row['sCT orig'],
        'ct patch': row['sCT patch'],
    })


def perform_benchmark(model_orig, model_patch, val_dataset, img_out_dir, device, wandb):
    scheduler = retrieve_scheduler()

    ssim = SSIMMetric(spatial_dims=2)
    psnr = PSNRMetric(max_val=1.0)
    if wandb is not None:
        wandb_table = wandb.Table(columns=['CT', 'sCT orig', 'sCT patch', 'SSIM orig', 'SSIM patch', 'PSNR orig', 'PSNR patch', 'histogram'])

    model_orig.eval()
    model_patch.eval()

    n = 1
    for idx, data in enumerate(tqdm(val_dataset)):
        # early stopping condition
        if idx == 50:
            break

        inputct = data["ct"][0, ...]  # Pick an input slice of the validation set to be segmented
        inputmr = data["mri"][0, ...]  # Check out the ground truth label mask. If it is empty, pick another input slice.

        input_ct = inputct[None, None, ...].to(device)
        input_mr = inputmr[None, None, ...].to(device)
        for k in range(n):
            noise = torch.randn_like(input_ct, device=device)
            current_img_orig = noise.clone()
            combined_orig = torch.cat((current_img_orig, input_mr), dim=1)

            current_img_patch = noise.clone()
            combined_patch = torch.cat((current_img_patch, input_mr), dim=1)

            scheduler.set_timesteps(num_inference_steps=1000)
            for t in tqdm(scheduler.timesteps):
                with autocast(enabled=False):
                    with torch.no_grad():
                        model_output_orig = model_orig(
                            x=combined_orig,
                            time=torch.tensor((t,), device=device),
                        )
                        current_img_orig, _ = scheduler.step(model_output_orig, t, current_img_orig)
                        combined_orig = torch.cat((current_img_orig, input_mr), dim=1)

                        model_output_patch = model_patch(
                            x=combined_patch,
                            time=torch.tensor((t,), device=device),
                        )
                        current_img_patch, _ = scheduler.step(model_output_patch, t, current_img_patch)
                        combined_patch = torch.cat((current_img_patch, input_mr), dim=1)

            img_data_dir = os.path.join(img_out_dir, f"{idx:02d}")
            if not os.path.exists(img_data_dir):
                os.makedirs(img_data_dir)

            compare = torch.cat((input_ct.cpu(), current_img_orig.cpu(), current_img_patch.cpu()), dim=-1)
            plt.figure(f"Comparison", figsize=(12,4))
            plt.style.use("default")
            plt.imshow(compare[0, 0, ...].cpu(), vmin=0, vmax=1, cmap="gray")
            plt.tight_layout()
            plt.axis("off")
            plt.savefig(os.path.join(img_data_dir, f"{k}.png"))


            curr_psnr_orig = psnr(y_pred=current_img_orig, y=input_ct).item()
            curr_ssim_orig = ssim(y_pred=current_img_orig, y=input_ct).item()

            curr_psnr_patch = psnr(y_pred=current_img_patch, y=input_ct).item()
            curr_ssim_patch = ssim(y_pred=current_img_patch, y=input_ct).item()
            if k == 0:
                print()
            print(f"====== {idx:02d}_{k} ======")
            print(f"SSIM orig = {curr_ssim_orig}")
            print(f"SSIM patch= {curr_ssim_patch}")
            print()
            print(f"PSNR orig = {curr_psnr_orig}")
            print(f"PSNR patch= {curr_psnr_patch}")
            print("------------------")

            log_ct = input_ct[0, 0, ...].cpu().numpy()
            log_s_ct_orig = current_img_orig[0, 0, ...].cpu().numpy()
            log_s_ct_patch = current_img_patch[0, 0, ...].cpu().numpy()

            df = pd.DataFrame(dict(
                series=np.concatenate((["CT"] * len(log_ct.flatten()), ["sCT orig"] * len(log_s_ct_orig.flatten()), ["sCT patch"] * len(log_s_ct_patch.flatten()))), 
                data=np.concatenate((log_ct.flatten(), log_s_ct_orig.flatten(), log_s_ct_patch.flatten()))
            ))
            histogram = px.histogram(df, x="data", color="series", nbins=100, barmode="overlay")
            histogram.update_layout(xaxis_title="Pixel Value", yaxis_title="Frequency")

            if wandb is not None:
                # add row to WandB table
                wandb_table.add_data(
                    # CT
                    wandb.Image(log_ct),
                    # synthetic CT orig-technique
                    wandb.Image(log_s_ct_orig),
                    # synthetic CT patch-technique
                    wandb.Image(log_s_ct_patch),
                    # SSIM
                    curr_ssim_orig,
                    curr_ssim_patch,
                    # PSNR
                    curr_psnr_orig,
                    curr_psnr_patch,
                    # Histogram
                    wandb.Html(plotly.io.to_html(histogram))
                )

    if wandb is not None:
        df = wandb_table.get_dataframe()

        for col in ['SSIM orig', 'SSIM patch', 'PSNR orig', 'PSNR patch']:
            max_idx = df[col].idxmax()
            row_max = df.loc[max_idx]
            log_table_row(index=max_idx, row=row_max, wandb=wandb)
            min_idx = df[col].idxmin()
            row_min = df.loc[min_idx]
            log_table_row(index=min_idx, row=row_min, wandb=wandb)
            
        wandb.log({'Evaluation Table': wandb_table})


def main(val_dataset,
         img_size: int,
         wandb_api=None,
         torch_device=None,
         model_base_path=None,
         model_patch_path=None):
    if val_dataset is None:
        raise AttributeError("Missing validation dataset")
    device = torch.device("cuda") if torch_device is None else torch.device(torch_device)
    if model_base_path is None:
        raise AttributeError('Missing argument that resolves the path to the baseline model state')
    if model_patch_path is None:
        raise AttributeError('Missing argument that resolves the path to the state of the model with patch embeddings')

    model_base, model_patch = load_models(model_base_path, model_patch_path, img_size=img_size, device=device)

    do_log_wandb = True
    if wandb_api is None:
        print('NO wandb api key was given, disabling wandb logging')
        do_log_wandb = False

    if do_log_wandb:
        wandb_key = args.wandb_api
        wandb.login(key=wandb_key)
        wandb.init(
            project="synthrad_2d_benchmark",
            name="baseline_vs_patch"
        )

    out_dir = 'outputs'
    img_out = os.path.join(out_dir, 'benchmarks', 'baseline_vs_patch',  'images')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    perform_benchmark(model_orig=model_base,
                      model_patch=model_patch,
                      val_dataset=val_dataset,
                      img_out_dir=img_out,
                      wandb=wandb if do_log_wandb else None,
                      device=device,
    )
