import torch
from torch.cuda.amp import autocast
from generative.networks.schedulers import DDPMScheduler, NoiseSchedules
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly
import plotly.express as px
import plotly.offline as pioff

import os
from argparse import ArgumentParser
from monai.metrics import SSIMMetric, PSNRMetric
import pandas as pd
import numpy as np
import wandb

from models.trunet_orig_newFattn import UNETR


def betas_for_alpha_bar(
    num_diffusion_timesteps, alpha_bar, max_beta=0.999
):  # https://github.com/openai/improved-diffusion/blob/783b6740edb79fdb7d063250db2c51cc9545dcd1/improved_diffusion/gaussian_diffusion.py#L45C1-L62C27
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return torch.tensor(betas)


@NoiseSchedules.add_def("cosine_poly", "Cosine schedule")
def _cosine_beta(num_train_timesteps: int, s: float = 8e-3, order: float = 2, *args):
    return betas_for_alpha_bar(
        num_train_timesteps,
        lambda t: np.cos((t + s) / (1 + s) * np.pi / 2) ** order,
    )


def retrieve_scheduler(schedule):
    return DDPMScheduler(num_train_timesteps=1000, schedule=schedule)


def load_models(lin_model_path, cos_model_path, img_size: int, device: torch.device):
    # first, load the baseline model
    model_lin = UNETR(
        in_channels=2,
        out_channels=1,
        num_heads=16,
        patch_size=4,
        img_size=img_size,
    ).to(device)
    model_lin.load_state_dict(torch.load(lin_model_path, map_location=device))

    # secondly, load the model patch
    model_cos = UNETR(
        in_channels=2,
        out_channels=1,
        num_heads=16,
        patch_size=4,
        img_size=img_size,
    ).to(device)
    model_cos.load_state_dict(torch.load(cos_model_path, map_location=device))

    return model_lin, model_cos


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


def perform_benchmark(model_lin, model_cos, val_dataset, img_out_dir, device, wandb):
    scheduler_lin = retrieve_scheduler('linear_beta')
    scheduler_cos = retrieve_scheduler('cosine_poly')

    ssim = SSIMMetric(spatial_dims=2)
    psnr = PSNRMetric(max_val=1.0)
    if wandb is not None:
        wandb_table = wandb.Table(columns=['CT', 'sCT lin', 'sCT cos', 'SSIM lin', 'SSIM cos', 'PSNR lin', 'PSNR cos', 'histogram'])

    model_lin.eval()
    model_cos.eval()

    ssim_all_lin = []
    ssim_all_cos = []
    psnr_all_lin = []
    psnr_all_cos = []

    n = 1
    for idx, data in enumerate(tqdm(val_dataset)):
        inputct = data["ct"][0, ...]  # Pick an input slice of the validation set to be segmented
        inputmr = data["mri"][0, ...]  # Check out the ground truth label mask. If it is empty, pick another input slice.

        input_ct = inputct[None, None, ...].to(device)
        input_mr = inputmr[None, None, ...].to(device)
        for k in range(n):
            noise = torch.randn_like(input_ct, device=device)
            current_img_lin = noise.clone()
            combined_lin = torch.cat((current_img_lin, input_mr), dim=1)

            current_img_cos = noise.clone()
            combined_cos = torch.cat((current_img_cos, input_mr), dim=1)

            scheduler_lin.set_timesteps(num_inference_steps=1000)
            scheduler_cos.set_timesteps(num_inference_steps=1000)
            for t in tqdm(scheduler.timesteps):
                with autocast(enabled=False):
                    with torch.no_grad():
                        model_output_lin = model_lin(
                            x=combined_lin,
                            time=torch.tensor((t,), device=device),
                        )
                        current_img_lin, _ = scheduler_lin.step(model_output_lin, t, current_img_lin)
                        combined_lin = torch.cat((current_img_lin, input_mr), dim=1)

                        model_output_cos = model_cos(
                            x=combined_cos,
                            time=torch.tensor((t,), device=device),
                        )
                        current_img_cos, _ = scheduler_cos.step(model_output_cos, t, current_img_cos)
                        combined_cos = torch.cat((current_img_cos, input_mr), dim=1)

            img_data_dir = os.path.join(img_out_dir, f"{idx:02d}")
            if not os.path.exists(img_data_dir):
                os.makedirs(img_data_dir)

            divider = torch.ones(input_ct.shape[0], input_ct.shape[1], input_ct.shape[2], 20)
            compare = torch.cat((input_ct.cpu(), divider, current_img_lin.cpu(), divider, current_img_cos.cpu()), dim=-1)
            plt.figure(f"Comparison", figsize=(12,4))
            plt.style.use("default")
            plt.imshow(compare[0, 0, ...].cpu(), vmin=0, vmax=1, cmap="gray")
            plt.tight_layout()
            plt.axis("off")
            plt.savefig(os.path.join(img_data_dir, f"{k}.png"))

            curr_psnr_lin = psnr(y_pred=current_img_lin, y=input_ct).item()
            curr_ssim_lin = ssim(y_pred=current_img_lin, y=input_ct).item()

            curr_psnr_cos = psnr(y_pred=current_img_cos, y=input_ct).item()
            curr_ssim_cos = ssim(y_pred=current_img_cos, y=input_ct).item()

            ssim_all_lin.append(curr_ssim_lin)
            ssim_all_cos.append(curr_ssim_cos)
            psnr_all_lin.append(curr_psnr_lin)
            psnr_all_cos.append(curr_psnr_cos)
            if k == 0:
                print()
            print(f"====== {idx:02d}_{k} ======")
            print(f"SSIM lin = {curr_ssim_lin}")
            print(f"SSIM cos = {curr_ssim_cos}")
            print()
            print(f"PSNR lin = {curr_psnr_lin}")
            print(f"PSNR cos = {curr_psnr_cos}")
            print("------------------")

            log_ct = input_ct[0, 0, ...].cpu().numpy()
            log_s_ct_lin = current_img_lin[0, 0, ...].cpu().numpy()
            log_s_ct_cos = current_img_cos[0, 0, ...].cpu().numpy()

            df = pd.DataFrame(dict(
                series=np.concatenate((["CT"] * len(log_ct.flatten()), ["sCT lin"] * len(log_s_ct_lin.flatten()), ["sCT cos"] * len(log_s_ct_cos.flatten()))), 
                data=np.concatenate((log_ct.flatten(), log_s_ct_lin.flatten(), log_s_ct_cos.flatten()))
            ))
            histogram = px.histogram(df, x="data", color="series", nbins=100, barmode="overlay")
            histogram.update_layout(xaxis_title="Pixel Value", yaxis_title="Frequency")
            hist_out = os.path.join(img_data_dir,f'histogram_{idx:02d}.html')
            pioff.plot(histogram, filename = hist_out, auto_open=False)

            if wandb is not None:
                # add row to WandB table
                wandb_table.add_data(
                    # CT
                    wandb.Image(log_ct),
                    # synthetic CT linear-schedule technique
                    wandb.Image(log_s_ct_lin),
                    # synthetic CT cosine-schedule technique
                    wandb.Image(log_s_ct_cos),
                    # SSIM
                    curr_ssim_lin,
                    curr_ssim_cos,
                    # PSNR
                    curr_psnr_lin,
                    curr_psnr_cos,
                    # Histogram
                    wandb.Html(plotly.io.to_html(histogram))
                )

    if wandb is not None:
        df = wandb_table.get_dataframe()

        for col in ['SSIM lin', 'SSIM cos', 'PSNR lin', 'PSNR cos']:
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
         model_lin_path=None,
         model_cos_path=None):
    if val_dataset is None:
        raise AttributeError("Missing validation dataset")
    device = torch.device("cuda") if torch_device is None else torch.device(torch_device)
    if model_lin_path is None:
        raise AttributeError('Missing argument: path to the model using linear schedule')
    if model_cos_path is None:
        raise AttributeError('Missing argument: path to the model using cosine schedule')

    model_lin, model_cos = load_models(model_lin_path, model_cos_path, img_size=img_size, device=device)

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
    img_out = os.path.join(out_dir, 'benchmarks', 'linear_vs_cosine',  'images')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    perform_benchmark(model_lin=model_lin,
                      model_cos=model_cos,
                      val_dataset=val_dataset,
                      img_out_dir=img_out,
                      wandb=wandb if do_log_wandb else None,
                      device=device,
    )
