import torch
from torch.cuda.amp import autocast
from generative.networks.schedulers import DDIMScheduler
from tqdm import tqdm

import matplotlib.pyplot as plt
import plotly
import plotly.express as px
import plotly.offline as pioff
import plotly.graph_objects as go
from scipy.stats import gaussian_kde

import os
from monai.metrics import SSIMMetric, PSNRMetric
import pandas as pd
import numpy as np
from generative.networks.schedulers import NoiseSchedules


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
    return DDIMScheduler(num_train_timesteps=1000, schedule=schedule)


def log_table_row(index, row, wandb):
    wandb.log({
        'id': index,
        'ssim lin': row['SSIM lin'],
        'ssim cos': row['SSIM cos'],
        'psnr lin': row['PSNR lin'],
        'psnr cos': row['PSNR cos'],
        'ct': row['CT'],
        'ct lin': row['sCT lin'],
        'ct cos': row['sCT cos'],
    })

def create_histogram(ct, ct_lin, ct_cos):
    fig = go.Figure()

    def add_kde_trace(fig, data, rgb_color, kde_color, title):
        hist, bin_edges = np.histogram(data, bins=100, density=False)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        kde = gaussian_kde(data)
        kde_values = kde(bin_centers)
        scaling_factor = len(data) / sum(kde_values)
        kde_values_scaled = [val * scaling_factor for val in kde_values]
        # Convert RGB values to HEX format
        hex_color = f'#{rgb_color[0]:02x}{rgb_color[1]:02x}{rgb_color[2]:02x}'
        fig.add_trace(go.Bar(x=bin_centers, y=hist, marker=dict(color=hex_color), name=title, opacity=0.5, width=np.diff(bin_edges)))
        fig.add_trace(go.Scatter(x=bin_centers, y=kde_values_scaled, mode='lines', line=dict(color=kde_color), name='KDE'))
        fig.update_xaxes(title_text=title)

    add_kde_trace(fig, ct, [101, 110, 242], 'blue', 'CT')
    add_kde_trace(fig, ct_lin, [222, 96, 70], 'red', 'sCT (lineárny)')
    add_kde_trace(fig, ct_cos, [93, 201, 154], 'green', 'sCT (kosínusový)')

    fig.update_layout(title_text="Distribúcia hodnôt pixelov", yaxis=dict(title='Frekvencia'))
    return fig

def perform_benchmark(model_lin, model_cos, val_dataset, img_out_dir, wandb):
    device = torch.device('cuda')
    scheduler_lin = retrieve_scheduler('linear_beta')
    scheduler_cos = retrieve_scheduler('cosine_poly')

    ssim = SSIMMetric(spatial_dims=2)
    psnr = PSNRMetric(max_val=1.0)
    wandb_table = wandb.Table(columns=['CT', 'sCT lin', 'sCT cos', 'SSIM lin', 'SSIM cos', 'PSNR lin', 'PSNR cos', 'histogram', 'histogram-kde'])

    model_lin.eval()
    model_cos.eval()

    ssim_all_lin = []
    ssim_all_cos = []
    psnr_all_lin = []
    psnr_all_cos = []

    n = 1
    for idx, data in enumerate(tqdm(val_dataset)):
        # early stopping condition
        # if idx == 50:
        #     break

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

            scheduler_lin.set_timesteps(num_inference_steps=20)
            scheduler_cos.set_timesteps(num_inference_steps=20)
            for t in scheduler_lin.timesteps:
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

            img_data_dir = os.path.join(img_out_dir, f"{idx:03d}")
            if not os.path.exists(img_data_dir):
                os.makedirs(img_data_dir)

            divider = torch.ones(input_ct.shape[0], input_ct.shape[1], input_ct.shape[2], 20)
            compare = torch.cat((input_ct.cpu(), divider, current_img_lin.cpu(), divider, current_img_cos.cpu()), dim=-1)
            plt.figure(f"Comparison", figsize=(12,4))
            plt.style.use("default")
            plt.imshow(compare[0, 0, ...].cpu(), vmin=0, vmax=1, cmap="gray")
            plt.tight_layout()
            plt.axis("off")
            plt.savefig(os.path.join(img_data_dir, f"{idx:03d}.png"))


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
            hist_out = os.path.join(img_data_dir,f'histogram_{idx:03d}.html')
            pioff.plot(histogram, filename=hist_out, auto_open=False)

            hist_experimental = create_histogram(log_ct.flatten(), log_s_ct_lin.flatten(), log_s_ct_cos.flatten())
            hist_experimental_out = os.path.join(img_data_dir,f'histogram_distribution_{idx:03d}.html')
            pioff.plot(hist_experimental, filename=hist_experimental_out, auto_open=False)

            # add row to WandB table
            wandb_table.add_data(
                # CT
                wandb.Image(log_ct),
                # synthetic CT orig-technique
                wandb.Image(log_s_ct_lin),
                # synthetic CT patch-technique
                wandb.Image(log_s_ct_cos),
                # SSIM
                curr_ssim_lin,
                curr_ssim_cos,
                # PSNR
                curr_psnr_lin,
                curr_psnr_cos,
                # Histogram
                wandb.Html(plotly.io.to_html(histogram)),
                # Histogram + Density Distribution
                wandb.Html(plotly.io.to_html(hist_experimental)),
            )

    df = wandb_table.get_dataframe()

    for col in ['SSIM lin', 'SSIM cos', 'PSNR lin', 'PSNR cos']:
        max_idx = df[col].idxmax()
        row_max = df.loc[max_idx]
        log_table_row(index=max_idx, row=row_max, wandb=wandb)
        min_idx = df[col].idxmin()
        row_min = df.loc[min_idx]
        log_table_row(index=min_idx, row=row_min, wandb=wandb)
        
    wandb.log({'Evaluation Table': wandb_table})

    lin_ssim_mean = np.mean(ssim_all_lin)
    lin_ssim_std = np.std(ssim_all_lin)
    cos_ssim_mean = np.mean(ssim_all_cos)
    cos_ssim_std = np.std(ssim_all_cos)

    lin_psnr_mean = np.mean(psnr_all_lin)
    lin_psnr_std = np.std(psnr_all_lin)
    cos_psnr_mean = np.mean(psnr_all_cos)
    cos_psnr_std = np.std(psnr_all_cos)

    print(f'SSIM linear: {lin_ssim_mean:.2f} ± {lin_ssim_std:.2f}')
    print(f'SSIM cosine: {cos_ssim_mean:.2f} ± {cos_ssim_std:.2f}')
    print()
    print(f'PSNR linear: {lin_psnr_mean:.2f} ± {lin_psnr_std:.2f}')
    print(f'PSNR cosine: {cos_psnr_mean:.2f} ± {cos_psnr_std:.2f}')

