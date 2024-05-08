import os
import time
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from argparse import ArgumentParser
from monai.utils import set_determinism

from generative.networks.schedulers.ddpm import DDPMScheduler
from generative.inferers import DiffusionInferer

from torch.optim.lr_scheduler import OneCycleLR

from preprocess import get_data_paths, create_loaders, preprocess_data
from models.trunet_orig_drop_small import UNETR

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import plotly
import plotly.express as px

import wandb
############### CHECK VERSIONS############
import monai
import generative
import torch
print(f"monai={monai.__version__} | generative={generative.__version__} | torch={torch.__version__} | numpy={np.__version__}")
##########################################


parser = ArgumentParser()
parser.add_argument('--data_path', type=str, help='path to data')
parser.add_argument('--wandb_api', type=str, help='API key for WandB login')

args = parser.parse_args()

# WandB login and config setup
wandb_key = args.wandb_api
wandb.login(key=wandb_key)
wandb.init(
    project="synthrad_2d_since_drpout",
    config={
        "learning_rate": 2.5e-5,
        "epochs": 2_000,
        "img_size": (168, 168),
        "patch_size": (4, 4),
        "batch_size": 8,
        "num_heads": 16,
        "noise_schedule": "linear_beta",
        "dropout": 0.5,
    }
)

# Data path setup
data_path = args.data_path
out_dir = 'outputs'
img_out = os.path.join(out_dir, 'images')
model_out = os.path.join(out_dir, 'model_checkpoints')

if not os.path.exists(img_out):
    os.makedirs(img_out)
if not os.path.exists(model_out):
    os.makedirs(model_out)

set_determinism(42)

img_size = wandb.config['img_size']
patch_size = wandb.config['patch_size']
batch_size = wandb.config['batch_size']

# vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
# code taken from these discussions:
# https://github.com/Project-MONAI/GenerativeModels/discussions/468
# https://github.com/Project-MONAI/GenerativeModels/issues/397
from generative.networks.schedulers import NoiseSchedules
import numpy as np

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
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

# we are shuffling the seeds here because of the way the images are provided
# The images were taken in 3 different sites
# If we dont shuffle, then most of the train data would be from site 1 and 2 and only a few from site 3
#   and then all of the validation data will be from site 3
# (We are providing a specific seed here for reproducibility)
tr_images, val_images = get_data_paths(data_path, shuffle_seed=42)
tr_images, val_images = preprocess_data(train_images=tr_images, val_images=val_images, img_size=img_size)
train_loader, val_loader = create_loaders(tr_images, val_images, batch_size=batch_size)

device = torch.device("cuda")

model = UNETR(
    in_channels=2,
    out_channels=1,
    num_heads=wandb.config['num_heads'],
    patch_size=patch_size[0],
    img_size=img_size[0],
    dropout=wandb.config['dropout']
).to(device)

diffusion_steps = 1000

scheduler = DDPMScheduler(num_train_timesteps=1000, schedule=wandb.config['noise_schedule'])
optimizer = torch.optim.Adam(params=model.parameters(), lr=wandb.config['learning_rate'])
inferer = DiffusionInferer(scheduler)

n_epochs = wandb.config['epochs']
val_interval = 10
epoch_loss_list = []
val_epoch_loss_list = []

best_model_epoch = 0
best_val = 2.0 # we assume that the validation will go under this value

# lrs = []
# lr_scheduler = OneCycleLR(optimizer, max_lr=wandb.config['learning_rate'] * 10, total_steps=len(train_loader) * n_epochs)

scaler = GradScaler()
total_start = time.time()
for epoch in (p_bar := tqdm(range(n_epochs))):
    p_bar.set_description(f'Epoch {epoch + 1}')
    model.train()
    epoch_loss = 0
    for idx, data in enumerate(train_loader):
        ct = data["ct"].to(device)
        mr = data["mri"].to(device)

        optimizer.zero_grad(set_to_none=True)
        timestep = torch.randint(0, diffusion_steps, (len(ct),)).to(device)
        with autocast(enabled=True):
            # offset noise for deeper colors
            noise = torch.randn_like(ct).to(device)
            noisy_images = scheduler.add_noise(original_samples=ct, noise=noise, timesteps=timestep)

            combined = torch.cat((noisy_images, mr), dim=1)

            noise_prediction = model(
                x=combined,
                time=timestep,
            )
            loss = F.mse_loss(noise_prediction.float(), noise.float())
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # lr_scheduler.step()

        epoch_loss += loss.item()
    train_epoch_loss = epoch_loss / (idx + 1)
    p_bar.set_postfix({'Epoch Loss': train_epoch_loss})
    # track learning rates across the epochs
    # lrs.append(lr_scheduler.get_last_lr()[0])

    epoch_loss_list.append(train_epoch_loss)
    if epoch % val_interval == (val_interval - 1):
        model.eval()
        val_epoch_loss = 0
        for step, data_val in enumerate(val_loader):
            ct = data_val["ct"].to(device)
            mr = data_val["mri"].to(device)
            timesteps = torch.randint(0, 1000, (len(ct),)).to(device)
            with torch.no_grad():
                with autocast(enabled=True):
                    noise = torch.randn_like(ct).to(device)
                    noisy_images = scheduler.add_noise(original_samples=ct, noise=noise, timesteps=timesteps)

                    combined = torch.cat((noisy_images, mr), dim=1)
                    prediction = model(
                        x=combined,
                        time=timesteps,
                    )
                    val_loss = F.mse_loss(prediction.float(), noise.float())
            val_epoch_loss += val_loss.item()
        print("")
        curr_val_epoch_loss = val_epoch_loss / (step + 1)
        print("Epoch", epoch, "Validation loss", curr_val_epoch_loss)
        wandb.log({
            "epoch": epoch,
            "train_loss": train_epoch_loss,
            "val_loss": curr_val_epoch_loss,
            # "last_learning_rate": lrs[-1],
        })
        val_epoch_loss_list.append(curr_val_epoch_loss)

        # record model if the validation is currently best
        if curr_val_epoch_loss < best_val:
            best_val = curr_val_epoch_loss
            best_model_epoch = epoch
            torch.save(model.state_dict(), os.path.join(model_out, f"model_{epoch:04d}.pt"))

torch.save(model.state_dict(), os.path.join(out_dir, "ddpm_unetr.pt"))
total_time = time.time() - total_start
print(f"train diffusion completed, total time: {total_time}.")

# load the state of the best model
model.load_state_dict(torch.load(os.path.join(model_out, f"model_{best_model_epoch:04d}.pt")))


##########################
# Plot the LEARNING CURVES
##########################
seaborn_brights = [x for x in plt.style.available if x.startswith('seaborn') and 'bright' in x]
plt.style.use(seaborn_brights[0] if len(seaborn_brights) > 0 else "default")
plt.title("Learning Curves Diffusion Model", fontsize=20)
plt.plot(np.linspace(1, n_epochs, n_epochs), epoch_loss_list, color="C0", linewidth=2.0, label="Train")
plt.plot(
    np.linspace(val_interval, n_epochs, int(n_epochs / val_interval)),
    val_epoch_loss_list,
    color="C1",
    linewidth=2.0,
    label="Validation",
)
plt.yticks(fontsize=12)
plt.xticks(fontsize=12)
plt.xlabel("Epochs", fontsize=16)
plt.ylabel("Loss", fontsize=16)
plt.legend(prop={"size": 14})
plt.savefig(os.path.join(img_out, 'learning_curves.png'))


#########################
# Plot the Learning Rates
#########################
# plt.cla()
# plt.title("Learning Rates", fontsize=20)
# plt.plot(lrs, color="C0", linewidth=2.0)
# plt.yticks(fontsize=12)
# plt.xticks(fontsize=12)
# plt.xlabel("Epochs", fontsize=16)
# plt.ylabel("LR", fontsize=16)
# plt.legend(prop={"size": 14})
# plt.savefig(os.path.join(img_out, 'learning_rates.png'))


#############################
# Plot the evaluation results
#############################
from monai.metrics import SSIMMetric, MultiScaleSSIMMetric, MAEMetric, PSNRMetric

ssim = SSIMMetric(spatial_dims=2)
ms_ssim = MultiScaleSSIMMetric(spatial_dims=2)
psnr = PSNRMetric(max_val=1.0)
mae = MAEMetric()


wandb_table = wandb.Table(columns=['MRI', 'CT', 'sCT', 'SSIM', 'PSNR', 'histogram'])

model.eval()

n = 5
for idx, data in enumerate(tqdm(val_loader.dataset)):
    # evaluate first 5 images
    if idx == 5:
        break
    inputct = data["ct"][0, ...]  # Pick an input slice of the validation set to be segmented
    inputmr = data["mri"][0, ...]  # Check out the ground truth label mask. If it is empty, pick another input slice.

    input_ct = inputct[None, None, ...].to(device)
    input_mr = inputmr[None, None, ...].to(device)
    ensemble = []
    for k in range(n):
        noise = torch.randn_like(input_ct).to(device)
        current_img = noise

        combined = torch.cat((noise, input_mr), dim=1)

        scheduler.set_timesteps(num_inference_steps=1000)
        chain = torch.zeros(current_img.shape)
        for t in scheduler.timesteps:
            with autocast(enabled=False):
                with torch.no_grad():
                    model_output = model(
                        x=combined,
                        time=torch.Tensor((t,)).to(device),
                    )
                    current_img, _ = scheduler.step(model_output, t, current_img)
                    if t % 100 == 0:
                        chain = torch.cat((chain, current_img.cpu()), dim=-1)
                    combined = torch.cat((current_img, input_mr), dim=1)

        img_data_dir = os.path.join(img_out, f"{idx:02d}")
        if not os.path.exists(img_data_dir):
            os.makedirs(img_data_dir)

        plt.cla()
        plt.style.use("default")
        plt.imshow(chain[0, 0, ..., img_size[0]:].cpu(), vmin=0, vmax=1, cmap="gray")
        plt.tight_layout()
        plt.axis("off")
        plt.savefig(os.path.join(img_data_dir, f"progression_{k}.png"))

        compare = torch.cat((input_mr.cpu(), input_ct.cpu(), current_img.cpu()), dim=-1)
        plt.cla()
        plt.figure(f"Comparison", figsize=(12,8))
        plt.style.use("default")
        plt.imshow(compare[0, 0, ...].cpu(), vmin=0, vmax=1, cmap="gray")
        plt.tight_layout()
        plt.axis("off")
        plt.savefig(os.path.join(img_data_dir, f"comparison_{k}.png"))

        
        curr_psnr = psnr(y_pred=current_img, y=input_ct).item()
        curr_ssim = ssim(y_pred=current_img, y=input_ct).item()
        if k == 0:
            print()
        print(f"====== {idx:02d}_{k} ======")
        print(f"SSIM={curr_ssim}")
        print(f"PSNR={curr_psnr}")
        print("------------------")

        log_mr = input_mr[0, 0, ...].cpu().numpy()
        log_ct = input_ct[0, 0, ...].cpu().numpy()
        log_s_ct = current_img[0, 0, ...].cpu().numpy()

        df = pd.DataFrame(dict(
            series=np.concatenate((["CT"] * len(log_ct.flatten()), ["sCT"] * len(log_s_ct.flatten()))), 
            data=np.concatenate((log_ct.flatten(), log_s_ct.flatten()))
        ))
        histogram = px.histogram(df, x="data", color="series", nbins=100, barmode="overlay")
        histogram.update_layout(xaxis_title="Pixel Value", yaxis_title="Frequency")

        # add row to WandB table
        wandb_table.add_data(
            # MRI
            wandb.Image(log_mr),
            # CT
            wandb.Image(log_ct),
            # synthetic CT
            wandb.Image(log_s_ct),
            # SSIM
            curr_ssim,
            # PSNR
            curr_psnr,
            # Histogram
            wandb.Html(plotly.io.to_html(histogram))
        )

        ensemble.append(current_img)


df = wandb_table.get_dataframe()
min_max_table = wandb.Table(columns=['id', 'objective', 'SSIM', 'PSNR', 'histogram'])
for col in ['SSIM', 'PSNR']:
    # add max values
    max_idx = df[col].idxmax()
    row_max = df.loc[max_idx]
    min_max_table.add_data(
        max_idx,
        'max ' + str(col),
        row_max['SSIM'],
        row_max['PSNR'],
        row_max['histogram']
    )
    # now add min values
    min_idx = df[col].idxmin()
    row_min = df.loc[min_idx]
    min_max_table.add_data(
        min_idx,
        'min ' + str(col),
        row_min['SSIM'],
        row_min['PSNR'],
        row_min['histogram']
    )

wandb.log({
    'Evaluation Table': wandb_table,
    'Min Max Table': min_max_table,
})
wandb.finish()
