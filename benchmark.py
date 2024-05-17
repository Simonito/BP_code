from benchmarks import (
    baseline_vs_patch,
    linear_vs_cosine,
    lowSampleSteps_lin_vs_cos,
    dropout_lin_vs_cos,
)
from preprocess import get_data_paths, preprocess_data, create_loaders
from argparse import ArgumentParser


def load_val_dataset(data_path, img_size, batch_size):
    tr_images, val_images = get_data_paths(data_path, shuffle_seed=42)
    tr_images, val_images = preprocess_data(train_images=tr_images, val_images=val_images, img_size=img_size)
    _, val_loader = create_loaders(tr_images, val_images, batch_size=batch_size)
    return val_loader.dataset


parser = ArgumentParser()
parser.add_argument('--benchmark', type=int, choices=[1, 2, 3, 4], help='Which benchmark do you want to run')
parser.add_argument('--wandb_api', type=str, help='API key for WandB login')
parser.add_argument('--torch_device', type=str, help='Which torch device to select, default is `cuda`')
parser.add_argument('--model_base', type=str, help='Path to the saved state of the baseline model')
parser.add_argument('--model_patch', type=str, help='Path to the saved state of the model with patch embeddings')
parser.add_argument('--model_lin', type=str, help='Path to the saved state of the model with linear schedule')
parser.add_argument('--model_cos', type=str, help='Path to the saved state of the model with cosine schedule')
parser.add_argument('--inference_steps', type=int, help='Number of inference steps (applicable only with benchmark 3)')
args = parser.parse_args()

benchmark_num = args.benchmark
# That is the size, set in the main.py that trains and saves the model, if changed there, change also here
img_size = (168, 168)
# Batch size is small for this small example
batch_size = 1
val_dataset = load_val_dataset('./data', img_size=img_size, batch_size=batch_size)
if benchmark_num == 1:
    baseline_vs_patch.main(val_dataset=val_dataset,
                           img_size=img_size[0],
                           wandb_api=args.wandb_api,
                           torch_device=args.torch_device,
                           model_base_path=args.model_base,
                           model_patch_path=args.model_patch,
    )
elif benchmark_num == 2:
    linear_vs_cosine.main(val_dataset=val_dataset,
                          img_size=img_size[0],
                          wandb_api=args.wandb_api,
                          torch_device=args.torch_device,
                          model_lin_path=args.model_lin,
                          model_cos_path=args.model_cos,
    )
elif benchmark_num == 3:
    if args.inference_steps is None:
        print('Number of inference steps not specified. Using default 1 000')
    lowSampleSteps_lin_vs_cos.main(val_dataset=val_dataset,
                                   img_size=img_size[0],
                                   wandb_api=args.wandb_api,
                                   torch_device=args.torch_device,
                                   model_lin_path=args.model_lin,
                                   model_cos_path=args.model_cos,
                                   inference_steps=args.inference_steps,
    )
elif benchmark_num == 4:
    dropout_lin_vs_cos.main(val_dataset=val_dataset, img_size=img_size[0])
else:
    raise ValueError(f'Got unexpected benchmark argument: {benchmark_num}. Valid values are: 1, 2, 3 and 4')
