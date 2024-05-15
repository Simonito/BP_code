from benchmarks import (
    baseline_vs_patch,
    linear_vs_cosine,
    lowSampleSteps_lin_vs_cos,
    dropout_lin_vs_cos,
)
from preprocess import get_data_paths, preprocess_data, create_loaders
from argparse import ArgumentParser, ArgumentError


def load_val_dataset(data_path, img_size, batch_size):
    tr_images, val_images = get_data_paths(data_path, shuffle_seed=42)
    tr_images, val_images = preprocess_data(train_images=tr_images, val_images=val_images, img_size=img_size)
    _, val_loader = create_loaders(tr_images, val_images, batch_size=batch_size)
    return val_loader.dataset


parser = ArgumentParser()
parser.add_argument('--benchmark', type=int, help='Which benchmark do you want to run')
args = parser.parse_args()

benchmark_num = args.benchmark
# That is the size, set in the main.py that trains and saves the model, if changed there, change also here
img_size = (168, 168)
# Batch size is small for this small example
batch_size = 1
val_dataset = load_val_dataset('./data', img_size=img_size, batch_size=batch_size)
if benchmark_num == 1:
    baseline_vs_patch.main(val_dataset=val_dataset, img_size=img_size[0])
elif benchmark_num == 2:
    linear_vs_cosine.main(val_dataset=val_dataset, img_size=img_size[0])
elif benchmark_num == 3:
    lowSampleSteps_lin_vs_cos.main(val_dataset=val_dataset, img_size=img_size[0])
elif benchmark_num == 4:
    dropout_lin_vs_cos.main(val_dataset=val_dataset, img_size=img_size[0])
else:
    raise ArgumentError(f'Got unexpected benchmark argument: {benchmark_num}. Valid values are: 1, 2, 3 and 4')