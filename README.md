### Environment setup

Using Conda create an environment by running the following command:

```bash
conda env create --name synthrad --file=environment.yml
```

Activate the environment by running:
```bash
conda activate synthrad
```

### Training
To train the model with default values for cli arguments
(that is baseline model, on a cuda device, without wandb logging, during 2000 epochs) run following:
```bash
python main.py
```
Supported CLI arguments:
| argument | description |
| -------- | ----------- |
| `--wandb_api` | Specify yout wandb API key to enable logging to your wandb dashboard|
| `--torch_device` | Specify torch device name to override default `cuda` |
| `--epochs` | Specify the number of training epochs to override the default 2000 |
| `--model` | Which model to train<br>options={baseline, patch, cosine, dropout_lin, dropout_cos}  |


### Benchmarks
To compare the trained models you can run prepared benchmarks.
```bash
python benchmark.py --benchmark <benchmark_id>
```
This command on its own is not enough and each benchmark
needs additional information about the paths to the saved states of models.
Details are below.

Supported `benchmark_id`s are 1, 2, 3 and 4:

##### 1. Baseline vs. Patch
Benchmark with `benchmark_id = 1` compares the baseline model to the one
with patch and position embeddings.

To run this benchmark, include arguments `--model_base` and `model_patch`
that specify the path to the saved states of the models:
```bash
python benchmark.py --benchmark 1 --model_base <model_baseline_path> --model_patch <model_patch_path>
```

##### 2. Linear vs. Cosine
Benchmark with `benchmark_id = 2` compares the baseline model using linear schedule
with the baseline model using cosine schedule.

To run this benchmark, include arguments `--model_lin` and `model_cos`
that specify the path to the saved states of the models:
```bash
python benchmark.py --benchmark 2 --model_lin <model_linear_path> --model_cos <model_cosine_path>
```

##### 3. Low Number of Inference Steps on Linear vs. Cosine Schedules
Benchmark with `benchmark_id = 3` compares the baseline model using linear schedule
with the baseline model using cosine schedule, but this time on a low (variable) number
of inference steps.

To run this benchmark, include arguments `--model_lin` and `model_cos`
that specify the path to the saved states of the models.
Additionally, to control the number of inference steps include argument `--inference_steps`:
```bash
python benchmark.py --benchmark 3 --model_lin <model_linear_path> --model_cos <model_cosine_path> --inference_steps <num_of_steps>
```

##### 4. Baseline model with Dropout on Linear vs. Cosine Schedules
Benchmark with `benchmark_id = 4` compares the baseline model with *dropout*
using linear schedule with the baseline model using cosine schedule.

To run this benchmark, include arguments `--model_lin` and `model_cos`
that specify the path to the saved states of the models.
The option to control the number of inference steps was kept from previous (3) benchmark.
To control the number of inference steps include argument `--inference_steps`
(which is completely optional and by default uses 1000 inference steps):
```bash
python benchmark.py --benchmark 4 --model_lin <model_linear_path> --model_cos <model_cosine_path>
```

Supported CLI arguments:
| argument | description |
| -------- | ----------- |
| `--benchmark` | Specify id of the benchmark you want to run (options=[1, 2, 3, 4])|
| `--torch_device` | Specify torch device name to override default `cuda` |
| `--model_base` | Path to the saved state of the baseline model |
| `--model_patch` | Path to the saved state of the model with patch embeddings |
| `--model_lin` | Path to the saved state of the model trained with linear schedule |
| `--model_cos` | Path to the saved state of the model trained with cosine schedule |
| `--inference_steps` | Specify the number of inference steps (applicable only to benchmarks 3 and 4) |
| `--wandb_api` | Specify yout wandb API key to enable logging to your wandb dashboard|

