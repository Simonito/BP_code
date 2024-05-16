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
To train the model on a cuda device, run following:
```bash
python main.py
```
Supported CLI arguments:
| argument | description |
| -------- | ----------- |
| `--wandb_api` | specify yout wandb API key to enable logging to your wandb dashboard|
| `--torch_device` | specify torch device name to override default `cuda` |
| `--epochs` | specify the number of training epochs to override the default 2000 |

