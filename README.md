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
To enable `wandb` logging include *wandb API key* in the
arguments:
```bash
python main.py --wandb_api <YOUR WANDB API KEY>
```
If you want to train the model on another torch device,
include the device name in the arguments
(in this example we are specifying the `cpu` device):
```bash
python main.py --torch_device cpu
```
