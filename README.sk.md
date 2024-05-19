# Bakalárska práca na tému spracovanie rádiologických dát pomocou hlbokých neurónových sietí
Tento repozitár obsahuje kód pre bakalársku prácu,
ktorá sa zameriava na spracovanie radiologických dát
pomocou hlbokých neurónových sietí.
Konkrétne sa zameriava na syntézu CT obrazu z MRI.

# Nastavenie prostredia

Pomocou Conda vytvorte prostredie spustením nasledujúceho príkazu:

```bash
conda env create --name synthrad --file=environment.yml
```

Prostredie aktivujte spustením:
```bash
conda activate synthrad
```

# Tréning
Ak chcete trénovať model s predvolenými hodnotami
pre argumenty príkazového riadka
(t.j. základný model, na CUDA zariadení, bez wandb logovania, počas 2000 epôch),
spustite nasledujúci príkaz:

```bash
python main.py
```

Podporované argumenty príkazového riadka:
| argument | description |
| -------- | ----------- |
| `--wandb_api` | Zadajte váš wandb API kľúč pre povolenie logovania do vášho wandb dashboardu |
| `--torch_device` | Zadajte názov torch zariadenia pre prepísanie predvoleného `cuda` |
| `--epochs` | Zadajte počet tréningových epôch pre prepísanie predvolených 2000 |
| `--model` | Ktorý model trénovať<br>možnosti={baseline, patch, cosine, dropout_lin, dropout_cos}  |


# Porovnania
Na porovnanie natrénovaných modelov môžete spustiť pripravené benchmarky.

```bash
python benchmark.py --benchmark <benchmark_id>
```
Tento príkaz sám o sebe nestačí a každý benchmark potrebuje
ďalšie informácie o cestách k uloženým stavom modelov.
Detaily sú uvedené nižšie.

Podporované `benchmark_id` sú 1, 2, 3 a 4:

##### 1. Základná metóda vs. Vnorenie záplat
Benchmark s `benchmark_id = 1` porovnáva základný model
s modelom s vnorenými záplatami a ich pozíciami.

Ak chcete spustiť tento benchmark,
zahrňte argumenty `--model_base` a `--model_patch`,
ktoré špecifikujú cestu k uloženým stavom modelov:

```bash
python benchmark.py --benchmark 1 --model_base <model_baseline_path> --model_patch <model_patch_path>
```

##### 2. Lineárny plánovač vs. Kosínusový plánovač
Benchmark s `benchmark_id = 2` porovnáva základný model používajúci
lineárny plánovač so základným modelom používajúcim kosínusový plánovač.

Ak chcete spustiť tento benchmark,
zahrňte argumenty `--model_lin` a `--model_cos`,
ktoré špecifikujú cestu k uloženým stavom modelov:

```bash
python benchmark.py --benchmark 2 --model_lin <model_linear_path> --model_cos <model_cosine_path>
```

##### 3 Nízký počet inferenčných krokov na lineárnom vs. kosínusovom plánovači
Benchmark s `benchmark_id = 2` porovnáva základný model používajúci lineárny rozvrh
so základným modelom používajúcim kosínusový rozvrh,
ale tentokrát na nízkom (meniacom sa) počte inferenčných krokov.

Ak chcete spustiť tento benchmark, zahrňte argumenty `--model_lin` a `--model_cos`,
ktoré špecifikujú cestu k uloženým stavom modelov.
Okrem toho, na ovládanie počtu inferenčných krokov, zahrňte argument `--inference_steps`:

```bash
python benchmark.py --benchmark 3 --model_lin <model_linear_path> --model_cos <model_cosine_path> --inference_steps <num_of_steps>
```

##### 4. Základný model s dropout pri použití lineárneho vs. kosínusového plánovača
Benchmark s `benchmark_id = 4` porovnáva základný model s dropout používajúci lineárny rozvrh
so základným modelom používajúcim kosínusový rozvrh.

Ak chcete spustiť tento benchmark, zahrňte argumenty `--model_lin` a `--model_cos`,
ktoré špecifikujú cestu k uloženým stavom modelov.
Možnosť ovládať počet inferenčných krokov bola ponechaná
z predchádzajúceho (3) benchmarku.
Na ovládanie počtu inferenčných krokov zahrňte argument `--inference_steps`
(ktorý je úplne voliteľný a predvolene používa 1000 inferenčných krokov):

```bash
python benchmark.py --benchmark 4 --model_lin <model_linear_path> --model_cos <model_cosine_path>
```

Podporované argumenty príkazového riadka:
| argument | description |
| -------- | ----------- |
| `--benchmark` | Zadajte id benchmarku, ktorý chcete spustiť (možnosti=[1, 2, 3, 4])|
| `--torch_device` | Zadajte názov torch zariadenia pre prepísanie predvoleného `cuda` |
| `--model_base` | Cesta k uloženému stavu základného modelu |
| `--model_patch` | Cesta k uloženému stavu modelu s vnorenými záplatami |
| `--model_lin` | Cesta k uloženému stavu modelu trénovaného s lineárnym plánovačom |
| `--model_cos` | Cesta k uloženému stavu modelu trénovaného s kosínusovým plánovačom |
| `--inference_steps` | Zadajte počet inferenčných krokov (platí len pre benchmarky 3 a 4) |
| `--wandb_api` | Zadajte váš wandb API kľúč pre povolenie logovania do vášho wandb dashboardu|

