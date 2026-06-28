# GELD-CVRPTW

Extension of [GELD](https://github.com/CIAM-Group/NCO_code) (Global-view Encoder and Local-view Decoder) to the **Capacitated Vehicle Routing Problem with Time Windows (CVRPTW)**.

The model keeps GELD's core design — a lightweight global encoder (RALA) and a heavyweight local decoder (AFM) — and adds CVRPTW-specific node features, feasibility masking, beam-search decoding, and reconstruction post-processing. Capacity and time-window constraints are enforced during decoding via the environment mask.

The original GELD-TSP implementation lives in `geld/` in this repository.

## Installation

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
uv sync
uv sync --extra dev   # include pytest
uv sync --extra wandb # optional Weights & Biases logging
```

## Project structure

```
geld_cvrptw/
├── model/       # GeldCvrptwModel, Global Encoder, Local Decoder
├── env/         # CVRPTW environment (masking, step, dynamic state)
├── data/        # Loaders, augmentations, instance generation, HGS labelling
├── inference/   # Decoders (greedy, beam search), reconstruction, evaluator
├── training/    # Stage-1 supervised learning trainer
└── cli/         # Command-line entry points
```

## Data

### Training data

Stage-1 supervised learning uses RCT-format CVRPTW-100 instances with HGS reference tours.

**Pre-generated data** from [Rethinking Constraint Tightness](https://github.com/CIAM-Group/Rethinking_Constraint_Tightness) (CIAM Group):

- Download from the [Google Drive folder](https://drive.google.com/drive/folders/1KlmVm1fiplF5jZKfVG1SpKqz3bbrxQy-)
- Place the `.pkl` problem files and matching `hgs_*.pkl` label files in `data/training_stage_1/`

Each problem file contains instances in RCT format; each `hgs_*.pkl` file holds the corresponding HGS tour and cost used as training labels.

**Generate additional data** locally with the Hybrid Genetic Search (HGS) labeller:

```bash
uv run python scripts/generate_cvrptw.py --num-samples 10000
```

This writes RCT-format instances and HGS labels to `data/training_stage_1/` by default. Use `--output-dir` to change the destination. The generator follows the same format and HGS settings as the RCT repository.

### Benchmark data

Evaluation supports three benchmark suites under `data/test/`:

| Suite | Source | Location |
|-------|--------|----------|
| **Solomon** | [CVRPLIB VRPTW instances](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/instances/2) | `data/test/solomon/` |
| **Homberger** | [CVRPLIB VRPTW instances](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/instances/2) | `data/test/homberger/` |
| **Synthetic** | [Routing-MVMoE](https://github.com/RoyalSkye/Routing-MVMoE) | `data/test/synthetic/` |

**Solomon and Homberger** — download and extract from CVRPLIB:

```bash
mkdir -p data/test/solomon data/test/homberger

curl -fsSL -o data/test/Solomon.7z \
  "https://galgos.inf.puc-rio.br/cvrplib/index.php/en/download/instance-set/22"
curl -fsSL -o data/test/Homberger.7z \
  "https://galgos.inf.puc-rio.br/cvrplib/index.php/en/download/instance-set/23"

uv run --with py7zr python - <<'EOF'
import py7zr
from pathlib import Path

root = Path("data/test")
for archive, subdir, nested in [
    ("Solomon.7z", "solomon", "Solomon"),
    ("Homberger.7z", "homberger", "Holmberger"),
]:
    out = root / subdir
    with py7zr.SevenZipFile(root / archive, mode="r") as z:
        z.extractall(path=out)
    nested_dir = out / nested
    if nested_dir.is_dir():
        for path in nested_dir.iterdir():
            path.rename(out / path.name)
        nested_dir.rmdir()
    (root / archive).unlink()
EOF
```

Each instance is a `.txt` file in Solomon format; matching `.sol` reference solutions are included.

**Synthetic** — MVMoE-style uniform VRPTW-100 instances with HGS reference costs:

```bash
mkdir -p data/test/synthetic
curl -fsSL -o data/test/synthetic/vrptw100_uniform.pkl \
  "https://raw.githubusercontent.com/RoyalSkye/Routing-MVMoE/main/data/VRPTW/vrptw100_uniform.pkl"
curl -fsSL -o data/test/synthetic/hgs_vrptw100_uniform.pkl \
  "https://raw.githubusercontent.com/RoyalSkye/Routing-MVMoE/main/data/VRPTW/hgs_vrptw100_uniform.pkl"
```

## Usage

### Training

Stage 1 — supervised learning on CVRPTW-100 instances with HGS labels:

```bash
uv run geld-cvrptw-train-stage1
```

Common options:

```bash
uv run geld-cvrptw-train-stage1 \
  --epochs 50 \
  --batch-size 1024 \
  --cuda-device 0 \
  --wandb --wandb-run-name cvrptw-stage1
```

Resume from a checkpoint:

```bash
uv run geld-cvrptw-train-stage1 \
  --model-load-path result/your_run \
  --model-load-epoch 49
```

Quick smoke run:

```bash
uv run geld-cvrptw-train-stage1 --debug
```

### Evaluation

Evaluate a trained checkpoint on synthetic, Solomon, and Homberger benchmarks:

```bash
uv run geld-cvrptw-eval \
  --checkpoint-path result/your_run \
  --checkpoint-epoch 49 \
  --benchmark all
```

Use `--benchmark synthetic`, `solomon`, or `homberger` for a single suite.

Decoding defaults to beam search with reconstruction post-processing. Disable either with `--no-beam` or `--no-rc`:

```bash
uv run geld-cvrptw-eval \
  --checkpoint-path result/your_run \
  --checkpoint-epoch 49 \
  --benchmark solomon \
  --no-rc
```

Results are written under `result/<timestamp>/` (`eval_instances.csv`, `eval_summary.json`).

## Experiment logging

Each run writes a timestamped folder under `result/`:

| File | When | Contents |
|------|------|----------|
| `log.txt` | always | Human-readable log |
| `metrics.csv` | training | One row per epoch |
| `metrics.json` | training | Same metrics plus run metadata |
| `plots/*.png` | training | Loss and length curves |
| `eval_instances.csv` | evaluation | Per-instance gaps |
| `eval_summary.json` | evaluation | Aggregated gap statistics |

Example — load training curves:

```python
import pandas as pd
df = pd.read_csv("result/20260101_120000_train_stage_1/metrics.csv")
df.plot(x="epoch", y=["train_loss", "train_reference_length"])
```

## Acknowledgements

GELD-CVRPTW builds on the GELD architecture and training pipeline from:

- **GELD** — [GELD: A unified neural model for efficiently solving traveling salesman problems across different scales](https://github.com/CIAM-Group/NCO_code) (Xiao et al., Pattern Recognition, 2026)
- **LEHD** — original TSP training data and baseline code from the [CIAM Group NCO repository](https://github.com/CIAM-Group/NCO_code/tree/main/single_objective/LEHD)

Training data and generation format follow [Rethinking Constraint Tightness](https://github.com/CIAM-Group/Rethinking_Constraint_Tightness) (CIAM Group). Benchmark instances come from [CVRPLIB](https://galgos.inf.puc-rio.br/cvrplib/) (Solomon, Homberger) and [Routing-MVMoE](https://github.com/RoyalSkye/Routing-MVMoE) (synthetic VRPTW). Reference labels are produced with Hybrid Genetic Search (HGS).

The CVRPTW environment draws on ideas from the [MVMoE VRPTW environment](https://github.com/RoyalSkye/Routing-MVMoE/blob/main/envs/VRPTWEnv.py).

### Citation

If you use GELD, please cite:

```bibtex
@ARTICLE{Xiao2025,
  author={Yubin Xiao and Di Wang and Rui Cao and Xuan Wu and Boyang Li and You Zhou},
  journal={Pattern Recognition},
  title={GELD: A unified neural model for efficiently solving traveling salesman problems across different scales},
  year={2026},
  volume={173},
  pages={1-15},
}
```
