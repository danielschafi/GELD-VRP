<h1 align="center">GELD: A unified neural model for efficiently solving traveling salesman problems across different scales</h1>

Global-view Encoder and Local-view Decoder (GELD) for TSP construction and PRC repair.

## Installation

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
uv sync
uv sync --extra dev   # include pytest
```

## Project structure

```
geld/
├── model/       # GeldModel, Global Encoder (RALA), Local Decoder (AFM)
├── env/         # Synthetic and TSPLIB environments
├── data/        # Dataset loaders and augmentations
├── search/      # Beam search, PRC, InferenceSolver
├── training/    # Stage-1 SL and stage-2 curriculum trainers
├── inference/   # Unified evaluator
└── cli/         # Command-line entry points
```

## Datasets

### Training Dataset

Download `train_TSP100_n100w-001.txt` from [LEHD](https://github.com/CIAM-Group/NCO_code/tree/main/single_objective/LEHD) and place it in `SL_training_data/`.

### Test Datasets

Benchmark instances live in `Test_data/` (synthetic, TSPLIB, National TSP). Baseline tours for post-processing are in `baseline_solutions/`.

## Usage

### Training

Stage 1 — supervised pre-training on LEHD data:

```bash
uv run geld-train-sl
```

Stage 2 — curriculum learning with greedy/beam/PRC self-improvement (requires stage-1 checkpoint):

```bash
uv run geld-train-stage2 --model-load-path result/your_sl_run --model-load-epoch 49
```

### Evaluation

Synthetic benchmarks (100–10000 nodes, four distributions):

```bash
uv run geld-eval-synthetic --checkpoint-path result/pre_trained_model --checkpoint-epoch 49
```

National TSP / real-world instances:

```bash
uv run geld-eval-tsplib --checkpoint-path result/pre_trained_model --checkpoint-epoch 49
```

PRC post-processing on baseline solutions:

```bash
uv run geld-eval-tsplib --postprocess --checkpoint-path result/pre_trained_model --checkpoint-epoch 49
```

TSPLIB (instead of National TSP):

```bash
uv run geld-eval-tsplib --tsplib --checkpoint-path result/pre_trained_model --checkpoint-epoch 49
```

### Tests

```bash
uv run pytest
```

## Dependencies

- Python >= 3.9
- PyTorch >= 2.2.2
- numpy, matplotlib, tqdm, pytz

## Acknowledgements

GELD's implementation is based on [LEHD](https://github.com/CIAM-Group/NCO_code/tree/main/single_objective/LEHD).

### Citation

```tex
@ARTICLE{Xiao2025,
  author={Yubin Xiao and Di Wang and Rui Cao and Xuan Wu and Boyang Li and You Zhou},
  journal={Pattern Recognition},
  title={GELD: A unified neural model for efficiently solving traveling salesman problems across different scales},
  year={2026},
  volume={173},
  pages={1-15},
}
```
