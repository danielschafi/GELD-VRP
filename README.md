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

TSP benchmark instances live in `Test_data/` (synthetic, TSPLIB, National TSP). Baseline tours for post-processing are in `baseline_solutions/`.

CVRPTW benchmark instances (Solomon and Homberger) are stored under `data/test/`:

```
data/test/
├── solomon/     # 56 VRPTW instances (100 customers)
├── homberger/   # 300 VRPTW instances (200–1000 customers)
└── synthetic/
```

Download from [CVRPLIB VRPTW instances](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/instances/2):

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

Each instance is a `.txt` file in Solomon format; matching `.sol` reference solutions are included. If you have `p7zip-full` installed, you can extract with `7z x` instead of the Python snippet above.

MVMoE-style synthetic VRPTW test sets for gap evaluation (optional, used by `geld-cvrptw-eval --benchmark synthetic`):

```bash
mkdir -p data/test/synthetic
curl -fsSL -o data/test/synthetic/vrptw100_uniform.pkl \
  "https://raw.githubusercontent.com/RoyalSkye/Routing-MVMoE/main/data/VRPTW/vrptw100_uniform.pkl"
curl -fsSL -o data/test/synthetic/hgs_vrptw100_uniform.pkl \
  "https://raw.githubusercontent.com/RoyalSkye/Routing-MVMoE/main/data/VRPTW/hgs_vrptw100_uniform.pkl"
```

### CVRPTW evaluation

Evaluate a trained GELD-CVRPTW checkpoint on synthetic, Solomon, and Homberger benchmarks:

```bash
uv run geld-cvrptw-eval \
  --checkpoint-path result/your_run \
  --checkpoint-epoch 49 \
  --benchmark all
```

Use `--benchmark synthetic`, `solomon`, or `homberger` for a single suite. Results are written under `result/<timestamp>/` (`eval_instances.csv`, `eval_summary.json`).

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

## Experiment logging

Each run writes a timestamped folder under `result/` with structured artifacts:

| File | When | Use |
|------|------|-----|
| `log.txt` | always | Human-readable text log |
| `metrics.csv` | training | One row per epoch — load directly in pandas |
| `metrics.json` | training | Same metrics plus run metadata |
| `plots/*.png` | training | Matplotlib curves (loss, lengths, …) |
| `eval_synthetic_summary.csv` | synthetic eval | Size × distribution gap table |
| `eval_instances.csv` | TSPLIB / postprocess | Per-instance gaps |
| `eval_summary.json` | evaluation | Aggregated gap and bucket stats |

Example — load training curves in pandas:

```python
import pandas as pd
df = pd.read_csv("result/20260101_120000_train_sl/metrics.csv")
df.plot(x="epoch", y=["train_loss", "train_reference_length"])
```

Optional Weights & Biases logging:

```bash
uv sync --extra wandb
uv run geld-train-sl --wandb --wandb-run-name sl-baseline
```

Training batch progress is logged every 50 batches by default (plus 10% milestones). Override with `--batch-log-interval`.

## Dependencies

- Python >= 3.9
- PyTorch >= 2.2.2
- numpy, matplotlib, tqdm, pytz
- wandb (optional, `uv sync --extra wandb`)

## Acknowledgements

GELD's implementation is based on [LEHD](https://github.com/CIAM-Group/NCO_code/tree/main/single_objective/LEHD).

### Citation


GELD Paper
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




## Implementation Plan
Implementation follows in a few stages with goals

| #   | Stage                     | Objective / Deliverable                                                                                      |
| --- | ------------------------- | ------------------------------------------------------------------------------------------------------------ |
| 1   | Core CVRPTW               | End-to-End SL pipeline<br>Stage 1 GELD with greedy decoding (No Beam Search yet)                             |
| 2   | Beam Search Decoding      | Improve decoding by using Beam Search instead of Greedy decoding                                             |
| 3   | Re-Construction           | Implement a Re-Construction post-processing that respects the Capacity and TW constraints                    |
| 4   | Self Improvement Learning | Implement Stage 2 of GELD training pipeline with self improvement using improved labels from Re-construction |
**Levels**
1. Stages 1,2 and working on instances $k=n$ (Acceptable MVP)
2. Stages 1,2 done and testing on larger instances (From best know results)
3. Stages 1,2 + 3 (Re-Construction), testing on larger instances
4. Stages 1,2,3, +4 (Self Improvement Learning) testing on larger instances

I am not sure if I can reach all levels in the next 6 weeks, but 1-2 are 99% doable, I think rest is also doable. 
- 4 weeks implementation & testing (writing what I am doing as I am doing it.) 
- 2 weeks for writing

**Note**
- After each stage optimally a test is performed on synthetic + real world instances, this gives us a little ablation study at the end to see if each part contributed for CVRPTW
### Stage 1 - Core CVRPTW
Implement the equivalent of GELD's training Stage-1: Initial problem solving abilities on size $k=n$ problems.
**Steps**
1. Add the CVRPTW environment with the same interface as the current GELD or adapt GELD's to accept MVMoE's environment
2. GE: Adapt to new node feature vector + depot embedding
3. LD: Adapt for k feasible candidates, new dynamic node features (load, current time) and masking in the decoding step
4. Greedy Decoding 
5. Adapt the SL trainer for CVRPTW problem. 
6. Train model
**Goal**
- Training runs on $k=n$ size problems, SL loss decreases
- Inference with greedy decoding possible
- Manual inspection shows valid tours with constraints respected

### Stage 2 - Beam Search Decoding
Improve the results from Stage 1 by changing the greedy decoding to beam search.
This *should* be relatively easy, as it should work exactly the same way as in GELD. The feasibility is enforced through masking by the environment.
**Steps**
- Add Beam Search on top of the LD to replace greedy decoding. 
- Must use per beam an env state with load, current time, and feasibility mask
**Goal**
- Better results 

No retraining necessary, this works with the same probabilities from the LD as greedy 


If we complete this, I think its already a success. Will need to check time after this.

### Stage 3 - Re-Construction
For GELD the Re-Construction post processing brought good improvements to solution quality. 
It needs to be adapted so that TW and capacity constraints can be respected.
Improvement will run only on individual segments between depot visits (depot -> c1 -> ... -> ck -> depot) 

GELDS idea in RC is to split the TSP tour into smaller segments randomly and then improve these segments, keeping start and end points fixed. They do this again using the LD and Beam Search.

This should transfer to Tw constraints to if we just record the entry time and load and exit time and load, then we only need to watch out that the nodes within this sub-segment are visited in a way that respects the TW constraints (capacity could be disregarded since we already know that traveling across these nodes respects the capacity)

**Changes**
- Optimize each segment separately
- Depot node needs to stay fixed, 
- Fix entry 
- Split Segment into sub segments
	- depot must be kept at start and end, we have a specific start and end time that needs to be respected at the depot
	- We can still split segment in sub segments by randomly choosing an index and then always taking m nodes into a sub segment to optimize, but here we will have two segments with node count != m at the sub segment that leaves the depot and at the sub segment that returns to the depot
		- That was the same with GELD's TSP RC, but there only one segment with length not m

**Pseudocode**
1. For t iterations
	1. Split into depot-to-depot segments to optimize individually
	2. Each Segment
		1. split into sub segments
		2. For Each sub segment
			1. record entry and exit time
			2. Decode "tour" for this sub segment 
			3. If distance of new < old -> keep new
		3. Check feasibility (but should be already respected)

**Steps**
1. Adjust the RC to fit the new result structure 
2. Adjust splitting into sub segments
3. For each segment check that decoding works

**Goal**
- RC should be able to improve the solutions
- Respect the constraints

**Remarks**
- Apply same augmentation as in GELD (rotations and some normalization) before splitting into segments.

### Stage 4 - Self improvement learning (SIL)
Gelds Stage 2 of training was done with SIL to not have to rely on pre-computed labels as such are expensive to obtain especially when problem scale increases.
Thats why we need to complete RC to obtain better solutions, than what greedy decoding alone produces. 
Through that we essentially tell the model that it should have predicted what we got after RC in the first place.

This assumes RC+BS improved tour is better than the greedy decoded one.

**Steps**
- Generate pseudo labels using PRC
- As in GELD increase in problem size, but maybe not to from size 100 to 10'000 but only to 2000 or so (just depends on time and our training data)

**Goal**
- SIL increases the models ability to work with bigger problem sizes
