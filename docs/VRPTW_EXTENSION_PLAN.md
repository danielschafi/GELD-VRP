# GELD → VRPTW Extension Plan

This document summarizes the planned extension of GELD from TSP to **Capacitated Vehicle Routing with Time Windows (VRPTW)**, following the POMO / Routing-MVMoE formulation. It covers the problem statement, implementation stages, data acquisition, design decisions, and risks.

**Reference implementation:** `resources/Routing-MVMoE/envs/VRPTWEnv.py`

**Current GELD baseline:** TSP-only (`geld/env/base.py`, `geld/model/geld_model.py`)

---

## 1. Problem statement

### 1.1 What we solve

We target **Euclidean VRPTW** with:


| Element           | Setting (MVMoE / POMoE-style)                                  |
| ----------------- | -------------------------------------------------------------- |
| Depot             | Single depot (node index 0)                                    |
| Customers         | Capacity demand, service time, `[tw_start, tw_end]`            |
| Travel time       | Euclidean distance / `speed` (default `speed = 1.0`)           |
| Capacity          | Normalized to 1.0; demand in `(0, 1]` per customer             |
| Depot time window | `[depot_start, depot_end]` (default `[0, 3]`)                  |
| Objective         | **Minimize total travel distance** (all legs, including depot) |
| Fleet size        | **Unlimited** — no fixed number of vehicles                    |
| Route structure   | One autoregressive sequence with **repeated depot visits**     |


### 1.2 Solution encoding

The policy builds **one sequential node sequence**, e.g.:

```
depot → c3 → c7 → depot → c1 → c5 → depot → c2 → … → done
```

Each block between depot visits is one **trip** (one capacity cycle). Customers appear **at most once** globally across the full sequence.

### 1.3 Semantics of depot reset (important)

On every depot visit the environment:

- **Refills capacity** to 1.0
- **Resets clock** to 0
- **Resets per-trip distance counter**

This is **not** one physical truck on a shared timeline (trip 2 would start when trip 1 ends). It models **a fresh virtual vehicle** leaving at `t = 0` for each trip.

**Result equivalence:** The feasible outputs correspond to partitioning customers into multiple depot-out-and-back routes, each independently feasible from `(depot, t=0, full capacity)` — i.e. **unlimited identical vehicles all departing at t=0**, not parallel construction but parallel semantics in the solution.

**Not modeled:**

- Fixed fleet size (Solomon-style `m` vehicles)
- Minimize number of vehicles
- Shared global clock across trips on one truck
- Depot congestion / staggered departures

### 1.4 Constraint enforcement

Constraints are enforced via **hard masking** (`ninf_mask = -inf`), not penalty terms in the loss:

1. **Visited customers** — cannot revisit (depot is special; see below)
2. **Capacity** — cannot serve customer if `remaining_load < demand`
3. **Time windows** — arrival (with waiting) must be ≤ `tw_end`
4. **Return to depot** — from each candidate, must be able to return to depot before `depot_end` (closed-route VRPTW)

The decoder adds `ninf_mask` to logits before softmax; infeasible actions receive zero probability.

### 1.5 When is the depot selectable?

Depot is **not** only available when the vehicle is full.


| Situation                         | Depot as next node?                                          |
| --------------------------------- | ------------------------------------------------------------ |
| At a customer                     | Yes, if return is TW-feasible (even with capacity remaining) |
| At depot                          | No — must leave to a customer (unless episode finished)      |
| No customer fits (capacity or TW) | Often depot is the only feasible move                        |
| All customers served              | Depot unmasked to close / finish                             |


Early return with partial load is allowed when the mask permits it.

---

## 2. Architecture changes (GELD → VRPTW)

### 2.1 Global Encoder (GE)


| Current (TSP)                 | Planned (VRPTW)                                                 |
| ----------------------------- | --------------------------------------------------------------- |
| `Linear(2, d)` on coordinates | Customers: `Linear(5, d)` on `(x, y, demand, tw_start, tw_end)` |
| Same embedding for all nodes  | **Separate depot embedding** (2D coords or padded features)     |


Static TW information lives in the encoder; it informs *which* customers exist and their windows, not whether a move is legal (that is the mask).

### 2.2 Local Decoder (LD)


| Current (TSP)                       | Planned (VRPTW)                                                                     |
| ----------------------------------- | ----------------------------------------------------------------------------------- |
| First + last + k-NN node embeddings | Depot anchor + last visited + **k feasible** neighbors                              |
| Visit mask only (via prob scatter)  | `**ninf_mask` from env** applied before softmax                                     |
| No dynamic scalar context           | **Dynamic context:** `remaining_capacity`, `current_time` (optional: dist-to-depot) |


### 2.3 Environment

New `VRPTWEnvironment` (port from `VRPTWEnv`):

- State: `load`, `current_time`, `current_coord`, `selected_node_list`, `ninf_mask`, `at_the_depot`, `finished`
- `done` when all customers visited (variable episode length)
- Reward (for RL / eval): negative total travel distance at episode end

### 2.4 k-NN candidate selection

**Risk:** Distance-only k-NN at inference can exclude TW-feasible nodes.

**Recommended approach (stage 1–2):**

- **Training:** consider all **feasible** unvisited nodes (like POMO/MVMoE at train time)
- **Inference:** sort unvisited by distance; take the first **k nodes that pass `ninf_mask`** (k-feasible-by-distance)

Optional later: urgency-weighted scoring (e.g. factor in remaining TW slack).

---

## 3. Implementation stages

### Stage 1 — Core VRPTW + supervised learning (greedy only)

**Goal:** End-to-end SL pipeline at moderate n (e.g. 20–100).

**Deliverables:**

- `geld/env/vrptw.py` — VRPTW environment with masking and dynamic state
- GE: 5-dim customer features + separate depot embedding
- LD: dynamic `(load, current_time)` + k-feasible candidates + `ninf_mask` before softmax
- Greedy decoding only (extend `InferenceSolver._run_greedy` for VRPTW termination)
- SL trainer on labeled routes (depot-inclusive action sequences)
- Data loader for supervised instances + label tours

**Success criteria:** Feasible greedy tours on validation set; SL loss decreases; tour distance reasonable vs. labels.

**Out of scope for stage 1:** Beam search, PRC, stage-2 curriculum, large-n scaling.

---

### Stage 2 — Beam search decoding

**Goal:** Improve inference via beam search over masked LD probabilities.

**Deliverables:**

- Per-beam env state: `load`, `current_time`, `ninf_mask` for each beam branch
- Beam visit mask: **do not permanently mask depot** (align with env depot-revisit logic)
- k-feasible candidate selection under beam
- Integrate into `InferenceSolver._run_beam`

**Notes:** Beam machinery unchanged in principle — it maximizes probability over already-masked logits. Env must be the source of truth for feasibility.

---

### Stage 3 — PRC on single-route segments (depot → depot)

**Goal:** Post-processing repair analogous to GELD TSP PRC, scoped to VRPTW.

**Segment definition:** One route between consecutive depot visits:

```
depot → c1 → … → ck → depot
```

**Boundary conditions (fixed interface repair):**

- Fix entry state: `(time_in, load_in)` at segment start (from full-tour simulation)
- Pin exit state: `(time_out, load_out)` must match original segment interface so the suffix tour remains valid
- Reorder / reroute **interior customers only** (same multiset)
- Accept repair if segment distance decreases (and full tour remains feasible)

**Deliverables:**

- Route splitter: full sequence → list of depot-to-depot segments
- Sub-env initialized with boundary state (not `t=0`, full load from scratch unless segment starts at depot)
- `apply_prc_iteration` variant for VRPTW
- Full-tour feasibility check after acceptance

**Out of scope for stage 3:** Cross-route segments, global roll/flip augmentations, large-n PRC.

---

### Stage 4 — Self-improvement learning (SIL)

**Goal:** Stage-2-style training using improved pseudo-labels.

**Rationale:** PRC-improved tours are stronger labels than greedy/beam alone. SIL after PRC is preferred over SIL-before-PRC for VRPTW.

**Deliverables:**

- Generate pseudo-labels: greedy/beam → PRC on training set
- SIL trainer at fixed n (no full GELD 100→10k curriculum initially)
- Metrics: distance vs. SL baseline, vs. OR-Tools/HGS where available

**Scope constraint (time):** Fixed problem size or mild curriculum; fewer PRC iterations than full-scale GELD TSP.

---

### Optional future stage — RL (POMO-style)

If SL/SIL plateaus:

- POMO parallel rollouts + REINFORCE on `-distance`
- Same GE/LD/k-feasible architecture; swap trainer for MVMoE-style `_train_one_batch`
- Optional: SL pretrain → short RL fine-tune

---

## 4. Data acquisition

### 4.1 Instance fields (required)

Per instance:

- `depot_xy` — shape `(2,)`
- `node_xy` — shape `(n, 2)`
- `node_demand` — normalized by vehicle capacity
- `service_time` — per customer
- `tw_start`, `tw_end` — per customer

### 4.2 Supervised labels (stage 1)

Labels must be **full action sequences** including depot revisits:

- Node index sequence the env would accept step-by-step
- Each step must be feasible under `ninf_mask` logic
- Prefer optimal or high-quality routes (OR-Tools, HGS, pyVRP)

**Label alignment caveat:** Solomon/HGS solutions use fixed fleet and global time per route. When flattening m routes into one depot-separated sequence, verify feasibility under **reset-at-depot** semantics (each trip starts at `t=0`). Routes that are feasible independently from depot at t=0 align well; labels assuming one shared clock on a single truck do not.

### 4.3 Sources


| Source                   | Use                                     | Location / tool                                 |
| ------------------------ | --------------------------------------- | ----------------------------------------------- |
| MVMoE random generator   | Training instances, consistent with env | `VRPTWEnv.get_random_problems()`                |
| MVMoE `.pkl` datasets    | Validation, benchmarking                | `resources/Routing-MVMoE/data/vrptw/`           |
| MVMoE `generate_data.py` | Bulk instance generation                | `resources/Routing-MVMoE/generate_data.py`      |
| Solomon benchmarks       | Evaluation (mind semantics gap)         | `resources/Routing-MVMoE/data/Vrp-Set-Solomon/` |
| HGS / OR-Tools labels    | SL labels, opt gaps                     | `resources/Routing-MVMoE/baselines/`            |
| Custom supervised set    | Stage 1 primary training                | TBD — user-provided                             |


### 4.4 Recommended data pipeline

1. **Train:** user-provided labeled instances OR MVMoE random + HGS labels
2. **Val:** MVMoE `vrptw{n}_uniform.pkl` + HGS optimal gaps
3. **Test (optional):** Solomon (report semantic differences in eval notes)

Store format suggestion: extend LEHD-like format or reuse MVMoE pickle tuple  
`(depot_xy, node_xy, node_demand, capacity, service_time, tw_start, tw_end)` + separate label sequence file.

---

## 5. Training paradigm


| Phase    | Method                                | Labels                                   |
| -------- | ------------------------------------- | ---------------------------------------- |
| Stage 1  | Supervised learning (teacher forcing) | External optimal/heuristic routes        |
| Stage 2  | — (inference only)                    | —                                        |
| Stage 3  | — (post-processing)                   | —                                        |
| Stage 4  | Self-improvement learning             | PRC-improved tours from stage 3 pipeline |
| Optional | POMO + REINFORCE                      | None (reward = −distance)                |


Stage 1 loss: negative log-likelihood of teacher actions (same pattern as `geld/training/sl_trainer.py`).

No constraint penalty in the loss — feasibility is guaranteed by masking during rollout.

---

## 6. Risks and mitigations


| Risk                                     | Impact                                               | Mitigation                                                          |
| ---------------------------------------- | ---------------------------------------------------- | ------------------------------------------------------------------- |
| **k-NN excludes TW-feasible nodes**      | Unserved customers, invalid solutions                | k-feasible-by-distance; train on all feasible nodes                 |
| **Solomon / HGS label mismatch**         | SL learns infeasible or inconsistent actions         | Flatten routes; validate each step under our env; filter bad labels |
| **Reset-at-depot ≠ user's real problem** | Correct on benchmark but wrong for fixed-fleet VRPTW | Document semantics; add fixed-fleet env later if needed             |
| **Beam breaks depot revisits**           | Invalid multi-route solutions                        | Do not permanently mask depot in beam; per-beam env state           |
| **PRC breaks suffix feasibility**        | Improved segment, invalid full tour                  | Single-route segments; pin exit `(time, load)`; verify full tour    |
| **SL quality ceiling**                   | Stops improving below OR-Tools                       | Stage 4 SIL; optional RL fine-tune                                  |
| **GELD large-n curriculum**              | Out of time budget                                   | Stage 4 at fixed n=20–100 only                                      |
| **Local decoder train/inference gap**    | Weaker inference                                     | k-feasible at inference; ablate k                                   |
| **No explicit vehicle limit**            | Too many “virtual routes”                            | Accept for this formulation; count routes in eval only              |


---

## 7. Additional remarks

### 7.1 Relation to current GELD TSP

GELD today (`geld/env/base.py`) builds fixed-length Hamiltonian tours with no constraints. VRPTW requires:

- Variable episode length
- Depot as revisitable action
- Dynamic `load` / `current_time` in env and decoder
- Different `done` condition and reward

The GE/LD **pattern** (global encode once, local decode with neighborhood) carries over; the env and feature contract change.

### 7.2 POMO dimension

If adopting POMO for RL or data augmentation: `pomo_size` parallel rollouts differ by **first customer after depot**, not by vehicle ID. Optional for stage 1 (single greedy rollout is enough).

### 7.3 Open VRPTW variant

MVMoE also defines **OVRPTW** (open routes, no return-to-depot constraint). This plan targets **closed VRPTW** only. OVRPTW drops the `fail_return_depot` mask.

### 7.4 Evaluation metrics

Report at minimum:

- Total distance (primary objective)
- Feasibility rate (all customers served, zero mask violations)
- Number of depot-to-depot trips (informative; not optimized)
- Optimality gap vs. HGS/OR-Tools when available

### 7.5 Code references


| Component             | Path                                                                        |
| --------------------- | --------------------------------------------------------------------------- |
| VRPTW env (reference) | `resources/Routing-MVMoE/envs/VRPTWEnv.py`                                  |
| MVMoE model features  | `resources/Routing-MVMoE/models/SINGLEModel.py`                             |
| MVMoE training loop   | `resources/Routing-MVMoE/Trainer.py`                                        |
| GELD env              | `geld/env/base.py`, `geld/env/synthetic.py`                                 |
| GELD model            | `geld/model/geld_model.py`, `geld/model/local_decoder.py`                   |
| GELD beam / PRC       | `geld/search/beam_search.py`, `geld/search/prc.py`, `geld/search/solver.py` |
| GELD SL trainer       | `geld/training/sl_trainer.py`                                               |


---

## 8. Stage summary checklist

```
Stage 1  [ ] VRPTW env   [ ] GE/LD features   [ ] ninf_mask in LD   [ ] Greedy   [ ] SL
Stage 2  [ ] Per-beam state   [ ] Depot-aware beam   [ ] k-feasible under beam
Stage 3  [ ] Route splitter   [ ] Boundary-aware sub-env   [ ] PRC accept/reject
Stage 4  [ ] PRC pseudo-labels   [ ] SIL at fixed n
Future   [ ] RL (POMO)   [ ] Fixed-fleet variant   [ ] Large-n curriculum
```

---

*Last updated: planning document from GELD-VRP design discussions. Revise as implementation proceeds.*