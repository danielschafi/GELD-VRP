# Extension from GELD TSP to VRP

![[Pasted image 20260610102331.png]]
After the ablation studies it is clear that GELD's approach to solving TSPs is promising. We don't just want to replicate the studies results, but push in a new direction using GELD's framework as a basis. Our chosen direction is the extension to one of the vehicle routing problems. TSPs with more constraints, e.g. time windows constraints, capacity constraints, multiple vehicles etc. making it more difficult than just minimizing tour length in the TSP.

## VRP Variants

- Vehicle Routing Problem: Single depot, multiple vehicles
- Capacitated Vehicle Routing Problem: The vehicles have a limited carrying capacity for goods to be delivered
- Capacitated Vehicle Routing Problem with Time Windows: Same as CVRP but delivery must be made within a timewindows
- Vehicle Routing Problem with Time Windows: Delivery locations have time windows within which the deliveries must be made.
- Vehicle Routing Problem with Profits: Profit attributed to each customer and costs to each edge
- Vehicle Routing Problem with Backhauling: Disjoint set of delivery and pickup customers. Deliver from depots to customers and from customer to depot.
- Vehicle Routing Problem with Pickup and Delivery: Goods need to be moved from some pickup locations to other delivery locations. 
- Vehicle Routing Problem with LIFO: Similar to VRPPD but with restriction to loading of vehicle. At each delivery location, the item to be delivered has to be the most recent to be picked up.
- Multi Depot Vehicle Routing Problem: Multiple depots exist from which vehicles can start and end.

## Challenges of Extension

![[Pasted image 20260610102014.png|266]]
*GELD-TSP architecture*

### Capacitated Vehicle Routing Problem

- Multiple vehicles, capacity constraint
- Enforced by:
  - Masking out nodes that exceed vehicle capacity when predicting probabilities for nodes to visit. Add more context to Local Decoder with information about distance to depot and remaining capacity (MvMoE / POMO)
  - OR: Nodes that a vehicle can service in one run could be pre-computed, afterwards the problem is reduced to solving multiple TSPs (See GLOP paper)
- Estimate: If done by enforcement option 2 then it is easy, we just add a kind of constrained clustering on top of it and then use the existing GELD model to solve the sub problems. 
- If done with masking at prediction time. Then its a bit more involved since the model would need to be extended GE (x,y) -> (x,y, demand) and LD (First, last, knn node embeddings) -> (Depot location, last visited, k nearest feasible neighbors, remaining capacity) these constraints are enforced at the same location as time window constraints. 
- Also needs to allow returns to depot

### Vehicle Routing Problem with Time Windows

- Generally more important for real world applications
- Multiple vehicles, each node has a time windows for delivery
  - Soft: Penalty if not within that timewindows
  - Hard: time windows *must* be satisfied exactly (typical case)
- Service Time: Vehicle needs to stop at customer for some duration. If the vehicle arrives early, it waits until the window opens, then serves
- Enforcement via masking, lagrangian penalty function or Preference Optimization
- Masking seems like the most common option for hard constraints (MvMoE / POMO)

### Capacitated Vehicle Routing Problem with Time Windows (CVRPTW)

- Quite realistic
- Same as VRPTW, but masking needs to mask out infeasible nodes based on TW too and add current time in LD part as dynamic information.
- Extension to Capacitated Problem only minor upgrade over TW as its both a masking mechanism. 
- MVMoE already has an Env (masking, step, rewards) for this problem (could mostly reuse it) only difference, MVMoE used RL and we use SL/SIL getting labels will be harder.

## Decision

Either do CVPR with the pre-selecting of nodes not exceeding a capacity constraint per cluster OR do CVPRTW. If we do VRPTW adding the capacity constraint is basically no effort at all. 

*Because MVMoE paper already has an env that we can partially reuse for CVRPTW I would do CVPRTW.*
[Code for Env](https://github.com/RoyalSkye/Routing-MVMoE/blob/main/envs/VRPTWEnv.py)

## CVRPTW

### Problem Definition

Starting from a depot node, an *unlimited* amount of vehicles try to serve customer nodes demand each within their time window while minimizing the total travel time.

- Single depot
- Constraints
  - Time Windows: Vehicle needs to be at customer node within a certain time window
  - Capacity: Each vehicle has the same fixed capacity, each customer has a demand. The vehicle can serve as many customers demands as it can fit within this capacity. If it runs out of capacity, it needs to return to the depot to load up again.
  - Depot TW: Vehicle must complete trip within the depots time window
- Objective: Minimize the total distance (Euclidean distance) of all vehicles combined.
- Distance matrix: Euclidean, Symmetric
- Fleet size: unlimited
- There are other variations that fix the number of vehicles or have the objective to minimize the number of required vehicles.

For a visit to node `i`:

1. Depart previous node at `current_time` (end of service there).
2. Travel for `dist / speed` → raw arrival `t_arrive = current_time + travel`.
3. Wait if early: `t_service_start = max(t_arrive, tw_start[i])`.
4. Serve for `service_time[i]`.
5. Depart at `t_depart = t_service_start + service_time[i]`.

- `tw_start` = earliest time service can begin (ready time)
- `tw_end` = latest time service can begin (due date)

### Dataset / Benchmark

#### Data Definition

Per instance we need:

- depot coords
- depot time window
- node coords
- node demand 
- time window start and end per customer

#### Supervised Learning Labels

Label must be of same problem type as our problem definition

- The node index sequences in the labels need to be acceptable by our env  
- Need data that is made to optimize Distance primarily, not nr of vehicles
Obtaining this data is quite hard

#### Pre-Labeled Data

[Rethinking Constraint Tightness](https://github.com/CIAM-Group/Rethinking_Constraint_Tightness)
Its set of one million synthetic instances, paired with high-quality HGS reference solutions and varying constraint tightness levels, helps prevent the model from overfitting to specific capacity-to-demand ratios

#### Benchmarks

- Solomon dataset
- Homberger and Gehring datasets
- No real standard training data. Most use RL maybe partially because of that reason. Easier to just compare solution quality than provide gold labels

### Metrics

Report at minimum:

- Total distance (primary objective)
- Feasibility rate (all customers served, zero mask violations)
- Number of depot-to-depot trips (informative; not optimized)
- Optimality gap vs. HGS/OR-Tools when available
- Compare vs other NCO solvers

## Changes

### Constraint Enforcement

Constraints are enforced using hard masking. Setting probabilities of infeasible nodes to 0.
Masked out are:

- Visited: already visited nodes
- Capacity: nodes exceeding the remaining capacity
- Time Windows: arrival (including waiting time) must be within time window start <= arrival <= time window end

### Architecture Changes

#### Data


|               | TSP             | CVRPTW                                                 |
| ------------- | --------------- | ------------------------------------------------------ |
| Node features | Distance matrix | Distance, capacity, time window start, time window end |


#### Global Encoder


|                 | TSP         | CVRPTW                                                                       |
| --------------- | ----------- | ---------------------------------------------------------------------------- |
| Graph Embedding | Linear(2,d) | Linear(5,d) (Added node features)                                            |
| Depot Embedding | -           | just coords or padded features + projection like other nodes (always node 0) |


#### Local Decoder


|                                                 | TSP                         | CVRPTW                                                   |
| ----------------------------------------------- | --------------------------- | -------------------------------------------------------- |
| Input Feature Embeddings                        | First + k-NN + last visited | Depot + k feasible neighbors + last visited              |
| Mask                                            | visited nodes               | all constraints, applied before softmax                  |
| Dynamic information                             | -                           | remaining capacity, current time, evtl distance to depot |
| How should dynamic information be incorporated? |                             |                                                          |


1. Project to embedding space Linear(3, d)
  1. Either last embedding + context embedding
  2. Or add as an additional token (Depot, k feasible, last, context)

#### knn-selection

k nearest selection only based on distance might exclude nodes that are feasible / better suited based on TW or capacity constraint 

- All nodes will be visited, respecting its constraints. But might do so in a suboptimal way with another vehicle (Going there after returning to depot).
- knn should be based on k nearest feasible nodes (tw/capacity constraints will exclude a bigger part of nodes than only visited constraint)
  - if k=n how do we handle nr of feasible < k? needs to be padded
    - zero padding should work, those nodes get masked out anyways for selection and zero padding ensures no influence on other nodes during attention
- Could also do some kind of urgency weighting, combining distance and TW-end e.g. $dist \cdot (-timeUntilTWCloses)$

-> For a first implementation should be fine to just use knn on feasible neighbors and use zero padding or try with standard knn selection and see how it goes

#### Environment

State

- constraint mask
- vehicle load (remaining capacity)
- current time
- current coord
- is at depot
- finished flag
Done when all customers visited

### Remarks

#### Solution Encoding

The output solution will be one continuous node sequence with repeated depot visits, each block between depot visits is decoded for a different vehicle (clock etc gets reset on return).

```
depot → c3 → c7 → depot → c1 → c5 → depot → c2 → … → done
```

#### Depot Reset

On every depot visit, the environment:

- Refills capacity
- Resets clock to 0
- Resets per trip distance counter.
So the result is *not* one continuous tour by one truck with repeated depot visits, but it models a second, third ... vehicle starting also at t=0

**Not modeled:**

- Fixed fleet size (Solomon-style `m` vehicles)
- Minimize number of vehicles
- Shared global clock across trips on one truck
- Depot congestion

#### Supervised- vs Reinforcement Learning

Instead of using SL + SIL like GELD, we could also just use RL (POMO or MVMoE) and just use the Globale-view Encoder + Local-view Decoder structure to predict the next nodes.

We would need to do all the architectural changes too, but the environment, training data, evaluation could be based on an existing paper/repo (MVMMoE). 

For GELD we need to rework the training process.

It is more difficult to obtain ground truth labels for this than for TSP problem, mostly only produced by heuristics (LHK-3 or OR-Tools) or Best known results

But RL potentially less efficient learning?

- Though that is a problem if the state space is too big, which it is not when we constrain it to the k nearest nodes. So it shouldn't take too long to train as well (same as first stage GELD-TSP training I'd guess)

**SIL vs RL**
If SL/SIL plateaus:

- POMO parallel rollouts + REINFORCE on `-distance`
- Same architecture but swap trainer for MVMoE-style RL-Training
- SL pretrain → short RL fine-tune

## Implementation Plan

Implementation follows in a few stages with goals


| #          | Stage                     | Objective / Deliverable                                                                                      |
| ---------- | ------------------------- | ------------------------------------------------------------------------------------------------------------ |
| 1          | Core CVRPTW               | End-to-End SL pipeline Stage 1 GELD with greedy decoding (No Beam Search yet)                                |
| 2          | Beam Search Decoding      | Improve decoding by using Beam Search instead of Greedy decoding                                             |
| 3          | Re-Construction           | Implement a Re-Construction post-processing that respects the Capacity and TW constraints                    |
| 4          | Self Improvement Learning | Implement Stage 2 of GELD training pipeline with self improvement using improved labels from Re-construction |
| **Levels** |                           |                                                                                                              |


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

---

Train Data:
Download from here [https://drive.google.com/drive/folders/1KlmVm1fiplF5jZKfVG1SpKqz3bbrxQy-](https://drive.google.com/drive/folders/1KlmVm1fiplF5jZKfVG1SpKqz3bbrxQy-)
That is train data used in [https://github.com/CIAM-Group/Rethinking_Constraint_Tightness](https://github.com/CIAM-Group/Rethinking_Constraint_Tightness)
Generate additional data using the generate_cvrptw.py script
Delete the 2.5 scale labels (they have missing labels hgs files)