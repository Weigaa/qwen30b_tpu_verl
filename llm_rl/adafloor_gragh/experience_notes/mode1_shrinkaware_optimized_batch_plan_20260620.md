# Mode1 Shrink-Aware Optimized Batch Plan

Date: 2026-06-20

## Problem

The manual `5:2:1` heuristic does not guarantee that every rollout step follows
the desired staged shrink pattern. A step can miss `8 -> 4` when wave2 ranks
still have unfinished requests at the same time as final survivor ranks.

The real trigger condition is rank-level unfinished state, so grouping by mean
response length is too weak. The tail response length and per-rank active KV
peak matter more.

## Optimization Model

For each prompt, compute from the baseline run:

- `max_len`: max of its 16 responses. This approximates rank finish order.
- `peak_active_tokens`: `max_t(t * number_of_responses_with_len >= t)` over
  the 32 responses on a rank after pairing two prompts. This approximates
  preemption/KV pressure better than simple response-length sum.

Each step is split into:

- donor group: 8 ranks, 16 prompts
- wave group: 4 ranks, 8 prompts
- final group: 4 ranks, 8 prompts

Constraints:

- every rank receives exactly 2 prompts
- donor max rank load < wave/final rank load
- wave max rank load < final max rank load
- rank active-token peak <= the configured KV cap

Objective:

- minimize within-step, within-group rank-load spread
- enforce active-token peak as a hard KV/no-preemption constraint
- when multiple final-survivor assignments are feasible, choose the one with
  the smallest global max rank active-token peak

## Implementation

Files:

- `tools/build_mode1_optimized_rank_plan.py`
- `run_mode1_local_shrinkaware_optimized_rank_plan.sh`
- `verl/experimental/dataset/shrink_aware_assignment.py`

The optimizer writes:

- `mode1_shrinkaware_optimized_rank_plan_floor4/oracle/optimized_train.parquet`
- `mode1_shrinkaware_optimized_rank_plan_floor4/oracle/optimized_rank_plan.json`
- `mode1_shrinkaware_optimized_rank_plan_floor4/oracle/optimized_rank_plan_summary.json`
- `mode1_shrinkaware_optimized_rank_plan_floor4/oracle/optimized_length_oracle.json`

The rollout uses `assignment_policy=optimized_rank_plan` so the runtime maps
the current batch's `dataset_item_idx` to the exact rank assignment in the
precomputed plan.

## 2026-06-20 Rank-Topology Correction

The first optimized run used the correct length buckets but an inconsistent
physical rank topology: the offline plan wrote rows in donor -> final -> wave
order while the runtime reorder path uses donor -> wave2 -> final order. As a
result, the medium-tail and long-tail groups were effectively swapped at
runtime. The run therefore produced five `16 -> 8` shrink events but zero true
`8 -> 4` shrink events.

The corrected topology is:

- donor ranks: `0-7`
- wave2 ranks: `8-11`
- final survivor ranks: `12-15`

The optimized train parquet, rank plan, and runtime survivor configuration are
now aligned to donor -> wave2 -> final. Historical response lengths can still be
noisy because rollout sampling and model updates change realized response
lengths, but the previous failure to reach floor 4 was primarily caused by this
rank-role/order mismatch, not merely by stochastic length drift.

## Current Validation

With KV cap `280576`, all 5 steps are feasible after the objective correction:

| step | donor max range | wave max range | final max range | active peak max |
| --- | --- | --- | --- | ---: |
| 1 | 2177-2698 | 7595-7794 | 16384-16384 | 266944 |
| 2 | 3166-3768 | 8023-8396 | 16384-16384 | 262144 |
| 3 | 4499-4892 | 8843-9149 | 16384-16384 | 255456 |
| 4 | 5394-5881 | 9616-10450 | 16384-16384 | 254388 |
| 5 | 6757-7094 | 10881-11221 | 16384-16384 | 225370 |

All steps satisfy:

- `expect_16_to_8=true`
- `expect_8_to_4=true`
- every rank has exactly 2 prompts
- every step has 32 unique prompts

## Practical Notes

If the plan is regenerated with a lower KV cap such as `262144`, some final
survivor assignments may become infeasible even though the staged order is still
correct. In that case the data distribution itself requires either a slightly
higher KV cap, a different sampled dataset, fewer prompts per rank, or a softer
no-preemption threshold.

The dataset contains 21 prompts with `max_len=16384`, so a formulation that
allows only 4 final long prompts per step is too restrictive. The correct
constraint is 4 final ranks * 2 prompts per rank = 8 final prompts per step.

## Formal Optimization Formulation

We model shrink-aware batch construction as a constrained assignment problem.
The notation below follows the current `mode=1, floor=4` setting, but the same
formulation generalizes to other staged-shrink topologies.

### Inputs

Let \(P=\{1,\ldots,N\}\) be the prompt set for one epoch and \(T\) be the
number of rollout steps. In this experiment, \(N=160\) and \(T=5\). Each step
consumes \(B=32\) prompts on \(R=16\) ranks, and each rank receives \(q=2\)
prompts.

For each prompt \(i\in P\), the baseline run provides \(n=16\) response lengths:

$$
\ell_i = (\ell_{i,1},\ldots,\ell_{i,n}).
$$

We derive two prompt-level statistics:

$$
m_i = \max_{j\in[1,n]} \ell_{i,j},
\qquad
s_i = \sum_{j=1}^{n} \ell_{i,j}.
$$

For a set of prompts \(S\) assigned to one rank, we estimate the peak number of
simultaneously live KV tokens as:

$$
a(S)=
\max_{\tau \ge 0}
\left(
  \tau \cdot
  \sum_{i\in S}\sum_{j=1}^{n}\mathbf{1}[\ell_{i,j}\ge \tau]
\right).
$$

The no-preemption budget is \(C\), set to the verified floor-4 KV cap in this
run:

$$
C = 280{,}576.
$$

### Decision Variables

Let \(x_{t,r,i}\in\{0,1\}\) indicate whether prompt \(i\) is assigned to rank
\(r\) in step \(t\). Let

$$
G_D=\{0,\ldots,7\},\qquad
G_W=\{8,\ldots,11\},\qquad
G_F=\{12,\ldots,15\}
$$

denote donor, wave2, and final-survivor ranks respectively. The intermediate
survivor set is \(G_W\cup G_F=\{8,\ldots,15\}\), and the final survivor set is
\(G_F=\{12,\ldots,15\}\). This matches the runtime manual survivor setting
`intermediate_survivor_ranks=[8,9,10,11,12,13,14,15]` and
`final_survivor_ranks=[12,13,14,15]`.

For each rank, define its assigned prompt set:

$$
S_{t,r}=\{i\in P\mid x_{t,r,i}=1\}.
$$

Define the rank completion proxy:

$$
M_{t,r}=\max_{i\in S_{t,r}} m_i,
$$

and the rank active-token peak:

$$
A_{t,r}=a(S_{t,r}).
$$

### Constraints

Each prompt is used exactly once in the epoch:

$$
\sum_{t=1}^{T}\sum_{r=0}^{R-1}x_{t,r,i}=1,\quad \forall i\in P.
$$

Each rank receives exactly two prompts per step:

$$
\sum_{i\in P}x_{t,r,i}=q=2,\quad \forall t,r.
$$

Each step has the desired staged-shrink role composition:

$$
\sum_{r\in G_D}\sum_i x_{t,r,i}=16,\quad
\sum_{r\in G_W}\sum_i x_{t,r,i}=8,\quad
\sum_{r\in G_F}\sum_i x_{t,r,i}=8.
$$

The predicted completion order must permit both shrink stages:

$$
\max_{r\in G_D} M_{t,r}
<
\min_{r\in G_W\cup G_F} M_{t,r},
\quad \forall t,
$$

$$
\max_{r\in G_W} M_{t,r}
<
\max_{r\in G_F} M_{t,r},
\quad \forall t.
$$

The no-preemption capacity constraint is:

$$
A_{t,r}\le C,\quad \forall t,r.
$$

### Objective

We minimize intra-group imbalance in the completion proxy. KV pressure is not
part of the weighted objective; it is a hard feasibility constraint.
For a rank group \(g\in\{G_D,G_W,G_F\}\), define:

$$
\Delta_M(t,g)=\max_{r\in g}M_{t,r}-\min_{r\in g}M_{t,r},
$$

The primary optimization objective is the single scalar:

$$
\min_x
\sum_{t=1}^{T}
\sum_{g\in\{G_D,G_W,G_F\}}
\Delta_M(t,g),
$$

subject to the staged-shrink ordering constraints and the hard KV capacity
constraint \(A_{t,r}\le C\).

When multiple feasible final-survivor assignments satisfy the primary objective,
we use a secondary selection criterion:

$$
x^\dagger =
\arg\min_{x\in\mathcal{X}_{feasible}}
\max_{t,r} A_{t,r}.
$$

This is a lexicographic choice, not a weighted multi-objective formulation: first
find assignments that satisfy the completion-order objective and hard capacity
constraints, then choose the feasible final assignment with the smallest global
maximum rank KV peak.

### Solver Structure

For the current experiment, we exploit monotonicity in \(m_i\). Prompts are
sorted by \(m_i\) and partitioned into donor, wave, and final buckets with
cardinalities \(80\), \(40\), and \(40\) across the epoch.

Donor and wave prompts are assigned to steps by contiguous chunks in sorted
\(m_i\) order. This minimizes the average step-level exit proxy instead of
artificially equalizing difficulty across steps:

$$
G_D(t)=D[(t-1)16:t16],
\qquad
G_W(t)=W[(t-1)8:t8].
$$

Final prompts are handled as a separate feasibility-selection problem. Because
final survivor prompts are intentionally long, contiguous final chunks can
concentrate KV-heavy prompts and violate the cap. The implementation therefore
sorts final prompts by long-tail/KV pressure and distributes them across steps,
then chooses pairings that minimize the maximum pair active-token peak.

Within a step and role group, rank pairing is based on the pairwise active-token
peak:

$$
A(\{i,j\})=
\max_{\tau \ge 0}
\left(
  \tau \cdot
  \sum_{p\in\{i,j\}}\sum_{k=1}^{n}\mathbf{1}[\ell_{p,k}\ge \tau]
\right).
$$

For 4-rank groups, all perfect pairings are enumerated and selected
lexicographically by capacity overflow, group \(M\)-spread, and max
\(A(\{i,j\})\). For the 8-rank donor group, the solver finds a minimax
active-peak perfect matching, then chooses a low-spread matching under that peak
threshold. This keeps solve time small while making KV cap feasibility depend on
\(A(\{i,j\})\), not on max response length alone.

## Concrete Instance From This Run

### Input

- Baseline source: `mode1_baseline_random_batch_floor4/rollout_data` and
  `mode1_baseline_random_batch_floor4/rollout_length`.
- Number of prompts: `160`.
- Responses per prompt: `16`.
- Steps: `5`.
- Prompts per step: `32`.
- Ranks per step: `16`.
- Prompts per rank: `2`.
- KV/no-preemption budget: `280576` active tokens.

### Output Summary

| step | feasible | donor \(M\) range | wave \(M\) range | final \(M\) range | max \(A\) |
| --- | --- | --- | --- | --- | ---: |
| 1 | true | 2177-2698 | 7595-7794 | 16384-16384 | 266944 |
| 2 | true | 3166-3768 | 8023-8396 | 16384-16384 | 262144 |
| 3 | true | 4499-4892 | 8843-9149 | 16384-16384 | 255456 |
| 4 | true | 5394-5881 | 9616-10450 | 16384-16384 | 254388 |
| 5 | true | 6757-7094 | 10881-11221 | 16384-16384 | 225370 |

### Rank-Level Assignment

`row_ids` are indices in `optimized_train.parquet`; `source_ids` are original
dataset indices before optimization. `prompt_max` lists \(m_i\) for the two
prompts on that rank. `rank_max` is \(M_{t,r}\); `active_peak` is \(A_{t,r}\).

| step | rank | role | row_ids | source_ids | prompt_max | rank_max | active_peak | token_sum |
| ---: | ---: | --- | --- | --- | --- | ---: | ---: | ---: |
| 1 | 0 | donor | [0, 1] | [78, 109] | [2070, 2177] | 2177 | 33576 | 51286 |
| 1 | 1 | donor | [2, 3] | [36, 120] | [1624, 2327] | 2327 | 31170 | 52056 |
| 1 | 2 | donor | [4, 5] | [8, 37] | [1949, 2423] | 2423 | 36425 | 54688 |
| 1 | 3 | donor | [6, 7] | [123, 143] | [2172, 2442] | 2442 | 41640 | 57524 |
| 1 | 4 | donor | [8, 9] | [136, 139] | [2069, 2476] | 2476 | 40610 | 54786 |
| 1 | 5 | donor | [10, 11] | [70, 15] | [1442, 2530] | 2530 | 27325 | 51002 |
| 1 | 6 | donor | [12, 13] | [54, 113] | [2074, 2579] | 2579 | 38272 | 55143 |
| 1 | 7 | donor | [14, 15] | [69, 62] | [2639, 2698] | 2698 | 42363 | 61880 |
| 1 | 8 | wave | [16, 17] | [110, 126] | [7229, 7595] | 7595 | 87536 | 155279 |
| 1 | 9 | wave | [18, 19] | [42, 86] | [7306, 7614] | 7614 | 108512 | 169909 |
| 1 | 10 | wave | [20, 21] | [157, 93] | [7114, 7780] | 7780 | 124200 | 175474 |
| 1 | 11 | wave | [22, 23] | [25, 148] | [7152, 7794] | 7794 | 126360 | 172488 |
| 1 | 12 | final | [24, 25] | [46, 31] | [12552, 16384] | 16384 | 163557 | 247517 |
| 1 | 13 | final | [26, 27] | [146, 12] | [14024, 16384] | 16384 | 252735 | 409231 |
| 1 | 14 | final | [28, 29] | [92, 137] | [11344, 16384] | 16384 | 262144 | 400100 |
| 1 | 15 | final | [30, 31] | [26, 74] | [16384, 16384] | 16384 | 266944 | 384166 |

## Length-Aware E2E Rollout-Time Plan

The optimized shrink-aware plan above targets reliable `16 -> 8 -> 4`
shrink behavior. A separate end-to-end rollout-time experiment removes the
donor/wave/final buckets and instead keeps the whole epoch close to a pure
length-sorted schedule.

The implementation is in:

- `tools/build_mode1_length_sorted_e2e_plan.py`
- `run_mode1_local_length_sorted_e2e_floor4.sh`

The main goal is to keep each rollout step internally length-homogeneous, so
earlier steps finish earlier and the final steps contain the long prompts. KV
capacity is treated as a feasibility constraint, not as the primary objective.

### Construction

For the epoch prompt set \(P\), sort all prompts by:

$$
m_i=\max_j \ell_{i,j}.
$$

Then form contiguous 32-prompt rollout steps:

$$
B_t = P_{\text{sorted}}[(t-1)32:t32].
$$

Each step still has 16 ranks and two prompts per rank. Given a step \(B_t\),
the solver assigns prompts to ranks by pairwise matching.

### Conservative KV Proxy

The raw active-token peak for a rank pair \(\{i,j\}\) is:

$$
A(\{i,j\})=
\max_{\tau \ge 0}
\left(
  \tau \cdot
  \sum_{p\in\{i,j\}}\sum_{k=1}^{16}
  \mathbf{1}[\ell_{p,k}\ge \tau]
\right).
$$

Because a later rollout can be longer than the baseline trace, the e2e planner
uses a conservative adjusted peak:

$$
A_\mu(\{i,j\}) =
A\left(\left\{\min(\mu\ell, L_{\max})\mid
\ell\in\ell_i\cup\ell_j\right\}\right),
$$

with:

$$
\mu=1.16,\qquad L_{\max}=16384.
$$

The cap by \(L_{\max}\) is important: a single prompt with all 16 responses
already clipped at 16k should remain bounded by \(16\times 16384=262144\), not
be inflated beyond the physical response-length limit.

The per-rank capacity constraint is:

$$
A_\mu(\{i,j\})\le C,\qquad C=280576.
$$

### Step-Internal Optimal Matching

For one step, there are 32 prompts. The solver builds the complete graph over
those prompts:

$$
|E|=\binom{32}{2}=496.
$$

Each edge \((i,j)\) stores:

- raw \(A(\{i,j\})\)
- adjusted \(A_\mu(\{i,j\})\)
- pair max length \(\max(m_i,m_j)\)
- pair max-length gap \(|m_i-m_j|\)
- pair token sum \(s_i+s_j\)

The step-internal rank assignment now treats KV as a hard constraint and
optimizes rollout-time locality inside each rank. For a fixed 32-prompt step,
let

$$
\mathcal{E}_{\text{cap}}
=\{(i,j): A_\mu(\{i,j\}) \le C_{\text{KV}}\}.
$$

The primary objective is:

$$
\min_{\mathcal{M}\subseteq \mathcal{E}_{\text{cap}}}
\sum_{(i,j)\in\mathcal{M}} |m_i-m_j|,
$$

where \(\mathcal{M}\) is a perfect matching covering all 32 prompts. In words:
as long as every rank's adjusted active-token peak stays under the KV cap, pair
prompts with similar maximum response lengths. This reduces the time one prompt
in a rank sits idle while its partner keeps decoding, and gives the rank a
better chance to exit earlier.

The solver avoids enumerating all perfect matchings:

$$
31!! = 191{,}898{,}783{,}962{,}510{,}625.
$$

Instead it builds the 496-edge graph, drops every edge with
\(A_\mu(\{i,j\}) > C_{\text{KV}}\), then runs a min-weight perfect matching.
The edge weight is dominated by the max-length gap:

$$
w(i,j)=10^6|m_i-m_j|+10^3\max(m_i,m_j)+A_\mu(\{i,j\})+10^{-3}(s_i+s_j).
$$

The large coefficients make the lexicographic priority explicit:

1. minimize total rank-internal max-length gap;
2. prefer lower pair max length;
3. prefer lower adjusted active-token peak;
4. prefer lower total response-token sum.

If no perfect matching exists under the KV cap, the planner falls back to the
old minimax-\(A_\mu\) threshold search only to produce the least-bad infeasible
plan. The cross-step repair pass can then swap prompts with nearby steps until
all steps become cap-feasible again.

Concretely, the solver creates an undirected weighted graph:

$$
G_t=(V_t,E_t),\qquad V_t=B_t,\qquad
E_t=\{(i,j): A_\mu(\{i,j\})\le C_{\text{KV}}\}.
$$

A valid rank assignment is a perfect matching \(\mathcal{M}\): it selects 16
edges and every prompt appears in exactly one selected edge. The planner calls
`networkx.algorithms.matching.min_weight_matching(G_t, weight="weight")`. This
uses a general-graph weighted matching algorithm, i.e. a Blossom-family exact
solver. Therefore the result is not a greedy nearest-neighbor pairing; for the
fixed 32 prompts and the fixed edge weights, it is the global minimum-weight
perfect matching under the KV-cap edge filter.

If the returned matching covers fewer than 32 prompts, the cap-feasible graph
has no perfect matching. That means the current step cannot be made KV-safe by
rank pairing alone, so the planner marks it infeasible and relies on the
neighbor-step repair stage.

### Small Matching Example

Consider 6 prompts assigned to 3 ranks. Suppose the adjusted pair peaks are:

| edge | \(A_\mu\) | edge | \(A_\mu\) | edge | \(A_\mu\) |
| --- | ---: | --- | ---: | --- | ---: |
| p0-p1 | 90 | p1-p2 | 85 | p2-p3 | 95 |
| p0-p2 | 80 | p1-p3 | 75 | p2-p4 | 88 |
| p0-p3 | 70 | p1-p4 | 65 | p2-p5 | 58 |
| p0-p4 | 60 | p1-p5 | 55 | p3-p4 | 92 |
| p0-p5 | 50 |  |  | p3-p5 | 62 |
|  |  |  |  | p4-p5 | 100 |

Assume \(C_{\text{KV}}=75\). The cap-feasible edges are those with
\(A_\mu\le 75\). Also suppose the prompt max lengths are:

| prompt | \(m_i\) |
| --- | ---: |
| p0 | 100 |
| p1 | 108 |
| p2 | 180 |
| p3 | 112 |
| p4 | 185 |
| p5 | 176 |

One cap-feasible matching is:

$$
(p0,p4)=60,\quad (p1,p3)=75,\quad (p2,p5)=58.
$$

Its load-gap cost is:

$$
|100-185|+|108-112|+|180-176|=93.
$$

Another cap-feasible matching is:

$$
(p0,p3)=70,\quad (p1,p4)=65,\quad (p2,p5)=58.
$$

Its load-gap cost is:

$$
|100-112|+|108-185|+|180-176|=93.
$$

Both are safe under the KV cap and have equal primary gap cost, so the
tie-breaker chooses the one with lower pair max length / adjusted peak / token
sum. The important change from the previous minimax-\(A_\mu\) formulation is
that a pair with slightly larger KV peak is acceptable if it keeps the two
prompts' expected exit lengths closer and still stays below the cap.

### Neighbor Step Repair

Pure contiguous length sorting can make the final step too dense with long
prompts. In the 2026-06-21 length-aware run, the original final step had:

$$
\max_r A_\mu(S_{5,r}) = 313775 > 280576,
$$

which matched the observed `Preempting` risk. The planner now performs a local
repair pass instead of failing immediately.

The repair policy is deliberately conservative:

1. identify infeasible steps;
2. choose the nearest neighboring step first;
3. from the infeasible step, try the shortest prompts first, so moving them to
   the earlier neighbor raises the neighbor step's predicted exit as little as
   possible;
4. from the neighbor step, also try the lightest prompts first;
5. swap one prompt at a time;
6. after each candidate swap, recompute the cap-constrained step-internal
   matching for
   both affected steps;
7. accept the swap only if the neighbor remains feasible and the pair of steps
   has lower overflow / lower maximum adjusted peak.

If the short-short candidate window cannot keep reducing overflow, the repair
expands the search window and finally adds a small high-risk prompt window from
the infeasible step as a fallback. This keeps the preferred repair direction
time-friendly without giving up KV feasibility.

This local repair is not a global optimization over all 160 prompts. It is a
bounded heuristic intended to preserve the e2e length-sorted shape while
repairing only the nearest overloaded region.

For the current data, the repair changed only steps 4 and 5:

| swap | moved out of step5 | moved into step5 | step5 max \(A_\mu\) after swap |
| ---: | --- | --- | ---: |
| 1 | src46 | src121 | 309655 |
| 2 | src132 | src82 | 307502 |
| 3 | src101 | src75 | 289334 |
| 4 | src61 | src156 | 278674 |

After repair:

| step | feasible | max raw \(A\) | max adjusted \(A_\mu\) | predicted exit |
| ---: | --- | ---: | ---: | ---: |
| 1 | true | 78990 | 91628 | 3768 |
| 2 | true | 103410 | 119956 | 5881 |
| 3 | true | 172231 | 199788 | 8396 |
| 4 | true | 233709 | 271102 | 16297 |
| 5 | true | 271422 | 278674 | 16384 |

The final repaired step satisfies:

$$
\max_r A_\mu(S_{5,r}) = 278674 < 280576.
$$

### Latest Short-Swap Run

The latest run using the cap-constrained load-gap matching and short-swap repair
is:

`mode1_length_sorted_e2e_gap_repair_shortswap_floor4/logs/wjqwen30b-a3b-record_graph_save4eagle3_20260622193642.txt`

It completed all 5 rollout steps with no `Preempting`.

| step | rollout output time | gen time | step time | response mean | response max | clip ratio |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 421.351 | 421.347 | 515.237 | 2107.2 | 4992 | 0.0000 |
| 2 | 548.403 | 548.400 | 667.911 | 3539.0 | 6537 | 0.0000 |
| 3 | 764.763 | 764.760 | 923.408 | 5304.4 | 9364 | 0.0000 |
| 4 | 1251.102 | 1251.098 | 1493.970 | 8492.9 | 16384 | 0.0098 |
| 5 | 1327.867 | 1327.864 | 1633.911 | 10621.2 | 16384 | 0.2129 |

Total generation time is about `4313.47s`, and total step time is about
`5234.44s`. Compared with the earlier gap-repair run, the short-swap repair
keeps step 5 KV-safe while lowering step 4's planned exit from `16384` to
`16297`, so the final two steps are better balanced for epoch-level rollout
time.

### Optimality Boundary

The step-internal matching is globally optimal for the fixed set of 32 prompts:
it exactly minimizes the sum of rank-internal max-length gaps subject to the
adjusted active-token KV cap. If the cap-feasible graph has no perfect matching,
the fallback minimax-\(A_\mu\) plan is not considered successful by itself; it
exists so the neighbor-step repair can search for feasible swaps.

The cross-step repair is intentionally local. It does not guarantee the best
possible arrangement over all epoch prompts, but it keeps the length-sorted
rollout-time objective mostly intact while repairing KV-cap feasibility with a
small number of nearest-neighbor swaps.

## Adaptive-Floor Length-Sorted E2E Plan

The next version keeps the first objective unchanged: sort prompts by predicted
generation length, then split every 32 prompts into one rollout step. This
preserves the shortest possible epoch-level rollout shape because each step
contains prompts from the same length band.

The new part is step-internal floor selection. For each fixed 32-prompt step:

1. solve the rank-pairing problem with no KV cap to estimate each rank's
   predicted maximum response length;
2. count how many ranks reach the step tail and derive the theoretical minimum
   floor from `{2, 4, 8, 16}`;
3. starting from that theoretical floor, try capacity-constrained minimum-weight
   perfect matching under the corresponding KV cap;
4. if the floor is not KV-feasible, move upward to the next floor;
5. only if even floor16 is infeasible, run the existing nearest-step prompt
   swap repair.

The earlier adaptive implementation used step-internal minimum gap matching:

$$
\min_M \sum_{(i,j)\in M} |m_i - m_j|
$$

subject to:

$$
A_\mu(\{i,j\}) \le C_f,
$$

This made the two prompts on the same rank finish close to each other, but it
did not directly optimize the time at which ranks become reusable. The current
implementation therefore uses release-area matching. Given a fixed length-sorted
32-prompt batch, it first finds the deepest KV-feasible floor and then searches
for the rank assignment that maximizes released rank-time area under that floor.

For a pair \(e=(i,j)\), define:

$$
M_e=\max(m_i,m_j),\qquad A_e=A_\mu(\{i,j\}).
$$

The adjusted active-token proxy remains:

$$
A_\mu(\{i,j\}) =
A(\{\min(\mu l, L_{\max}) : l \in R_i \cup R_j\}).
$$

where \(f \in \{2,4,8,16\}\) is the candidate floor and \(C_f\) is the KV cap
for that floor. The default caps are:

| floor | default cap |
| ---: | ---: |
| 2 | 147456 |
| 4 | 280576 |
| 8 | 377344 |
| 16 | 380800 |

The deployed adaptive role plan now supports more than two shrink stages. The
important rule is that runtime transitions must remain halving transitions such
as `16 -> 8`, `8 -> 4`, and `4 -> 2`. A test run that allowed a planned direct
`8 -> 2` transition exposed a very slow fallback: the `active_ranks=[14, 15]`
shrink entered from
`prev_active_ranks=(8,9,10,11,12,13,14,15)` and used `mode=object_broadcast`
for every layer. Its preload import was about 234 s, while the old optimized
floor2 run shrank from `prev_active_ranks=(12,13,14,15)` to `[14,15]`, used
`mode=direct_npu`, and completed preload import in about 0.12 s. So floor2
should not be implemented as a direct `8 -> 2` jump.

For floor2, the plan now writes `shrink_stages=[8,4,2]` and
`stage_survivor_ranks=[[8..15],[12..15],[14,15]]`. The trigger follows this
stage list, so it can only shrink `16 -> 8`, then `8 -> 4`, then `4 -> 2`.
This preserves the floor2 feasibility objective while avoiding the slow
non-halving import path.

For a candidate floor, feasibility has two parts:

1. KV feasibility: after filtering edges by \(A_e \le C_f\), the graph must
   still admit a perfect matching;
2. shrink usefulness: the selected floor must admit a quota schedule whose
   deepest shrink threshold is strictly before the batch tail \(T\).

The second condition matters for long-tail batches. If more than 8 ranks have
pair completion time \(M_e=T=16384\), a floor8 schedule can only shrink at
\(a=T\). That is KV-safe, but it releases no reusable rank-time, so the planner
does not treat it as a real floor8 solution. It keeps trying shallower shrink
and may finally choose floor16.

The planner selects the deepest floor satisfying both conditions.

For the chosen floor, the planner enumerates candidate shrink schedules from
the possible pair completion times \(M_e\):

- floor8: one threshold \(a\), quota \((a,8)\), release area
  \(8(T-a)\);
- floor4: thresholds \(a \le b\), quotas \((a,8),(b,12)\), release area
  \(8(T-a)+4(T-b)\);
- floor2: thresholds \(a \le b \le c\), quotas
  \((a,8),(b,12),(c,14)\), release area
  \(8(T-a)+4(T-b)+2(T-c)\).

Here \(T=\max_i m_i\) is the predicted tail time of the batch. A quota
\((a,8)\) means at least 8 selected rank pairs must have \(M_e \le a\), so
those ranks can be released at the `16 -> 8` shrink point.

For each schedule, the planner calls a quota-aware matching oracle. The oracle
solves a small exact 0/1 matching problem over the 496 possible pair edges:

- every prompt has degree exactly 1;
- only KV-safe edges \(A_e \le C_f\) may be selected;
- every schedule quota is satisfied;
- within a fixed release-area schedule, ties are broken by smaller
  rank-internal gap, lower pair max length, lower adjusted peak, and lower token
  sum.

Schedules are tried from largest release area to smallest, so the first feasible
oracle result is release-area optimal for the fixed batch and selected floor.
The selected pairs are sorted by \(M_e\) before assigning physical ranks, so for
floor4 ranks `0..7` are the first releasable group, ranks `8..11` are the
second releasable group, and ranks `12..15` are final survivors.

Offline planning result with the older floor2 baseline oracle:

| step | theoretical floor | selected floor | stages | cap | max \(A_\mu\) | feasible |
| ---: | ---: | ---: | --- | ---: | ---: | --- |
| 1 | 2 | 2 | 8,4,2 | 147456 | 91628 | true |
| 2 | 2 | 2 | 8,4,2 | 147456 | 119956 | true |
| 3 | 2 | 2 | 8,4,2 | 147456 | 147100 | true |
| 4 | 2 | 4 | 8,4 | 280576 | 254666 | true |
| 5 | 16 | 16 | 16 | 377344 | 374949 | true |

Run command:

```bash
cd /workspace/cann-recipes-train/llm_rl/qwen3_shrink_aware
./run_mode1_local_length_sorted_e2e_adaptive_floor2.sh
```

The generated plan is written to:

`mode1_length_sorted_e2e_adaptive_floor2/oracle/length_sorted_rank_plan.json`

## Current Adaptive Floor4 Natural/Planner Plan

The stable version used for the latest comparisons is the adaptive floor4
variant:

```bash
./run_mode1_local_length_sorted_e2e_adaptive_floor4.sh
```

It keeps the same mathematical objective, but raises the minimum runtime floor
from 2 to 4:

```text
MIN_ADAPTIVE_FLOOR=4
ACTIVE_PEAK_SAFETY_FACTOR=1.16
VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS_FLOOR2=147456
VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS_FLOOR4=280576
VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS_FLOOR8=377344
VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS=380800
```

For planner mode with floor8/floor4 planned subgroups kept resident across
steps, the warmed communication workspace must be treated as real memory cost.
From the 2026-06-26 logs, rank12-15 keep both the floor8 and floor4 planned
workspaces during floor4 stages and need about `6.5 GiB/rank` extra non-torch
memory after the first real shrink/warmup. With Qwen3 30B-A3B:

```text
KV bytes/token = 2 * 48 layers * 4 kv_heads * 128 head_dim * 2 bytes
               = 98304 bytes/token
147456 tokens ~= 13.5 GiB
114688 tokens ~= 10.5 GiB
```

So planned-residency mode now reserves floor-specific headroom:

```text
VLLM_ASCEND_MODE1_PARITY_PLANNED_FLOOR_GROUP_KV_HEADROOM_TOKENS_FLOOR4=147456
VLLM_ASCEND_MODE1_PARITY_PLANNED_FLOOR_GROUP_KV_HEADROOM_TOKENS_FLOOR8=114688
VLLM_ASCEND_MODE1_PARITY_PLANNED_FLOOR_GROUP_KV_HEADROOM_TOKENS_FLOOR16=0
VLLM_ASCEND_MODE1_PARITY_PRECREATE_COMM_CACHE=1
VLLM_ASCEND_MODE1_PARITY_PRECREATE_DISPATCH_WARMUP=1
```

The effective KV caps used by the offline planner are therefore:

| floor | raw cap | planned workspace headroom | effective cap |
| ---: | ---: | ---: | ---: |
| 2 | 147456 | 147456 | 0 |
| 4 | 280576 | 147456 | 133120 |
| 8 | 377344 | 114688 | 262656 |
| 16 | 380800 | 0 | 380800 |

The objective order is:

1. minimize the rollout time of the whole epoch by keeping the sorted
   32-prompt step buckets;
2. within each step, minimize rank-internal generation-length gaps under the
   KV cap;
3. choose the lowest feasible floor only after the first two conditions are
   fixed;
4. use cross-step prompt swap only if even floor16 is still KV-infeasible.

In other words, KV pressure is a hard feasibility constraint. It is not a second
objective traded against end-to-end time.

### Current Offline Plan

The current floor4 adaptive oracle for planned workspace residency produces:

| step | theoretical floor | selected floor | shrink stages | KV cap | max \(A_\mu\) |
| ---: | ---: | ---: | --- | ---: | ---: |
| 1 | 2 | 4 | 16 -> 8 -> 4 | 133120 | 91628.4 |
| 2 | 2 | 4 | 16 -> 8 -> 4 | 133120 | 119955.6 |
| 3 | 2 | 8 | 16 -> 8 | 262656 | 199787.96 |
| 4 | 2 | 8 | 16 -> 8 | 262656 | 254666.4 |
| 5 | 16 | 16 | no shrink | 380800 | 374949.12 |

Thus the current best deployment plan is:

- steps 1-2 use floor4 and may shrink as `16 -> 8 -> 4`;
- steps 3-4 use floor8 and may shrink as `16 -> 8`;
- step 5 uses floor16 and does not shrink;
- step repair is not needed for step5 because floor16 recovers full-world
  layout and `380800` KV tokens after lower planned groups are pruned.

This is different from the earlier
`mode1_length_sorted_e2e_gap_repair_shortswap_floor4` plan, whose last two
steps were repaired so that both could run under a fixed floor4 cap. In the
adaptive plan, step 5 is allowed to recover the full-world layout and a larger
KV cache, so the old short-swap repair is no longer required unless floor16 is
also infeasible.

### Rank Assignment Solver

For each fixed 32-prompt step, the solver builds a graph with one vertex per
prompt. An edge \((i,j)\) means the two prompts can share one rank. The edge is
kept only if:

$$
A_\mu(\{i,j\}) \le C_f
$$

where:

$$
A_\mu(\{i,j\}) =
A(\{\min(\mu l, L_{\max}) : l \in R_i \cup R_j\}),
\quad \mu=1.16,\quad L_{\max}=16384.
$$

The earlier implementation then minimized only the rank-internal length gap.
The current implementation instead treats shrink timing as the first-class
objective. For a candidate floor, it enumerates feasible shrink schedules and
calls a quota-aware exact matching oracle. The primary objective is to maximize
released rank-time area subject to KV feasibility and schedule quotas; length
gap, pair max length, adjusted peak, and token sum are only tie-breakers inside
a fixed release-area schedule. This aligns the offline assignment objective with
the system-level benefit of making ranks reusable earlier.

### Natural Runtime Policy

`natural` is the default runtime target policy:

```bash
export VLLM_ASCEND_SHRINK_AWARE_TARGET_POLICY=natural
```

It uses the same offline prompt/rank plan and selected floors, but the actual
survivor ranks are determined by the ranks that are still unfinished at runtime.
Therefore, due to generation-length drift, the 8-rank and 4-rank groups may not
be the fixed `[8..15]` and `[12..15]` groups.

This policy is the most realistic system behavior. It does not force a rank to
stay active merely because it belonged to an offline planned survivor set.

Recommended command:

```bash
cd /workspace/cann-recipes-train/llm_rl/qwen3_shrink_aware
VLLM_ASCEND_SHRINK_AWARE_TARGET_POLICY=natural \
./run_mode1_local_length_sorted_e2e_adaptive_floor4.sh
```

The natural policy should not need planned floor-group cache residency:

```text
VLLM_ASCEND_MODE1_PARITY_CACHE_PLANNED_FLOOR_GROUPS=0
VLLM_ASCEND_MODE1_PARITY_CACHE_FLOOR_GROUPS=0
VLLM_ASCEND_MODE1_PARITY_KEEP_MC2_GROUP_CACHE=0
VLLM_ASCEND_MODE1_PARITY_POST_SHRINK_WARMUP=0
```

### Planner Runtime Policy

`planned` fixes the shrink topology to the offline role plan:

```bash
export VLLM_ASCEND_SHRINK_AWARE_TARGET_POLICY=planned
```

For floor4 steps, the planned stages are:

```text
16 -> 8: [8,9,10,11,12,13,14,15]
8 -> 4 : [12,13,14,15]
```

Step 5 has selected floor16, so it stays on full world. This policy is useful
for a fixed-topology comparison with the older planned behavior. It may introduce
dummy run if a planned survivor has already finished while an unplanned rank is
still generating.

Without communication-group residency, planner has no guaranteed throughput
advantage over natural. Its main value is deterministic topology and easier
reproducibility.

Recommended fixed-topology command without floor-group residency:

```bash
cd /workspace/cann-recipes-train/llm_rl/qwen3_shrink_aware
VLLM_ASCEND_SHRINK_AWARE_TARGET_POLICY=planned \
VLLM_ASCEND_MODE1_PARITY_PRECREATE_PLANNED_FLOOR_GROUPS=0 \
VLLM_ASCEND_MODE1_PARITY_CACHE_PLANNED_FLOOR_GROUPS=0 \
./run_mode1_local_length_sorted_e2e_adaptive_floor4.sh
```

If the experiment needs planned floor8/floor4 group residency across steps,
use:

```bash
cd /workspace/cann-recipes-train/llm_rl/qwen3_shrink_aware
VLLM_ASCEND_SHRINK_AWARE_TARGET_POLICY=planned \
VLLM_ASCEND_MODE1_PARITY_PRECREATE_PLANNED_FLOOR_GROUPS=1 \
VLLM_ASCEND_MODE1_PARITY_CACHE_PLANNED_FLOOR_GROUPS=1 \
VLLM_ASCEND_MODE1_PARITY_PRECREATE_COMM_CACHE=1 \
VLLM_ASCEND_MODE1_PARITY_PRECREATE_DISPATCH_WARMUP=1 \
./run_mode1_local_length_sorted_e2e_adaptive_floor4.sh
```

With planner residency enabled, the current calibrated headroom is floor
specific:

```text
VLLM_ASCEND_MODE1_PARITY_PLANNED_FLOOR_GROUP_KV_HEADROOM_TOKENS_FLOOR4=147456
VLLM_ASCEND_MODE1_PARITY_PLANNED_FLOOR_GROUP_KV_HEADROOM_TOKENS_FLOOR8=114688
VLLM_ASCEND_MODE1_PARITY_PLANNED_FLOOR_GROUP_KV_HEADROOM_TOKENS_FLOOR16=0
```

so the effective KV capacities become:

```text
floor4 : 280576 - 147456 = 133120 tokens
floor8 : 377344 - 114688 = 262656 tokens
floor16: 380800 -      0 = 380800 tokens
```

This headroom reserves space for the resident planned floor8/floor4 MoE
communication cache and dispatcher workspaces. Paying this cost enables true
cross-step reuse, but it also means floor selection must be solved with the
reduced effective caps above.

### Communication-Group Cache Policy

The current planned-residency mode is:

```text
VLLM_ASCEND_MODE1_PARITY_PRECREATE_PLANNED_FLOOR_GROUPS=1
VLLM_ASCEND_MODE1_PARITY_CACHE_PLANNED_FLOOR_GROUPS=1
VLLM_ASCEND_MODE1_PARITY_PRECREATE_COMM_CACHE=1
VLLM_ASCEND_MODE1_PARITY_PRECREATE_DISPATCH_WARMUP=1
VLLM_ASCEND_MODE1_PARITY_PLANNED_FLOOR_GROUP_KV_HEADROOM_TOKENS_FLOOR4=147456
VLLM_ASCEND_MODE1_PARITY_PLANNED_FLOOR_GROUP_KV_HEADROOM_TOKENS_FLOOR8=114688
VLLM_ASCEND_MODE1_PARITY_PLANNED_FLOOR_GROUP_KV_HEADROOM_TOKENS_FLOOR16=0
```

`PRECREATE_PLANNED_FLOOR_GROUPS=1` creates the planned floor8/floor4 group
topology before KV sizing. `CACHE_PLANNED_FLOOR_GROUPS=1` means those group
handles may remain resident and be reused across steps. The two precreate
warmup flags mean the heavier MoE communication cache and dispatcher workspace
are also materialized before KV sizing, so the KV allocator sees the real
steady-state footprint.

The previous OOM failure pattern came from precreating floor8/floor4 groups and
materializing/warming their MoE comm cache/dispatcher workspaces while reserving
only `18432` KV tokens of headroom:

```text
cached_ranks=(8,9,10,11,12,13,14,15)
cached_ranks=(12,13,14,15)
comm_cache_layers=96
dispatch_warmup_groups=2
initialize_kv_cache_tensors -> torch.zeros -> OOM
```

The fix is not to disable residency, but to reserve enough KV headroom and pass
the reduced effective caps into the offline planner.

### 2026-06-26 Planner Residency Calibration

The latest planner-residency validation used two complementary logs:

- detailed threshold run:
  `mode1_length_sorted_e2e_adaptive_floor4_commcache_threshold/logs/wjqwen30b-a3b-record_graph_save4eagle3_20260626024759.txt`
- full 5-step run:
  `mode1_length_sorted_e2e_adaptive_floor4/logs/wjqwen30b-a3b-record_graph_save4eagle3_20260626030807.txt`

The detailed threshold run enables `VLLM_ASCEND_MODE1_COMM_CACHE_STATE_LOG=1`,
so it records the resident topology state at step boundaries and KV-resize
points. It shows that the planned communication cache does not grow
monotonically with step index. Instead, residency follows the current selected
floor:

| stage | resident topology by rank band | interpretation |
| --- | --- | --- |
| floor4 | rank0-7: full; rank8-11: full+8; rank12-15: full+8+4 | all planned floor groups needed for `16 -> 8 -> 4` are resident |
| floor8 | rank0-7: full; rank8-15: full+8 | floor4 group is pruned before floor8 KV sizing |
| floor16 | all ranks: full only | floor8/floor4 groups are pruned before full KV sizing |

The measured `non_torch` footprint also explains why the earlier
`81920`-token reservation was too small. Immediately after precreate, the extra
planned floor8/floor4 residency on rank12-15 is only about `3.3 GiB/rank`.
After the first real shrink and MoE dispatcher warmup, the backend HCCL/MC2/TBE
workspace footprint is closer to `6.5 GiB/rank`. Those backend workspaces are
not visible as Python tensor references, so the safe control knob is KV
headroom rather than trying to free a Python object that is not holding the
memory.

The full 5-step run completed rollout/training with:

| step | selected floor | effective KV tokens | rollout output time |
| ---: | ---: | ---: | ---: |
| 1 | 4 | 133120 | 406.902 |
| 2 | 4 | 133120 | 555.768 |
| 3 | 8 | 262656 | 863.593 |
| 4 | 8 | 262656 | 1206.085 |
| 5 | 16 | 380800 | 1397.365 |

The full run had no `Preempting`, `Memory_Allocation_Failure`, `OOM`, or
`Failed to allocate`. It also proved that floor16 can recover the full
`380800`-token KV cache after pruning the lower planned floor groups.

### Validation Checklist

A correct adaptive floor4 run should satisfy:

1. `length_sorted_rank_plan_summary.json` shows step1-step2
   `selected_floor=4`, step3-step4 `selected_floor=8`, and step5
   `selected_floor=16` under the current planner-residency headroom;
2. step1-step2 may shrink as `16 -> 8 -> 4`;
3. step3-step4 may shrink as `16 -> 8`;
4. step5 does not shrink;
5. with planner residency enabled, floor4 KV size is `133,120 tokens`;
6. floor8 KV size is `262,656 tokens`;
7. step5 floor16 KV size is `380,800 tokens`;
8. there is no `Preempting`, `Memory_Allocation_Failure`, or `OOM`;
9. planner residency logs retain planned floor groups only while the next step
   can still use them;
10. floor4 groups are pruned before a floor8 step and floor8/floor4 groups are
   pruned before a floor16 step;
11. planned precreate summary should show floor8/floor4 warmup residency on the
    survivor ranks during floor4 stages, then reduced topology counts in later
    stages.

For natural vs planner comparisons, a single run is not enough because response
lengths can drift. The robust comparison should report, per step:

- actual generated token count;
- `rollout_output_time_s`;
- throughput;
- survivor ranks at 8 and 4;
- dummy-run time, if any;
- resize / precreate / stale-group-release overhead.

## Dynamic Multi-Epoch Length-Aware Shrink

The current end-state workflow is implemented as an outer epoch driver:

```bash
./run_mode1_dynamic_length_aware_adaptive_floor4_epochs.sh
```

The driver turns the latest adaptive planner into a closed loop:

1. epoch0 runs `mode=0` / no shrink, trains normally, and records rollout
   lengths;
2. epoch1 rebuilds the adaptive length-sorted rank plan from epoch0
   `rollout_data` / `rollout_length`;
3. epoch2 rebuilds the next plan from epoch1 rollout lengths;
4. the same pattern repeats for all later epochs.

The key point is that the offline planner is regenerated before every mode1
epoch. This makes batch construction, rank pairing, selected floor, KV caps, and
shrink stages depend on the most recent measured response-length distribution,
not on a stale baseline trace.

### Why Use an Outer Driver

The driver starts a fresh Python/vLLM process for each epoch. This is deliberate:

- epoch0 uses `mode=0`, while later epochs use `mode=1`;
- `optimized_rank_plan_path` is cached inside the runtime process;
- KV cache tensors and elastic communication groups are process-local;
- planned communication-group residency should not accidentally leak from one
  planning epoch into another.

Running each epoch as a child process avoids those state-mixing problems while
still feeding the previous epoch's measured rollout lengths into the next plan.

### Epoch0 Mode0 Warmup/Training

Epoch0 enables:

```text
VLLM_ASCEND_ENABLE_ELASTIC_PARALLEL_SHRINK=0
VLLM_ASCEND_SHRINK_AWARE_ENABLE=0
MODE0_SAVE_ROLLOUT_ARTIFACTS=1
```

Epoch0 is not discarded. It performs the normal reward/logprob/advantage and
actor update path while also dumping `rollout_data` / `rollout_length`. Those
measured lengths seed the epoch1 adaptive planner.

The plan builder can read either:

- `rollout_length/length_*.txt`, or
- `rollout_data/*.jsonl` with `response_mask` if the length files are absent.

It also discovers numeric step files by order, so resumed runs whose global
steps are not named `1..5` can still be used as baseline traces.

### Subsequent Mode1 Epochs

Each later epoch calls:

```bash
./run_mode1_local_length_sorted_e2e_adaptive_floor4.sh
```

with:

```text
BASELINE_DIR=<previous epoch output directory>
PLAN_DIR=<current epoch output directory>/oracle
TRAINER_TOTAL_EPOCHS=1
PLAN_STEPS=5
```

The generated per-epoch outputs include:

- `oracle/length_sorted_train.parquet`
- `oracle/length_sorted_rank_plan.json`
- `oracle/length_sorted_rank_plan_summary.json`
- `oracle/length_sorted_length_oracle.json`
- `rollout_data/`
- `rollout_length/`

For the next epoch, the current output directory becomes the new `BASELINE_DIR`.

### Natural Mode

Natural mode uses runtime-finished ranks as the shrink target:

```bash
cd /workspace/cann-recipes-train/llm_rl/qwen3_shrink_aware
DYNAMIC_TOTAL_EPOCHS=3 \
DYNAMIC_SHRINK_POLICY=natural \
./run_mode1_dynamic_length_aware_adaptive_floor4_epochs.sh
```

It keeps the same adaptive length-sorted plan and per-step floors, but does not
force the fixed planned survivor ranks when realized generation lengths drift.

### Planner-Reuse Mode

Planner-reuse mode fixes the topology to the planned survivor groups and keeps
the planned floor8/floor4 communication groups resident when the selected floor
can use them:

```bash
cd /workspace/cann-recipes-train/llm_rl/qwen3_shrink_aware
DYNAMIC_TOTAL_EPOCHS=3 \
DYNAMIC_SHRINK_POLICY=planned \
./run_mode1_dynamic_length_aware_adaptive_floor4_epochs.sh
```

This is the default mode of the dynamic driver. It inherits the calibrated
planner-residency headroom from
`run_mode1_local_length_sorted_e2e_adaptive_floor4.sh`, so the offline planner
sees the reduced effective KV caps before selecting floors or attempting repair.

### Step-Level Tail Outlier Guard

The adaptive length-sorted plan optimizes batch construction from historical
response lengths, but a synchronous rollout step can still be dominated by a
rare pathological generation. In epoch2 of the planner run, step1 was planned
as a short step with predicted exit around `3768`, while five realized responses
hit `16384`; the step therefore behaved like a long-tail step even though most
responses were short.

The guard is not a fixed `max4096` smoke-test threshold. It is a calibrated
per-step upper bound derived from historical prediction error. For every prompt
we first reduce its `rollout.n=16` responses to a prompt-level tail statistic:

$$
\hat{M}_i=\max_j \hat{L}_{i,j}, \qquad
M_i=\max_j L_{i,j}.
$$

Using the maximum is deliberate: step latency is sensitive to the slowest
response for a prompt, and using the mean would hide exactly the tail behavior
that can dominate wall-clock time.

The per-prompt length predictor itself already uses EMA. The tail guard
therefore does not add another EMA on the error ratio. Instead, for each
adjacent historical transition, the planner predicts epoch \(k\) from the EMA of
epochs before \(k\), then measures the prompt-level underestimation ratio:

$$
r_i^+ =
\max\left(1, \frac{M_i}{\hat{M}_i+\epsilon}\right).
$$

The global calibration factor is the empirical high quantile over a recent
sliding window of adjacent transitions:

$$
\rho_q = Q_q(\{r_i^+ \mid k \in [T-K,T)\}), \qquad q=0.95.
$$

The current default is `K=3`. This keeps the uncertainty calibration close to
the current policy while avoiding the instability of using only the most recent
epoch transition. With only two available historical epochs, it naturally
degenerates to the single transition `epoch0 -> epoch1`.

For a new step \(B_t\), the predicted step tail is the maximum prompt-level
predicted tail inside the step:

$$
S_t = \max_{i\in B_t}\hat{M}_i.
$$

The runtime response cap for that step is then:

$$
C_t =
\min\left(
  L_{\max},
  \max\left(
    C_{\min},
    \left\lceil \frac{\rho_q S_t}{A} \right\rceil A
  \right)
\right),
$$

where the current defaults are:

```text
q = 0.95
K = 3
C_min = 4096
A = 512
L_max = 16384
```

Thus every step receives a tail guard, but long-tail steps naturally recover the
full 16k budget because \(\rho_q S_t \ge L_{\max}\). The cap is written into
`length_sorted_rank_plan.json` as `tail_guard_response_cap`, and runtime only
applies that planned value. This keeps the rule tied to the offline oracle
instead of using a hard-coded runtime threshold.

For the observed epoch2 planner example, only one adjacent transition was
available inside the sliding window, so calibrating from epoch0 -> epoch1 gave:

```text
rho_0.95 = 1.257025
```

The resulting per-step caps were:

| step | selected floor | predicted step exit | tail guard cap |
| ---: | ---: | ---: | ---: |
| 1 | 4 | 3767.7 | 5120 |
| 2 | 4 | 5767.5 | 7680 |
| 3 | 8 | 8131.9 | 10240 |
| 4 | 8 | 11569.8 | 14848 |
| 5 | 16 | 16384.0 | 16384 |

This makes the guard scale-invariant: short steps are protected from rare
outlier generations, while naturally long steps remain uncapped.

### Resume And Checkpointing

The dynamic driver isolates epoch processes. By default, the existing experiment
driver now enables checkpoint chaining:

```text
DYNAMIC_ENABLE_CKPT_CHAIN=1
```

With this setting, epoch0 saves a checkpoint after its normal mode0 training
epoch. Epoch1 resumes from that checkpoint, rebuilds the batch/rank/floor plan
from epoch0 rollout lengths, and then saves its own checkpoint for epoch2.

Set `DYNAMIC_ENABLE_CKPT_CHAIN=0` only when validating rollout scheduling in
isolation and intentionally starting each child run from the same initial model.

## Paper-Style Method Summary

This section summarizes the current method in a form closer to a systems-paper
methodology section. The goal is to make the algorithmic assumptions explicit
and separate policy choices from implementation details.

### Problem Statement

We consider RL rollout generation for a MoE LLM on \(R=16\) devices. Each
rollout step contains \(B=32\) prompts and each prompt produces
`rollout.n=16` sampled responses. The system can shrink the active execution
group during generation, for example `16 -> 8 -> 4`, so ranks that have
finished their assigned work can release memory and communication resources.

The scheduling problem has three coupled constraints:

1. response lengths are stochastic and change across training epochs;
2. the selected floor determines both potential rank-time savings and the
   available KV capacity;
3. planner mode can keep planned floor8/floor4 communication groups resident
   across steps, which improves reuse but consumes non-trivial device memory.

The objective is therefore not simply to minimize within-rank length skew. The
objective is to maximize useful rank-time release while preserving KV
feasibility and avoiding rare tail responses that dominate a synchronous rollout
step.

### Historical Length Predictor

For prompt \(i\) in epoch \(e\), let
\(L_{i,e,j}\) be the length of response sample \(j\). The scheduler uses a
prompt-level tail statistic:

$$
M_{i,e}=\max_j L_{i,e,j}.
$$

Using the maximum rather than the mean matches the latency-sensitive quantity:
a rank cannot be released until the longest live response on that rank
finishes. Across epochs, the predictor uses an exponential moving average:

$$
\hat{M}_{i,e+1}
=
\alpha \hat{M}_{i,e} + (1-\alpha)M_{i,e},
\qquad \alpha=0.7.
$$

Epoch0 is a normal `mode=0` no-shrink training epoch. It is not discarded; it
both updates the model and provides the first measured length distribution for
epoch1 planning. Later epochs rebuild the plan from the most recent rollout
history before launching generation.

### Batch Construction

Given predicted prompt tails \(\hat{M}_{i,e}\), prompts are sorted by predicted
tail length and split into contiguous 32-prompt batches:

$$
B_t =
\text{sort}_{\hat{M}}(P)[32(t-1):32t].
$$

This preserves an epoch-level length-sorted shape: early steps contain shorter
prompts and late steps contain longer prompts. The method deliberately avoids a
global reshuffle that would hide tail-heavy prompts inside many otherwise short
steps.

### KV Feasibility Model

For a rank pair \(S=\{i,j\}\), the planner estimates active KV pressure by:

$$
A(S)=
\max_{\tau\ge0}
\tau \sum_{p\in S}\sum_j \mathbf{1}[L_{p,j}\ge\tau].
$$

To account for prediction drift, every response length is inflated by a safety
factor and clipped at the physical response limit:

$$
A_\mu(S)=
A(\{\min(\mu L,L_{\max}) : L\in S\}),
\qquad
\mu=1.16,\quad L_{\max}=16384.
$$

A candidate rank pair is KV-feasible at floor \(f\) only if

$$
A_\mu(S)\le C_f,
$$

where \(C_f\) is the effective KV cap after subtracting any memory reserved for
resident communication groups.

### Release-Area Rank Assignment

For each fixed 32-prompt batch and candidate floor, the planner constructs all
\(\binom{32}{2}=496\) possible rank-pair edges. Each edge stores:

- pair completion proxy \(M_e=\max(\hat{M}_i,\hat{M}_j)\);
- adjusted KV peak \(A_\mu(e)\);
- token-sum and length-gap tie-breakers.

Let \(T=\max_e M_e\) be the predicted batch tail. A shrink schedule is
represented as quota constraints over pair completion times. For floor4, a
schedule with thresholds \(a\le b\) requires:

$$
\#\{e\in\mathcal{M}:M_e\le a\}\ge8,
\qquad
\#\{e\in\mathcal{M}:M_e\le b\}\ge12.
$$

Its released rank-time area is:

$$
\mathcal{A}_{4}(a,b)=8(T-a)+4(T-b).
$$

Analogously, floor8 uses \(\mathcal{A}_8(a)=8(T-a)\), and floor2 uses
\(\mathcal{A}_2(a,b,c)=8(T-a)+4(T-b)+2(T-c)\).

The matching oracle solves an exact 0/1 assignment problem:

$$
\max_{\mathcal{M}} \mathcal{A}_f
$$

subject to:

- every prompt appears in exactly one selected edge;
- every selected edge is KV-feasible under \(C_f\);
- all schedule quota constraints are satisfied.

Candidate schedules are enumerated from larger release area to smaller release
area. The first feasible exact matching is therefore release-area optimal for
the fixed batch and floor. Within one schedule, the solver breaks ties by
smaller rank-internal length gap, lower pair max length, lower adjusted KV
peak, and lower token sum. This hierarchy is important: length gap is useful,
but only after the shrink schedule has already maximized the system-level
rank-time release.

### Floor Selection

The planner tries floors from the minimum allowed floor upward. In the current
adaptive floor4 configuration:

```text
candidate floors = 4, 8, 16
```

The selected floor is the deepest floor that has both:

1. a KV-feasible exact matching; and
2. positive useful release area.

The second condition prevents a misleading case where a floor8 matching is
technically KV-feasible but can only shrink at \(T\). Such a schedule releases
zero rank-time and is treated as no useful shrink.

Runtime transitions are restricted to halving steps. Thus a floor4 step uses
`16 -> 8 -> 4`, while a floor8 step uses `16 -> 8`. Direct non-halving
transitions are avoided because they can fall back to slow object-broadcast
parameter movement instead of the optimized direct-NPU path.

### Natural And Planner Policies

The offline plan specifies prompts, rank assignments, floors, and planned
survivor topology. Runtime can consume this plan in two ways.

Natural policy lets the actually unfinished ranks determine the survivor set at
each shrink point. This is the most faithful system behavior under stochastic
generation drift. It avoids dummy work when a planned survivor happens to
finish early, and it does not keep planned floor subgroups resident across
steps.

Planner policy fixes the survivor topology to the planned groups. This gives a
deterministic communication topology and enables cross-step reuse of planned
floor8/floor4 communication groups. The cost is extra resident HCCL/MC2/TBE
workspace memory and possible dummy run if runtime completion order differs
from the planned order.

Planner residency is treated as an explicit resource contract. The effective
caps used by the offline planner are:

| floor | raw cap | planned-residency headroom | effective cap |
| ---: | ---: | ---: | ---: |
| 4 | 280576 | 147456 | 133120 |
| 8 | 377344 | 114688 | 262656 |
| 16 | 380800 | 0 | 380800 |

This makes the planner solve the same memory problem that the runtime will
face. Natural mode can use larger caps because it does not retain planned
floor-group residency, although real post-shrink runtime workspaces still lower
some observed capacities compared with cold profiles.

### Prediction-Error Tail Guard

Length-aware batching reduces average imbalance, but a single underpredicted
pathological response can still dominate a step. The tail guard is a
statistical cap on per-step generation length. It is not a hard-coded smoke
test threshold; it is derived from recent prediction error.

For an adjacent historical transition, define the one-sided prompt-tail
underestimation ratio:

$$
r_i^+ =
\max\left(1,\frac{M_{i,e}}{\hat{M}_{i,e}+\epsilon}\right).
$$

The uncertainty multiplier is a high empirical quantile over the most recent
\(K\) transitions:

$$
\rho_q = Q_q(\{r_i^+\}),\qquad q=0.95,\quad K=3.
$$

The ratio is one-sided because overestimation is conservative for latency,
whereas underestimation causes short steps to accidentally become long steps.
The method uses a sliding window rather than another EMA because the length
predictor already uses EMA; the guard is intended to calibrate residual
uncertainty, not to smooth the same signal twice.

For step \(t\), let

$$
S_t=\max_{i\in B_t}\hat{M}_{i,e}
$$

be the predicted step tail. The cap is:

$$
C_t =
\min\left(
  L_{\max},
  \max\left(C_{\min},
    \left\lceil \frac{\rho_q S_t}{A} \right\rceil A
  \right)
\right),
$$

with current defaults:

```text
q = 0.95
K = 3
C_min = 4096
A = 512
L_max = 16384
```

Short predicted steps therefore receive a bounded generation length, while
long-tail steps naturally recover the full 16k response budget. The cap is
stored in the offline rank plan as `tail_guard_response_cap` and is injected
into runtime as `response_max_tokens_cap` after the shrink-aware plan has been
applied. This ordering is required because the cap is plan-dependent.

### Algorithm Sketch

For each epoch \(e\):

1. collect rollout lengths from epoch \(e-1\);
2. update EMA prompt-tail predictions \(\hat{M}_{i,e}\);
3. compute the residual underestimation multiplier \(\rho_q\);
4. sort prompts by \(\hat{M}_{i,e}\) and form 32-prompt batches;
5. for each batch, solve release-area rank assignment under floor-specific KV
   caps;
6. write per-step floor, survivor topology, rank assignment, and
   `tail_guard_response_cap`;
7. run the epoch in natural or planner mode;
8. save rollout lengths and a checkpoint for the next epoch.

This closed loop makes the scheduler adaptive to model-policy drift while
keeping the optimization offline and deterministic for each launched epoch.

### Evaluation Protocol

A comparison suitable for a systems venue should report more than total wall
time. At minimum, each run should include:

- per-step generated token count and `rollout_output_time_s`;
- throughput in generated tokens per second;
- selected floor and actual shrink stages;
- tail guard cap and clipped-response ratio;
- KV cache capacity used by the step;
- resize / precreate / stale-group-release overhead;
- for planner mode, resident communication-group state and dummy-run time;
- end-to-end epoch time including rollout, reward/logprob, update, and
  checkpoint overhead.

Because generation is stochastic, natural/planner comparisons should be made
over repeated runs or normalized by generated-token count and observed response
length distribution. Single-run improvements should be interpreted together
with the realized length histogram, especially when rare 16k responses appear.
