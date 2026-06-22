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
