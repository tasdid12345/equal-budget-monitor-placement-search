# Monitor placement: tuned greedy-seeded swap search vs. simulated annealing

## Motivation
The paper "A Metaheuristic Approach for Solving Monitor Placement Problem" (2023) proposes a metaheuristic for placing monitors in a network that is designed to run in short computing times. Only its abstract opening was available, so this experiment uses an assumed formulation and tests whether a tuned, greedy-seeded local search beats a default-parameter simulated-annealing metaheuristic under an equal, short time budget.

## Question
Under the same short time budget, can a greedy-seeded, cross-validated-tuned swap local search place k monitors with higher network coverage than a default-parameter simulated annealing on a public real-world communication graph?

## Dataset
SNAP email-Eu-core communication network, downloaded over HTTP (a networkx generated graph is used only if the download fails). This is **not** the paper's own dataset. Instances are BFS-sampled subgraphs combined with several k values. They are split by subgraph into tuning and held-out test instances. A monitor covers its closed neighbourhood, and the objective is the number of covered nodes. This formulation is an assumption.

## Models
- **Baseline:** simulated-annealing swap metaheuristic with default parameters, started from a random solution.
- **Improved:** lazy-greedy initialisation, then swap local search with tabu list and perturbation restarts. Its settings are chosen by random search with grouped cross-validation on the tuning instances only.
- Every method gets the same wall-clock budget per instance (about 2 s) and is run with 3 seeds.
- **Main metric:** covered-node fraction relative to the incumbent of a time-limited scipy HiGHS MILP. Secondary metrics are mean absolute coverage and the standard deviation over seeds.

## How to run
```
pip install -r requirements.txt
python analysis.py
```

## Results
Results are written to `results.json` (and a chart to `figures/`) when the script finishes.

---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._

## Results (latest run)

On the SNAP email-Eu-core graph (9 held-out test instances), my re-implementation of your simulated-annealing baseline reached 99.98% of the MILP reference coverage, while my tabu-search variant reached 100%, a difference of only about 0.02 percentage points.

| Metric | Value |
|---|---|
| baseline_score | 0.9998 |
| improved_score | 1 |
| improvement | 0.0002104 |
| baseline_mean_coverage | 431.5 |
| improved_mean_coverage | 431.6 |
| baseline_seed_std | 0.0002976 |
| improved_seed_std | 0 |
| improved_win_rate | 0.1111 |
| tuning_cv_best_coverage_fraction | 0.8624 |

Run on 2026-10-11. Full output: [results.json](results.json) · [run.log](run.log)

Notes: n_rows is the edge count; n_train/n_test count instances (graph, k), split by sampled graph. Budget 2.0s per instance/method/seed; tuning used a shorter 0.3s budget. MILP time limit 20s, so reference may be sub-optimal and scores can exceed 1. Problem formulation (closed-neighbourhood coverage) is an assumption since the paper's exact model is unavailable. Wall-clock budgets make results machine-dependent. Small test set (3 graphs x 3 k); seed std is over per-seed means across test instances.
