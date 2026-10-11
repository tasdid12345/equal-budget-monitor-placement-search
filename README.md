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
