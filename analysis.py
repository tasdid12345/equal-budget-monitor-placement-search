"""Monitor placement (max closed-neighbourhood coverage): tuned greedy-seeded tabu swap search vs default SA."""
import gzip, heapq, json, math, os, sys, time
import numpy as np, networkx as nx, requests
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import hstack, vstack, identity, csr_matrix
from sklearn.model_selection import GroupKFold

SEED, BUDGET, TUNE_BUDGET, MILP_LIMIT, N_CFG = 42, 2.0, 0.3, 20, 12
URL = "https://snap.stanford.edu/data/email-Eu-core.txt.gz"
KS, SIZES, N_GRAPHS, SEEDS = [3, 6, 10], [300, 500, 700], 12, [42, 43, 44]
pc = time.perf_counter


def load_graph():
    """Download SNAP graph (retry/backoff); generated fallback only if download fails."""
    for a in range(4):
        try:
            r = requests.get(URL, timeout=60)
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(2 ** a); continue
            r.raise_for_status()
            lines = gzip.decompress(r.content).decode().splitlines()
            G = nx.Graph()
            for l in lines:
                u, v = map(int, l.split()[:2])
                if u != v: G.add_edge(u, v)
            return G, "SNAP email-Eu-core"
        except Exception as e:
            print("download failed:", e); time.sleep(2 ** a)
    return nx.powerlaw_cluster_graph(1000, 15, 0.3, seed=SEED), "networkx powerlaw_cluster fallback (download failed)"


def bfs_sample(G, size, rng):
    """BFS-sampled connected subgraph (random neighbour order) as closed-neighbourhood matrix."""
    start = list(G)[rng.integers(len(G))]
    seen, order = {start}, [start]
    i = 0
    while i < len(order) and len(order) < size:
        nbrs = list(G[order[i]]); rng.shuffle(nbrs); i += 1
        for v in nbrs:
            if v not in seen and len(order) < size:
                seen.add(v); order.append(v)
    idx = {v: j for j, v in enumerate(order)}
    A = np.eye(len(order), dtype=np.float32)
    for u, v in G.subgraph(order).edges: A[idx[u], idx[v]] = A[idx[v], idx[u]] = 1
    return A


def cover(A, S): return int(np.count_nonzero(A[S].sum(0) > 0))


def lazy_greedy(A, k):
    """Lazy greedy (valid because coverage is submodular)."""
    nb = [set(np.flatnonzero(r)) for r in A]
    heap = [(-len(nb[v]), v, 0) for v in range(len(A))]; heapq.heapify(heap)
    S, covered = [], set()
    while len(S) < k:
        g, v, r = heapq.heappop(heap)
        if r == len(S): S.append(v); covered |= nb[v]
        else: heapq.heappush(heap, (-len(nb[v] - covered), v, len(S)))
    return S


def sim_anneal(A, k, budget, rng, T0=1.0, alpha=0.999):
    """Baseline: default-parameter SA with random swap moves from a random start."""
    n = len(A); dl = pc() + budget
    S = rng.choice(n, k, replace=False); inS = np.zeros(n, bool); inS[S] = True
    c = A[S].sum(0); cur = best = int(np.count_nonzero(c > 0)); T = T0
    while pc() < dl:
        i = rng.integers(k); u = S[i]; w = rng.integers(n)
        if inS[w]: continue
        c2 = c - A[u] + A[w]; new = int(np.count_nonzero(c2 > 0)); d = new - cur
        if d >= 0 or rng.random() < math.exp(d / T):
            S[i] = w; inS[u] = False; inS[w] = True; c, cur = c2, new; best = max(best, cur)
        T = max(T * alpha, 1e-3)
    return best


def tabu_swap(A, k, budget, rng, p):
    """Improved: lazy-greedy start, then best-swap tabu search with perturbation restarts."""
    dl = pc() + budget; n = len(A)
    S = np.array(lazy_greedy(A, k)); inS = np.zeros(n, bool); inS[S] = True
    c = A[S].sum(0); cur = best = int(np.count_nonzero(c > 0)); bestS = S.copy()
    tabu = np.zeros(n); it = last = 0
    while pc() < dl:
        it += 1
        ok = [i for i in range(k) if tabu[S[i]] <= it] or list(range(k))
        bm = (-1, 0, 0)
        for i in rng.choice(ok, min(p["n_out"], len(ok)), replace=False):
            unc = ((c - A[S[i]]) == 0).astype(np.float32)
            tot = (n - unc.sum()) + A @ unc  # coverage after swapping S[i] -> each node
            tot[inS | ((tabu > it) & (tot <= best))] = -1  # members / tabu (with aspiration)
            w = int(np.argmax(tot + rng.random(n) * 0.5))
            if tot[w] > bm[0]: bm = (tot[w], i, w)
        if bm[0] >= 0:
            val, i, w = bm; u = S[i]
            c = c - A[u] + A[w]; inS[u] = False; inS[w] = True; S[i] = w
            tabu[u] = tabu[w] = it + p["tenure"]; cur = int(val)
            if cur > best: best, bestS, last = cur, S.copy(), it
        if bm[0] < 0 or (it - last >= p["patience"] and cur <= best):
            S = bestS.copy(); r = max(1, int(p["perturb"] * k))
            pool = np.flatnonzero(~np.isin(np.arange(n), S))
            S[rng.choice(k, r, replace=False)] = rng.choice(pool, r, replace=False)
            inS[:] = False; inS[S] = True; c = A[S].sum(0)
            cur = int(np.count_nonzero(c > 0)); tabu[:] = 0; last = it
    return best


def milp_ref(A, k):
    """HiGHS MILP reference (time-limited); returns coverage of its incumbent."""
    n = len(A)
    M = vstack([hstack([csr_matrix(-A), identity(n)]),
                csr_matrix(np.r_[np.ones(n), np.zeros(n)][None, :])])
    res = milp(np.r_[np.zeros(n), -np.ones(n)],
               constraints=LinearConstraint(M, -np.inf, np.r_[np.zeros(n), k]),
               integrality=np.r_[np.ones(n), np.zeros(n)], bounds=Bounds(0, 1),
               options={"time_limit": MILP_LIMIT})
    if res.x is None: return float("nan")
    return float(cover(A, np.argsort(-res.x[:n])[:k]))


def run(method, inst, seed, budget, p=None):
    rng = np.random.default_rng(seed)
    if method == "sa": return sim_anneal(inst["A"], inst["k"], budget, rng)
    return tabu_swap(inst["A"], inst["k"], budget, rng, p)


def main():
    rng = np.random.default_rng(SEED)
    G, name = load_graph()
    G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    print(f"graph {name}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    mats = [bfs_sample(G, min(SIZES[g % 3], len(G)), rng) for g in range(N_GRAPHS)]
    test_g = {int(rng.choice([g for g in range(N_GRAPHS) if g % 3 == s])) for s in range(3)}
    insts = [dict(A=mats[g], k=k, g=g, n=len(mats[g])) for g in range(N_GRAPHS) for k in KS]
    train = [x for x in insts if x["g"] not in test_g]; test = [x for x in insts if x["g"] in test_g]
    print(f"{len(train)} tuning instances, {len(test)} held-out test instances")

    # random search for tabu/restart settings, scored by grouped CV on tuning instances only
    cfgs = [dict(tenure=int(rng.choice([0, 3, 5, 10, 20])), patience=int(rng.choice([20, 50, 100, 300])),
                 perturb=float(rng.choice([0.1, 0.2, 0.4])), n_out=int(rng.choice([1, 3, 10])))
            for _ in range(N_CFG)]
    groups = [x["g"] for x in train]; cv = []
    for j, p in enumerate(cfgs):
        fold_scores = []
        for _, val in GroupKFold(n_splits=3).split(train, groups=groups):
            fold_scores.append(np.mean([run("ts", train[i], SEED, TUNE_BUDGET, p) / train[i]["n"] for i in val]))
        cv.append(float(np.mean(fold_scores))); print(f"cfg {j} {p} cv coverage={cv[-1]:.4f}")
    best_p = cfgs[int(np.argmax(cv))]; print("selected:", best_p)

    # held-out evaluation (test set used once)
    refs = []
    for x in test:
        refs.append(milp_ref(x["A"], x["k"])); print(f"MILP n={x['n']} k={x['k']} ref={refs[-1]}")
    refs = np.array(refs); valid = np.isfinite(refs) & (refs > 0)
    if not valid.any(): sys.exit("no valid MILP references")
    cov = {m: np.array([[run(m, x, s, BUDGET, best_p) for s in SEEDS] for x in test], float)
           for m in ("sa", "ts")}  # shape (instances, seeds)
    for m in cov: print(m, "mean coverage", cov[m][valid].mean())
    rel = {m: cov[m][valid] / refs[valid][:, None] / 1.0 for m in cov}
    base, imp = rel["sa"].mean(), rel["ts"].mean()
    if not (np.isfinite(base) and np.isfinite(imp)): sys.exit("non-finite scores")
    metrics = {"baseline_score": float(base), "improved_score": float(imp), "improvement": float(imp - base),
               "baseline_mean_coverage": float(cov["sa"][valid].mean()), "improved_mean_coverage": float(cov["ts"][valid].mean()),
               "baseline_seed_std": float(rel["sa"].mean(0).std()), "improved_seed_std": float(rel["ts"].mean(0).std()),
               "improved_win_rate": float((rel["ts"].mean(1) > rel["sa"].mean(1)).mean()),
               "tuning_cv_best_coverage_fraction": float(max(cv))}
    print(json.dumps(metrics, indent=2))
    res = {"question": "Under the same short time budget, can a greedy-seeded, CV-tuned swap local search place k monitors with higher coverage than default simulated annealing?",
           "method": "BFS-sampled subgraphs x several k; closed-neighbourhood coverage; random search on tuning instances with grouped CV; same wall-clock budget per instance; relative to HiGHS MILP incumbent.",
           "dataset": {"name": name, "source_url": URL, "n_rows": G.number_of_edges(), "n_train": len(train),
                       "n_test": int(valid.sum()), "same_as_paper": False},
           "metric_name": "covered-node fraction relative to MILP reference value", "higher_is_better": True,
           "baseline": "Simulated-annealing swap metaheuristic with default parameters (T0=1, alpha=0.999) from a random start.",
           "improved": f"Lazy-greedy initialisation + best-swap tabu local search with restarts; params from random search ({N_CFG} configs, 3-fold grouped CV on tuning instances): {best_p}.",
           "metrics": metrics,
           "notes": (f"n_rows is the edge count; n_train/n_test count instances (graph, k), split by sampled graph. Budget {BUDGET}s per instance/method/seed; "
                     f"tuning used a shorter {TUNE_BUDGET}s budget. MILP time limit {MILP_LIMIT}s, so reference may be sub-optimal and scores can exceed 1. "
                     "Problem formulation (closed-neighbourhood coverage) is an assumption since the paper's exact model is unavailable. "
                     "Wall-clock budgets make results machine-dependent. Small test set (3 graphs x 3 k); seed std is over per-seed means across test instances.")}
    with open("results.json", "w", encoding="utf-8") as f: json.dump(res, f, indent=2)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    os.makedirs("figures", exist_ok=True)
    plt.bar(["SA default", "Tuned greedy+tabu"], [base, imp], yerr=[metrics["baseline_seed_std"], metrics["improved_seed_std"]])
    plt.ylim(min(base, imp) - 0.05, max(base, imp) + 0.03); plt.ylabel("coverage / MILP reference")
    plt.title("Held-out test instances"); plt.savefig("figures/comparison.png", dpi=120, bbox_inches="tight")


if __name__ == "__main__":
    main()
