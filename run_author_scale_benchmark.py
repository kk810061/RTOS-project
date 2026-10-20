#!/usr/bin/env python3
"""
Author-Scale Benchmark Runner (500 DAGs x 10 Runs = 5,000 Instances)
Directly replicates the experimental scale of Toba, Baruah, & Azumi (ACM TECS 2026 Section 5.3):
"500 randomly generated DAGs were executed 10 times" = 5,000 instances.

Evaluates the Full 7-Configuration Matrix:
1. Toba Baseline (ACM TECS 2026)
2. Idea 2 Alone (NP-EDF Queue- & Contention-Aware Plaxity)
3. Idea 3 Alone (Bayesian Context-Coupled Plaxity)
4. Ideas 3 + 2 (Context + Queue, Detection Only)
5. Ideas 2 + 1 (Queue + Mitigate, NO Idea 3)
6. Ideas 3 + 1 (Context + Mitigate, NO Idea 2)
7. Triad Framework (Ideas 3 + 2 + 1: Context + Queue + Mitigate)
"""

import os
import sys
import glob
import math
import heapq
import time
import numpy as np
import pandas as pd
import networkx as nx
import yaml
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------
# 1. CORE TOBA ET AL. & DAG UTILITIES
# ----------------------------------------------------------------------------
def create_paper_pmf(wcet_bins: int) -> np.ndarray:
    wcet_bins = max(6, int(wcet_bins))
    x = np.arange(wcet_bins + 1, dtype=np.float64)
    mu = wcet_bins / 3.0
    sigma = mu / 2.0
    pmf = 0.98 * stats.norm.pdf(x, loc=mu, scale=max(sigma, 1e-3)) \
        + 0.02 * stats.norm.pdf(x, loc=wcet_bins, scale=max(0.2 * sigma, 1e-3))
    pmf[0] = 0.0
    s = pmf.sum()
    return pmf / s if s > 0 else pmf

def compute_plaxity(G, topo, pmfs, D):
    idx = {v: i for i, v in enumerate(topo)}
    n = len(topo)
    Fm = np.zeros((n, D + 1))
    lst_pmf = {}
    f_miss = {}
    mu = {v: float(np.sum(np.arange(len(p)) * p)) for v, p in pmfs.items()}
    rem_t, rem_j = {}, {}

    for v in reversed(topo):
        succs = list(G.successors(v))
        c = pmfs[v]
        if not succs:
            lft = np.zeros(D + 1)
            lft[D] = 1.0
            rem_t[v], rem_j[v] = mu[v], 1.0
        else:
            if len(succs) == 1:
                lft = lst_pmf[succs[0]].copy()
            else:
                log_surv = np.zeros(D + 1)
                for u in succs:
                    log_surv += np.log1p(-np.minimum(f_miss[u], 1.0 - 1e-16))
                f_lft = -np.expm1(log_surv)
                lft = np.zeros(D + 1)
                lft[0] = f_lft[0]
                lft[1:] = np.clip(np.diff(f_lft), 0.0, 1.0)
            rem_t[v] = mu[v] + min(rem_t[u] for u in succs)
            rem_j[v] = 1.0 + min(rem_j[u] for u in succs)

        corr = np.correlate(lft, c, mode="full")
        z = len(c) - 1
        lst = np.zeros(D + 1)
        lst[0] = corr[: z + 1].sum()
        end = min(len(corr), z + D + 1)
        lst[1: end - z] = corr[z + 1: end]
        s = lst.sum()
        if s > 0:
            lst /= s
        lst_pmf[v] = lst
        f_miss[v] = np.clip(np.cumsum(lst), 0.0, 1.0)
        Fm[idx[v]] = f_miss[v]
    return Fm, mu, rem_t, rem_j

def eq15_18(G, topo, w, rem, th_min=0.8, th_max=1.0):
    th = {}
    for v in topo:
        preds = list(G.predecessors(v))
        if not preds:
            th[v] = th_min
        elif G.out_degree(v) == 0:
            th[v] = th_max
        else:
            th[v] = max(th[u] + (th_max - th[u]) / rem[v] for u in preds)
    return {v: float(th[v]) for v in topo}

def build_dag_context(filepath, d_factor=1.2):
    with open(filepath, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    G = nx.DiGraph()
    for nd in data["nodes"]:
        G.add_node(int(nd["id"]), wcet=max(12, int(nd.get("execution_time", 25)) * 3))
    for e in data.get("links", data.get("edges", [])):
        G.add_edge(int(e["source"]), int(e["target"]))
    
    topo = list(nx.topological_sort(G))
    sink = [v for v in topo if G.out_degree(v) == 0][0]
    pmfs = {v: create_paper_pmf(G.nodes[v]["wcet"]) for v in topo}
    
    cp = {}
    for v in topo:
        cp[v] = max((cp[u] for u in G.predecessors(v)), default=0) + G.nodes[v]["wcet"]
    D = int(round(cp[sink] * d_factor))
    
    Fm, mu, rem_t, rem_j = compute_plaxity(G, topo, pmfs, D)
    th_time = eq15_18(G, topo, mu, rem_t, th_min=0.8, th_max=1.0)
    th_job = eq15_18(G, topo, {v: 1.0 for v in topo}, rem_j, th_min=0.8, th_max=1.0)

    levels = {}
    for v in topo:
        preds = list(G.predecessors(v))
        levels[v] = 0 if not preds else max(levels[u] for u in preds) + 1
    max_lvl = max(levels.values()) if levels else 1

    rho = {}
    alpha = {}
    for v in topo:
        norm_lvl = levels[v] / max_lvl
        if norm_lvl < 0.35:
            rho[v] = float(np.clip(0.85 - 0.2 * norm_lvl, 0.65, 0.95))
            alpha[v] = 0.85
        elif norm_lvl < 0.70:
            rho[v] = float(np.clip(0.55 - 0.3 * (norm_lvl - 0.35), 0.30, 0.65))
            alpha[v] = 0.75
        else:
            rho[v] = float(np.clip(0.15 - 0.1 * (norm_lvl - 0.70), 0.05, 0.20))
            alpha[v] = 0.50

    rem_sens = {}
    for v in reversed(topo):
        succs = list(G.successors(v))
        if not succs:
            rem_sens[v] = rho[v] * mu[v]
        else:
            rem_sens[v] = rho[v] * mu[v] + min(rem_sens[u] for u in succs)

    return {
        "G": G, "topo": topo, "sink": sink, "D": D, "n": len(topo),
        "Fm": Fm, "mu": mu, "sigma": {v: max(1.0, mu[v] / 2.0) for v in topo},
        "wcet": {v: G.nodes[v]["wcet"] for v in topo},
        "cdf": {v: np.cumsum(pmfs[v]) for v in topo},
        "th_time": th_time, "th_job": th_job, "levels": levels,
        "rem_t": rem_t, "cp_wcet": cp[sink],
        "rho": rho, "alpha": alpha, "rem_sens": rem_sens
    }

def make_background(rng, D, intensity=1.0):
    lf = rng.gamma(3.2, 0.35)
    n_arr = int(rng.poisson(50 * lf * intensity))
    hi = max(2, int(D * 0.60))
    tasks = []
    for _ in range(n_arr):
        rel = int(rng.integers(1, hi))
        w_b = int(rng.integers(40, 135))
        dur = max(2, int(round(rng.normal(w_b / 3.0, w_b / 6.0)))) if rng.random() < 0.98 else w_b
        bg_d = rel + dur + int(rng.integers(5, 65))
        tasks.append((rel, bg_d, dur))
    return tasks

def calculate_amdahl_duration(nom_dur, alpha_v, dvfs_ratio=1.33):
    scaled = nom_dur * (alpha_v / dvfs_ratio + (1.0 - alpha_v))
    return max(1, int(round(scaled)))

# ----------------------------------------------------------------------------
# 2. EXACT NP-EDF DISPATCH DELAY ESTIMATOR (IDEA 2)
# ----------------------------------------------------------------------------
def compute_exact_node_dispatch_delay(t, ready, running_cores, target_vid, target_prio, target_seq, num_cores):
    core_heap = []
    for c in range(num_cores):
        if c in running_cores:
            finish_t, _ = running_cores[c]
            rem_t = max(0, finish_t - t)
            heapq.heappush(core_heap, rem_t)
        else:
            heapq.heappush(core_heap, 0)

    higher_prio_tasks = []
    for it in ready:
        prio_k, s_seq, is_dag, vid, dur = it
        if vid == target_vid and is_dag:
            continue
        if (prio_k < target_prio) or (prio_k == target_prio and s_seq < target_seq):
            higher_prio_tasks.append((prio_k, s_seq, dur))

    higher_prio_tasks.sort(key=lambda x: (x[0], x[1]))

    if core_heap[0] == 0 and len(higher_prio_tasks) == 0:
        return 0

    for _, _, dur in higher_prio_tasks:
        earliest_free = heapq.heappop(core_heap)
        heapq.heappush(core_heap, earliest_free + dur)

    expected_delay = heapq.heappop(core_heap)
    return int(expected_delay)

# ----------------------------------------------------------------------------
# 3. CLOSED-LOOP SIMULATOR
# ----------------------------------------------------------------------------
def simulate_job(ctx, c_job_nominal, D, num_cores, bg_tasks,
                  detector_mode="context_coupled",
                  enable_queue_delay=False,
                  enable_mitigation=False,
                  dvfs_ratio=1.33):
    G, topo = ctx["G"], ctx["topo"]
    topo_idx = {v: i for i, v in enumerate(topo)}
    th_vec = ctx["th_time"]
    c_job = dict(c_job_nominal)

    in_deg = {v: G.in_degree(v) for v in G.nodes}
    events, ready = [], []
    seq = [0]
    for rel, bg_d, dur in bg_tasks:
        heapq.heappush(events, (rel, 1, (bg_d, dur, seq[0]), -1))
        seq[0] += 1
    for v in G.nodes:
        if in_deg[v] == 0:
            heapq.heappush(ready, (D, seq[0], True, v, c_job[v]))
            seq[0] += 1

    free = list(range(num_cores))
    running_cores = {}
    ast, ft = {}, {}

    sigma_0 = 1.0
    accum_prec = 1.0 / (sigma_0 ** 2)
    accum_num = 0.0

    alarmed = [False]
    t_alarm = [None]
    alarm_node = [-1]
    alarm_w_v = [0]
    mitigation_active = [False]

    def get_current_posterior_mean():
        if accum_prec <= 1.0 / (sigma_0 ** 2):
            return 0.0
        return accum_num / accum_prec

    def trigger_mitigation(t):
        if not mitigation_active[0]:
            mitigation_active[0] = True
            new_ready = []
            for it in ready:
                prio_key, s_seq, is_dag, vid, dur = it
                if is_dag:
                    new_prio = -1
                    new_dur = calculate_amdahl_duration(dur, ctx["alpha"][vid], dvfs_ratio)
                    new_ready.append((new_prio, s_seq, is_dag, vid, new_dur))
                else:
                    new_ready.append(it)
            heapq.heapify(new_ready)
            ready.clear()
            ready.extend(new_ready)

    def check_detection_and_dispatch(t):
        if not alarmed[0]:
            hat_xi = get_current_posterior_mean() if "context_coupled" in detector_mode else 0.0
            
            for item in ready:
                if item[2]:  # is_dag
                    prio_k, s_seq, _, vid, _ = item
                    threshold = th_vec[vid]

                    w_v = compute_exact_node_dispatch_delay(t, ready, running_cores, vid, prio_k, s_seq, num_cores) if enable_queue_delay else 0

                    if detector_mode == "tecs_time":
                        t_eff = min(t, D)
                    elif detector_mode == "queue_only":
                        t_eff = int(np.clip(t + w_v, 0, D))
                    elif detector_mode == "context_coupled":
                        delta_rem = max(0.0, hat_xi * ctx["rem_sens"][vid])
                        t_eff = int(np.clip(round(t + delta_rem + w_v), 0, D))
                    else:
                        t_eff = min(t, D)

                    miss_prob = ctx["Fm"][topo_idx[vid], t_eff]
                    if miss_prob > (1.0 - threshold):
                        alarmed[0] = True
                        t_alarm[0] = t
                        alarm_node[0] = vid
                        alarm_w_v[0] = w_v

                        if enable_mitigation:
                            trigger_mitigation(t)
                        break

        while free and ready:
            prio_k, s_seq, is_dag, vid, dur = heapq.heappop(ready)
            core = free.pop()
            running_cores[core] = (t + dur, vid)
            if is_dag:
                ast[vid] = t

            heapq.heappush(events, (t + dur, 0, (is_dag, vid), core))

    check_detection_and_dispatch(0)
    n = G.number_of_nodes()

    while len(ft) < n and events:
        t, et, pay, cid = heapq.heappop(events)
        batch = [(et, pay, cid)]
        while events and events[0][0] == t:
            _, e2, p2, c2 = heapq.heappop(events)
            batch.append((e2, p2, c2))
        for et, pay, core in batch:
            if et == 0:
                free.append(core)
                if core in running_cores:
                    del running_cores[core]
                is_dag, vid = pay
                if is_dag:
                    ft[vid] = t
                    actual_dur = t - ast[vid]
                    mu_v = ctx["mu"][vid]
                    sigma_v = ctx["sigma"][vid]
                    rho_v = ctx["rho"][vid]
                    
                    w_v = (rho_v * mu_v) / sigma_v
                    y_v = (actual_dur - mu_v) / sigma_v
                    accum_prec += w_v ** 2
                    accum_num += w_v * y_v

                    for s in G.successors(vid):
                        in_deg[s] -= 1
                        if in_deg[s] == 0:
                            seq[0] += 1
                            prio = -1 if mitigation_active[0] else D
                            dur_s = calculate_amdahl_duration(c_job[s], ctx["alpha"][s], dvfs_ratio) if mitigation_active[0] else c_job[s]
                            heapq.heappush(ready, (prio, seq[0], True, s, dur_s))
            else:
                bg_d, dur, bseq = pay
                heapq.heappush(ready, (bg_d, bseq, False, -1, dur))
        check_detection_and_dispatch(t)

    exit_time = ft.get(ctx["sink"], D + 1)
    actual_miss = bool(exit_time > D)

    return {
        "actual_miss": actual_miss,
        "alarmed": alarmed[0],
        "alarm_node": int(alarm_node[0]) if alarmed[0] else None,
        "alarm_w_v": int(alarm_w_v[0]) if alarmed[0] else 0,
        "t_alarm": int(t_alarm[0]) if (alarmed[0] and t_alarm[0] is not None) else None,
        "exit_time": int(exit_time),
        "deadline": int(D),
        "lead_time": max(0, D - t_alarm[0]) if (alarmed[0] and t_alarm[0] is not None) else 0,
        "mitigated": mitigation_active[0]
    }

# ----------------------------------------------------------------------------
# 4. STATISTICAL UTILITIES
# ----------------------------------------------------------------------------
def wilson_ci(k, n, confidence=0.95):
    if n == 0:
        return 0.0, 0.0, 0.0
    z = stats.norm.ppf((1 + confidence) / 2)
    p = k / n
    denom = 1 + (z ** 2) / n
    center = (p + (z ** 2) / (2 * n)) / denom
    margin = (z * math.sqrt((p * (1 - p) + (z ** 2) / (4 * n)) / n)) / denom
    low = max(0.0, center - margin)
    high = min(1.0, center + margin)
    return p * 100.0, low * 100.0, high * 100.0

def mean_ci(arr, confidence=0.95):
    arr = np.asarray(arr, dtype=np.float64)
    n = len(arr)
    if n == 0:
        return 0.0, 0.0, 0.0
    if n == 1:
        return float(arr[0]), float(arr[0]), float(arr[0])
    mean = float(np.mean(arr))
    sem = stats.sem(arr)
    margin = sem * stats.t.ppf((1 + confidence) / 2, df=n - 1)
    return mean, max(0.0, mean - margin), mean + margin

# ----------------------------------------------------------------------------
# 5. MASTER AUTHOR-SCALE BENCHMARK (500 DAGs x 10 Runs = 5,000 Instances)
# ----------------------------------------------------------------------------
def run_benchmark(n_dags=500, n_runs_per_dag=10, seed=42):
    dag_files = sorted(glob.glob("RD-Gen/DAGs/**/dag_*.yaml", recursive=True))
    if len(dag_files) < n_dags:
        print(f"Warning: requested {n_dags} DAGs but found {len(dag_files)}. Using all available.")
        selected_dags = dag_files
    else:
        # Pick exactly n_dags evenly spaced
        stride = max(1, len(dag_files) // n_dags)
        selected_dags = dag_files[::stride][:n_dags]

    actual_n_dags = len(selected_dags)
    total_instances = actual_n_dags * n_runs_per_dag

    print("=" * 80)
    print(f"AUTHOR-SCALE BENCHMARK (Toba et al. ACM TECS 2026 Section 5.3)")
    print(f"Selected DAGs: {actual_n_dags} | Runs per DAG: {n_runs_per_dag} | Total Instances: {total_instances}")
    print("=" * 80)

    # Pre-generate cohort
    t0 = time.time()
    rng = np.random.default_rng(seed)
    cohort = []
    
    print("[1/3] Building DAG contexts and synthetic execution instances...")
    for d_idx, yf in enumerate(selected_dags):
        ctx = build_dag_context(yf)
        topo, D = ctx["topo"], ctx["D"]

        for r in range(n_runs_per_dag):
            is_clutter = (rng.random() < 0.40)
            xi = float(rng.normal(1.3, 0.3)) if is_clutter else float(rng.normal(0.0, 0.25))

            c_job_nom = {}
            for v in topo:
                cdf = ctx["cdf"][v]
                base = min(int(np.searchsorted(cdf, rng.random(), side="right")), len(cdf) - 1)
                mult = 1.0 + ctx["rho"][v] * xi
                c_job_nom[v] = max(1, int(round(base * mult)))

            bg_tasks = make_background(rng, D, intensity=1.0)

            # Counterfactual: would miss without mitigation?
            unmit = simulate_job(ctx, c_job_nom, D, num_cores=8, bg_tasks=bg_tasks,
                                 detector_mode="context_coupled",
                                 enable_queue_delay=False,
                                 enable_mitigation=False)
            would_miss = unmit["actual_miss"]

            cohort.append({
                "ctx": ctx,
                "c_job_nom": c_job_nom,
                "D": D,
                "bg_tasks": bg_tasks,
                "would_miss": would_miss,
                "xi": xi
            })

    t_cohort = time.time()
    print(f"Cohort generation complete: {total_instances} instances in {t_cohort - t0:.2f}s")
    impending_miss_total = sum(1 for item in cohort if item["would_miss"])
    print(f"Impending misses identified: {impending_miss_total} / {total_instances} ({impending_miss_total/total_instances*100:.1f}%)")

    configs = [
        ("1. Toba Baseline (ACM TECS 2026)",           "tecs_time",       False, False),
        ("2. Idea 2 Alone (Queue Only)",              "queue_only",      True,  False),
        ("3. Idea 3 Alone (Context Only)",            "context_coupled", False, False),
        ("4. Ideas 3 + 2 (Context + Queue)",          "context_coupled", True,  False),
        ("5. Ideas 2 + 1 (Queue + Mitigate, NO Id3)", "queue_only",      True,  True),
        ("6. Ideas 3 + 1 (Context + Mitigate, NO Id2)","context_coupled",False, True),
        ("7. Triad Framework (Ideas 3 + 2 + 1)",      "context_coupled", True,  True),
    ]

    print("\n[2/3] Simulating all 7 configurations across the 5,000-instance cohort...")
    results = []

    for name, det_mode, eqd, mit in configs:
        t_start_cfg = time.time()
        tp, fp, tn, fn = 0, 0, 0, 0
        salvaged = 0
        leads = []

        for item in cohort:
            sim = simulate_job(item["ctx"], item["c_job_nom"], item["D"], num_cores=8,
                               bg_tasks=item["bg_tasks"],
                               detector_mode=det_mode,
                               enable_queue_delay=eqd,
                               enable_mitigation=mit)

            would_miss = item["would_miss"]
            actual_miss = sim["actual_miss"]
            alarmed = sim["alarmed"]

            if would_miss and mit and not actual_miss:
                salvaged += 1

            if would_miss and alarmed:
                tp += 1
                leads.append(sim["lead_time"])
            elif not would_miss and alarmed:
                fp += 1
            elif not would_miss and not alarmed:
                tn += 1
            else:
                fn += 1

        rec, rec_l, rec_h = wilson_ci(tp, tp + fn)
        prec, prec_l, prec_h = wilson_ci(tp, tp + fp)
        fpr, fpr_l, fpr_h = wilson_ci(fp, tn + fp)
        dsr, dsr_l, dsr_h = wilson_ci(salvaged, impending_miss_total)
        lead_m, lead_l, lead_h = mean_ci(leads)

        elapsed = time.time() - t_start_cfg
        print(f"  -> {name:<42} | DSR: {dsr:5.1f}% [{dsr_l:4.1f}, {dsr_h:4.1f}] | Lead: {lead_m:5.1f} ms | Rec: {rec:5.1f}% | Prec: {prec:5.1f}% ({elapsed:.2f}s)")

        results.append({
            "Configuration": name,
            "Context (Id3)": "Yes" if "context_coupled" in det_mode else "No",
            "Queue (Id2)": "Yes" if eqd else "No",
            "Mitigate (Id1)": "Yes" if mit else "No",
            "Total Jobs": total_instances,
            "Impending Misses": impending_miss_total,
            "TP": tp, "FP": fp, "TN": tn, "FN": fn,
            "Recall %": f"{rec:.1f}% [{rec_l:.1f}, {rec_h:.1f}]",
            "Precision %": f"{prec:.1f}% [{prec_l:.1f}, {prec_h:.1f}]",
            "FPR %": f"{fpr:.1f}% [{fpr_l:.1f}, {fpr_h:.1f}]",
            "Lead Time (ms)": f"{lead_m:.1f} [{lead_l:.1f}, {lead_h:.1f}]",
            "DSR %": f"{dsr:.1f}% [{dsr_l:.1f}, {dsr_h:.1f}]",
            "Salvaged Ratio": f"{salvaged} / {impending_miss_total}",
            "dsr_val": dsr,
            "lead_val": lead_m,
            "prec_val": prec,
            "rec_val": rec
        })

    # Save to CSV
    df = pd.DataFrame(results)
    out_csv = "author_scale_5000_benchmark_results.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n[3/3] Results saved successfully to {out_csv}")

    # Generate Publication Figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    names = [r["Configuration"].split(". ")[1].replace(" (ACM TECS 2026)", "") for r in results]
    dsrs = [r["dsr_val"] for r in results]
    leads = [r["lead_val"] for r in results]

    colors = ["#64748b", "#3b82f6", "#8b5cf6", "#06b6d4", "#f59e0b", "#ec4899", "#10b981"]

    # Bar 1: DSR
    bars1 = axes[0].barh(names, dsrs, color=colors, edgecolor="black", linewidth=1.2)
    axes[0].set_xlabel("Deadline Salvage Rate (DSR %)", fontweight="bold")
    axes[0].set_title(f"Author-Scale DSR Evaluation\n(500 DAGs x 10 Runs = {total_instances:,} Instances)", fontweight="bold")
    axes[0].set_xlim(0, 100)
    axes[0].grid(axis="x", alpha=0.3, linestyle="--")
    for bar in bars1:
        w = bar.get_width()
        axes[0].text(w + 1.5, bar.get_y() + bar.get_height() / 2, f"{w:.1f}%", va="center", fontweight="bold")

    # Bar 2: Warning Lead Time
    bars2 = axes[1].barh(names, leads, color=colors, edgecolor="black", linewidth=1.2)
    axes[1].set_xlabel("Warning Lead Time (ms)", fontweight="bold")
    axes[1].set_title("Warning Lead Time Comparison (ms)\n(Before Absolute Deadline D)", fontweight="bold")
    axes[1].set_xlim(0, max(leads) * 1.25)
    axes[1].grid(axis="x", alpha=0.3, linestyle="--")
    for bar in bars2:
        w = bar.get_width()
        axes[1].text(w + 3.0, bar.get_y() + bar.get_height() / 2, f"{w:.1f} ms", va="center", fontweight="bold")

    plt.tight_layout()
    out_png = "figure_author_scale_5000_benchmark.png"
    plt.savefig(out_png, dpi=300)
    plt.close()
    print(f"Publication-ready chart saved to {out_png}")
    print(f"Total benchmark elapsed time: {time.time() - t0:.2f}s")
    return results

if __name__ == "__main__":
    n_d = 500
    n_r = 10
    if len(sys.argv) > 1:
        n_d = int(sys.argv[1])
    if len(sys.argv) > 2:
        n_r = int(sys.argv[2])
    run_benchmark(n_dags=n_d, n_runs_per_dag=n_r, seed=42)
