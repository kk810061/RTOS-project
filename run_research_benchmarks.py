#!/usr/bin/env python3
"""
Publication-Grade Research Evaluation Engine for Dual-Novelty Real-Time DAG Framework:
1. Context-Coupled Plaxity (Correlated Environmental Scene Modeling & Bayesian Conjugate Residual Updates)
2. Active RTOS Dynamic Slack Reclamation (Non-Preemptive NP-EDF Priority Escalation & Amdahl-Aware DVFS)
3. 2x2 Ablation Analysis (None vs Priority-Only vs DVFS-Only vs Combined)
4. Lead-Time Sensitivity S-Curve (Causality Verification)
5. Comprehensive Statistical Benchmarking with 95% Wilson Confidence Intervals
6. High-Resolution Publication Figure Generation
"""

import os
import glob
import math
import heapq
import numpy as np
import pandas as pd
import networkx as nx
import yaml
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------
# 1. MATHEMATICAL FOUNDATIONS & RECURSIONS (TOBA ET AL.)
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
    with open(filepath, "r") as f:
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

    levels = {}
    for v in topo:
        preds = list(G.predecessors(v))
        levels[v] = 0 if not preds else max(levels[u] for u in preds) + 1
    max_lvl = max(levels.values()) if levels else 1

    # Domain-Grounded Autonomous Driving Sensitivities and CPU fractions
    # Early stages (Perception): High sensitivity to clutter, compute-bound
    # Late stages (Control/Actuation): Insensitive, deterministic loops
    rho = {}
    alpha = {}
    for v in topo:
        norm_lvl = levels[v] / max_lvl
        if norm_lvl < 0.35:      # Perception / Point cloud
            rho[v] = float(np.clip(0.85 - 0.2 * norm_lvl, 0.65, 0.95))
            alpha[v] = 0.85     # 85% CPU-bound, 15% memory stalls
        elif norm_lvl < 0.70:    # Tracking / Fusion / Planning
            rho[v] = float(np.clip(0.55 - 0.3 * (norm_lvl - 0.35), 0.30, 0.65))
            alpha[v] = 0.75     # 75% CPU-bound
        else:                    # Vehicle Control / CAN Bus / Actuation
            rho[v] = float(np.clip(0.15 - 0.1 * (norm_lvl - 0.70), 0.05, 0.20))
            alpha[v] = 0.50     # 50% CPU-bound

    # Downstream remaining sensitivity: weighted sum along remaining paths
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
        "th_time": th_time, "rem_t": rem_t, "cp_wcet": cp[sink],
        "levels": levels, "rho": rho, "alpha": alpha,
        "rem_sens": rem_sens
    }

def make_background(rng, D, intensity):
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

# ----------------------------------------------------------------------------
# 2. RIGOROUS CLOSED-LOOP SCHEDULER & BAYESIAN ESTIMATOR
# ----------------------------------------------------------------------------
def calculate_amdahl_duration(nom_dur, alpha_v, dvfs_ratio=1.33):
    """
    Amdahl-aware execution time scaling under DVFS frequency boost.
    f_boost / f_nom = 1.33 (e.g., 1.5 GHz -> 2.0 GHz).
    Speedup applies only to the CPU-bound portion alpha_v.
    """
    scaled = nom_dur * (alpha_v / dvfs_ratio + (1.0 - alpha_v))
    return max(1, int(round(scaled)))

def simulate_closed_loop_job(ctx, c_job_nominal, D, num_cores, bg_tasks,
                             detector_mode="tecs_time",
                             enable_priority_escalation=False,
                             enable_dvfs_boost=False,
                             dvfs_ratio=1.33,
                             alarm_delay_tau=0):
    """
    Simulates multi-core NP-EDF execution with:
    - Non-preemption preserved for currently executing cores.
    - Online Conjugate Bayesian estimation of latent context xi.
    - O(1) Effective Plaxity Shift lookup.
    - 2x2 Factorial Mitigation (Priority Escalation only, DVFS only, Both, Neither).
    - Injected notification delay tau for causality testing.
    """
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
    ast, ft = {}, {}
    gantt = []

    # Online Bayesian Conjugate State: xi ~ N(mu_0, sigma_0^2)
    # Precision-weighted accumulators
    sigma_0 = 1.0
    accum_prec = 1.0 / (sigma_0 ** 2)
    accum_num = 0.0

    alarmed = [False]
    t_alarm = [None]
    alarm_node = [-1]
    mitigation_active = [False]
    mitigation_trigger_time = [None]

    def get_current_posterior_mean():
        if accum_prec <= 1.0 / (sigma_0 ** 2):
            return 0.0
        return accum_num / accum_prec

    def trigger_mitigation(t):
        if not mitigation_active[0]:
            mitigation_active[0] = True
            mitigation_trigger_time[0] = t
            # Strictly non-preemptive: running tasks on cores continue untouched.
            # Only ready queue ordering and unstarted task durations are updated.
            new_ready = []
            for it in ready:
                prio_key, s_seq, is_dag, vid, dur = it
                if is_dag:
                    # Priority escalation: assign emergency priority (-1)
                    new_prio = -1 if enable_priority_escalation else D
                    # DVFS boost: scale CPU-bound duration via Amdahl's Law
                    new_dur = calculate_amdahl_duration(dur, ctx["alpha"][vid], dvfs_ratio) if enable_dvfs_boost else dur
                    new_ready.append((new_prio, s_seq, is_dag, vid, new_dur))
                else:
                    new_ready.append(it)
            heapq.heapify(new_ready)
            ready.clear()
            ready.extend(new_ready)

    def check_detection_and_dispatch(t):
        # Check if delayed mitigation is scheduled to fire now
        if alarmed[0] and not mitigation_active[0] and (enable_priority_escalation or enable_dvfs_boost):
            if t >= t_alarm[0] + alarm_delay_tau:
                trigger_mitigation(t)

        # Early Detection Check
        if not alarmed[0]:
            hat_xi = get_current_posterior_mean()
            
            for item in ready:
                if item[2]:  # is_dag
                    vid = item[3]
                    threshold = th_vec[vid]

                    if detector_mode == "tecs_time":
                        # Toba Baseline (Offline lookup assuming nominal independence)
                        t_eff = min(t, D)
                        miss_prob = ctx["Fm"][topo_idx[vid], t_eff]
                    elif detector_mode == "context_coupled":
                        # Idea 3: Conjugate Bayesian Shift Theorem
                        delta_rem = max(0.0, hat_xi * ctx["rem_sens"][vid])
                        t_eff = int(np.clip(round(t + delta_rem), 0, D))
                        miss_prob = ctx["Fm"][topo_idx[vid], t_eff]
                    else:
                        t_eff = min(t, D)
                        miss_prob = ctx["Fm"][topo_idx[vid], t_eff]

                    if miss_prob > (1.0 - threshold):
                        alarmed[0] = True
                        t_alarm[0] = t
                        alarm_node[0] = vid

                        if (enable_priority_escalation or enable_dvfs_boost):
                            if alarm_delay_tau == 0:
                                trigger_mitigation(t)
                        break

        # Dispatch available cores to ready tasks (Non-preemptive NP-EDF)
        while free and ready:
            _, _, is_dag, vid, dur = heapq.heappop(ready)
            core = free.pop()
            if is_dag:
                ast[vid] = t

            gantt.append({
                "core": int(core),
                "is_dag": bool(is_dag),
                "vid": int(vid) if is_dag else -1,
                "label": f"Node {vid}" if is_dag else "BG Task",
                "start": int(t),
                "duration": int(dur),
                "end": int(t + dur)
            })
            heapq.heappush(events, (t + dur, 0, (is_dag, vid), core))

    check_detection_and_dispatch(0)
    n = G.number_of_nodes()
    nonlocal_accum = [accum_prec, accum_num]

    while len(ft) < n and events:
        t, et, pay, cid = heapq.heappop(events)
        batch = [(et, pay, cid)]
        while events and events[0][0] == t:
            _, e2, p2, c2 = heapq.heappop(events)
            batch.append((e2, p2, c2))
        for et, pay, core in batch:
            if et == 0:
                free.append(core)
                is_dag, vid = pay
                if is_dag:
                    ft[vid] = t
                    # Update Conjugate Bayesian Residual
                    actual_dur = t - ast[vid]
                    mu_v = ctx["mu"][vid]
                    sigma_v = ctx["sigma"][vid]
                    rho_v = ctx["rho"][vid]
                    
                    # Weight and standardized observation
                    w_v = (rho_v * mu_v) / sigma_v
                    y_v = (actual_dur - mu_v) / sigma_v
                    
                    accum_prec += w_v ** 2
                    accum_num += w_v * y_v

                    for s in G.successors(vid):
                        in_deg[s] -= 1
                        if in_deg[s] == 0:
                            seq[0] += 1
                            prio = -1 if (mitigation_active[0] and enable_priority_escalation) else D
                            dur_s = calculate_amdahl_duration(c_job[s], ctx["alpha"][s], dvfs_ratio) if (mitigation_active[0] and enable_dvfs_boost) else c_job[s]
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
        "alarm_node": alarm_node[0],
        "t_alarm": t_alarm[0],
        "exit_time": exit_time,
        "deadline": D,
        "lead_time": max(0, D - t_alarm[0]) if (alarmed[0] and t_alarm[0] is not None) else 0,
        "mitigated": mitigation_active[0],
        "gantt": gantt
    }

# ----------------------------------------------------------------------------
# 3. STATISTICAL UTILITIES (WILSON SCORE INTERVALS & WELCH'S T-TEST)
# ----------------------------------------------------------------------------
def wilson_ci(k, n, confidence=0.95):
    """Computes exact Wilson score confidence interval for proportions."""
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    z = stats.norm.ppf((1 + confidence) / 2)
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = (z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    low = max(0.0, center - margin)
    high = min(1.0, center + margin)
    return p * 100.0, low * 100.0, high * 100.0

def mean_ci(data, confidence=0.95):
    """Computes sample mean and Student's t 95% confidence interval."""
    if not data:
        return 0.0, 0.0, 0.0
    arr = np.array(data, dtype=np.float64)
    m = float(np.mean(arr))
    if len(arr) < 2:
        return m, m, m
    se = stats.sem(arr)
    ci = se * stats.t.ppf((1 + confidence) / 2, len(arr) - 1)
    return m, max(0.0, m - ci), m + ci

# ----------------------------------------------------------------------------
# 4. LARGE-SCALE BENCHMARK EVALUATION (PHASE 4)
# ----------------------------------------------------------------------------
def run_large_scale_benchmarks(dags_dir="RD-Gen/DAGs", n_dags=20, n_jobs=30, intensity=1.0, seed=42):
    dag_files = sorted(glob.glob(f"{dags_dir}/**/dag_*.yaml", recursive=True))
    # Stratified selection across diverse DAG topologies
    stride = max(1, len(dag_files) // n_dags)
    selected_dags = dag_files[::stride][:n_dags]

    print("\n" + "=" * 80)
    print(f"LARGE-SCALE BENCHMARK: {len(selected_dags)} DIVERSE DAGs x {n_jobs} JOBS (TOTAL {len(selected_dags)*n_jobs} RUNS)")
    print("=" * 80)

    regimes = [
        ("1. Toba Baseline (Nominal)",           "tecs_time",        False, False, 0.0),
        ("2. Toba under Correlated Clutter",     "tecs_time",        False, False, 0.35),
        ("3. Context-Coupled Plaxity (Idea 3)",  "context_coupled",  False, False, 0.35),
        ("4. Full Dual Novelty (Ideas 3+1)",     "context_coupled",  True,  True,  0.35),
    ]

    records = []
    regime_leads = {r[0]: [] for r in regimes}

    for reg_name, det_mode, prio_on, dvfs_on, rho_env in regimes:
        tp, fp, tn, fn = 0, 0, 0, 0
        lead_times = []
        salvaged = 0
        impending_misses = 0

        rng = np.random.default_rng(seed)

        for yf in selected_dags:
            ctx = build_dag_context(yf)
            topo, D = ctx["topo"], ctx["D"]

            for _ in range(n_jobs):
                # Correlated environmental clutter event
                if rho_env > 0.0:
                    xi_true = rng.normal(1.3, 0.3) if rng.random() < 0.45 else rng.normal(0.0, 0.25)
                else:
                    xi_true = 0.0

                c_job_nom = {}
                for v in topo:
                    cdf = ctx["cdf"][v]
                    base = min(int(np.searchsorted(cdf, rng.random(), side="right")), len(cdf) - 1)
                    mult = 1.0 + ctx["rho"][v] * xi_true
                    c_job_nom[v] = max(1, int(round(base * mult)))

                bg_tasks = make_background(rng, D, intensity)

                # Counterfactual unmitigated run to detect true deadline failure
                unmit = simulate_closed_loop_job(ctx, c_job_nom, D, num_cores=8, bg_tasks=bg_tasks,
                                                 detector_mode=det_mode,
                                                 enable_priority_escalation=False,
                                                 enable_dvfs_boost=False)
                would_miss = unmit["actual_miss"]

                # Evaluated run
                sim = simulate_closed_loop_job(ctx, c_job_nom, D, num_cores=8, bg_tasks=bg_tasks,
                                               detector_mode=det_mode,
                                               enable_priority_escalation=prio_on,
                                               enable_dvfs_boost=dvfs_on)
                actual_miss = sim["actual_miss"]
                alarmed = sim["alarmed"]

                if would_miss:
                    impending_misses += 1
                    if (prio_on or dvfs_on) and not actual_miss:
                        salvaged += 1

                if would_miss and alarmed:
                    tp += 1
                    lead_times.append(sim["lead_time"])
                elif not would_miss and alarmed:
                    fp += 1
                elif not would_miss and not alarmed:
                    tn += 1
                else:
                    fn += 1

        regime_leads[reg_name] = lead_times
        total_runs = len(selected_dags) * n_jobs
        
        miss_p, miss_lo, miss_hi = wilson_ci(impending_misses, total_runs)
        rec_p, rec_lo, rec_hi = wilson_ci(tp, tp + fn)
        prec_p, prec_lo, prec_hi = wilson_ci(tp, tp + fp)
        fpr_p, fpr_lo, fpr_hi = wilson_ci(fp, tn + fp)
        dsr_p, dsr_lo, dsr_hi = wilson_ci(salvaged, impending_misses)
        lm, l_lo, l_hi = mean_ci(lead_times)

        records.append({
            "Regime": reg_name,
            "Impending Miss % [95% CI]": f"{miss_p:.1f}% [{miss_lo:.1f}, {miss_hi:.1f}]",
            "Recall % [95% CI]": f"{rec_p:.1f}% [{rec_lo:.1f}, {rec_hi:.1f}]",
            "Precision % [95% CI]": f"{prec_p:.1f}% [{prec_lo:.1f}, {prec_hi:.1f}]",
            "FPR % [95% CI]": f"{fpr_p:.1f}% [{fpr_lo:.1f}, {fpr_hi:.1f}]",
            "Lead Time (ms) [95% CI]": f"{lm:.1f} [{l_lo:.1f}, {l_hi:.1f}]",
            "DSR % [95% CI]": f"{dsr_p:.1f}% [{dsr_lo:.1f}, {dsr_hi:.1f}]",
            "Salvaged Ratio": f"{salvaged}/{impending_misses}"
        })

    df = pd.DataFrame(records)
    print("\n" + df.to_string(index=False))
    df.to_csv("benchmark_large_scale_results.csv", index=False)
    print("\n[+] Benchmark results saved to benchmark_large_scale_results.csv")
    return df, regime_leads

# ----------------------------------------------------------------------------
# 5. ABLATION STUDY: 2x2 FACTORIAL EXPERIMENT (PHASE 3)
# ----------------------------------------------------------------------------
def run_ablation_study(dags_dir="RD-Gen/DAGs", n_dags=10, n_jobs=25, seed=777):
    dag_files = sorted(glob.glob(f"{dags_dir}/**/dag_*.yaml", recursive=True))
    selected_dags = dag_files[::max(1, len(dag_files)//n_dags)][:n_dags]

    print("\n" + "=" * 80)
    print("2x2 FACTORIAL ABLATION STUDY: MITIGATION MECHANISMS")
    print("=" * 80)

    ablation_configs = [
        ("None (Passive Alarm)", False, False),
        ("Priority Escalation Only", True, False),
        ("Amdahl DVFS Boost Only", False, True),
        ("Combined (Dual Mitigate)", True, True)
    ]

    ablation_results = []
    rng = np.random.default_rng(seed)

    for name, prio_on, dvfs_on in ablation_configs:
        salvaged = 0
        impending = 0

        for yf in selected_dags:
            ctx = build_dag_context(yf)
            topo, D = ctx["topo"], ctx["D"]

            for _ in range(n_jobs):
                xi = rng.normal(1.3, 0.3) if rng.random() < 0.45 else rng.normal(0.0, 0.25)
                c_job_nom = {}
                for v in topo:
                    cdf = ctx["cdf"][v]
                    base = min(int(np.searchsorted(cdf, rng.random(), side="right")), len(cdf) - 1)
                    mult = 1.0 + ctx["rho"][v] * xi
                    c_job_nom[v] = max(1, int(round(base * mult)))

                bg_tasks = make_background(rng, D, intensity=1.0)

                unmit = simulate_closed_loop_job(ctx, c_job_nom, D, num_cores=8, bg_tasks=bg_tasks,
                                                 detector_mode="context_coupled",
                                                 enable_priority_escalation=False,
                                                 enable_dvfs_boost=False)
                if unmit["actual_miss"]:
                    impending += 1
                    sim = simulate_closed_loop_job(ctx, c_job_nom, D, num_cores=8, bg_tasks=bg_tasks,
                                                   detector_mode="context_coupled",
                                                   enable_priority_escalation=prio_on,
                                                   enable_dvfs_boost=dvfs_on)
                    if not sim["actual_miss"]:
                        salvaged += 1

        p, lo, hi = wilson_ci(salvaged, impending)
        ablation_results.append({
            "Mechanism": name,
            "Priority": "Active (-1)" if prio_on else "Nominal (D)",
            "DVFS": "Boost (1.33x)" if dvfs_on else "Nominal (1.0x)",
            "Salvaged": salvaged,
            "Impending Misses": impending,
            "DSR %": p,
            "DSR_lo": lo,
            "DSR_hi": hi
        })

    df_ab = pd.DataFrame(ablation_results)
    print("\n" + df_ab.to_string(index=False))
    df_ab.to_csv("ablation_results.csv", index=False)
    print("\n[+] Ablation results saved to ablation_results.csv")
    return df_ab

# ----------------------------------------------------------------------------
# 6. LEAD-TIME SENSITIVITY S-CURVE: CAUSALITY PROOF (PHASE 3)
# ----------------------------------------------------------------------------
def run_lead_time_scurve(dags_dir="RD-Gen/DAGs", n_dags=8, n_jobs=25, seed=999):
    dag_files = sorted(glob.glob(f"{dags_dir}/**/dag_*.yaml", recursive=True))
    selected_dags = dag_files[::max(1, len(dag_files)//n_dags)][:n_dags]

    print("\n" + "=" * 80)
    print("LEAD-TIME SENSITIVITY S-CURVE: CAUSALITY ANALYSIS")
    print("=" * 80)

    tau_delays = [0, 25, 50, 75, 100, 150, 200]
    scurve_results = []
    rng = np.random.default_rng(seed)

    for tau in tau_delays:
        salvaged = 0
        impending = 0

        for yf in selected_dags:
            ctx = build_dag_context(yf)
            topo, D = ctx["topo"], ctx["D"]

            for _ in range(n_jobs):
                xi = rng.normal(1.3, 0.3) if rng.random() < 0.45 else rng.normal(0.0, 0.25)
                c_job_nom = {}
                for v in topo:
                    cdf = ctx["cdf"][v]
                    base = min(int(np.searchsorted(cdf, rng.random(), side="right")), len(cdf) - 1)
                    mult = 1.0 + ctx["rho"][v] * xi
                    c_job_nom[v] = max(1, int(round(base * mult)))

                bg_tasks = make_background(rng, D, intensity=1.0)

                unmit = simulate_closed_loop_job(ctx, c_job_nom, D, num_cores=8, bg_tasks=bg_tasks,
                                                 detector_mode="context_coupled",
                                                 enable_priority_escalation=False,
                                                 enable_dvfs_boost=False)
                if unmit["actual_miss"]:
                    impending += 1
                    sim = simulate_closed_loop_job(ctx, c_job_nom, D, num_cores=8, bg_tasks=bg_tasks,
                                                   detector_mode="context_coupled",
                                                   enable_priority_escalation=True,
                                                   enable_dvfs_boost=True,
                                                   alarm_delay_tau=tau)
                    if not sim["actual_miss"]:
                        salvaged += 1

        p, lo, hi = wilson_ci(salvaged, impending)
        scurve_results.append({
            "tau_delay_ms": tau,
            "salvaged": salvaged,
            "impending": impending,
            "dsr_pct": p,
            "dsr_lo": lo,
            "dsr_hi": hi
        })
        print(f"Delay tau = {tau:3d} ms | Salvaged: {salvaged:2d}/{impending:2d} | DSR: {p:5.1f}% [{lo:4.1f}%, {hi:4.1f}%]")

    df_sc = pd.DataFrame(scurve_results)
    df_sc.to_csv("scurve_results.csv", index=False)
    return df_sc

# ----------------------------------------------------------------------------
# 7. PUBLICATION FIGURE GENERATION (PHASE 5)
# ----------------------------------------------------------------------------
def generate_publication_figures(df_bench, df_ab, df_sc):
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams.update({"font.size": 11, "figure.autolayout": True})

    # Figure 1: Ablation Study 2x2 Bar Chart
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)
    labels = df_ab["Mechanism"].tolist()
    dsr_vals = df_ab["DSR %"].tolist()
    yerr_lo = [dsr - lo for dsr, lo in zip(dsr_vals, df_ab["DSR_lo"])]
    yerr_hi = [hi - dsr for dsr, hi in zip(dsr_vals, df_ab["DSR_hi"])]
    colors = ["#94a3b8", "#38bdf8", "#818cf8", "#10b981"]

    bars = ax.bar(labels, dsr_vals, yerr=[yerr_lo, yerr_hi], capsize=5, color=colors, edgecolor="#0f172a", width=0.55)
    ax.set_ylabel("Deadline Salvage Rate (%)", fontweight="bold")
    ax.set_title("2x2 Factorial Ablation of RTOS Mitigation Mechanisms", fontweight="bold", pad=12)
    ax.set_ylim(0, 100)
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, h + 3),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontweight="bold")
    plt.xticks(rotation=15, ha="right")
    plt.savefig("figure_ablation_matrix.png", dpi=300)
    plt.close()
    print("[+] Saved figure_ablation_matrix.png")

    # Figure 2: Lead-Time Sensitivity S-Curve
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)
    x = df_sc["tau_delay_ms"]
    y = df_sc["dsr_pct"]
    y_lo = df_sc["dsr_lo"]
    y_hi = df_sc["dsr_hi"]

    ax.plot(x, y, "o-", color="#2563eb", linewidth=2.5, markersize=7, label="Empirical DSR")
    ax.fill_between(x, y_lo, y_hi, color="#3b82f6", alpha=0.25, label="95% Wilson CI")
    ax.set_xlabel("Mitigation Notification Delay $\\tau$ (ms)", fontweight="bold")
    ax.set_ylabel("Deadline Salvage Rate (%)", fontweight="bold")
    ax.set_title("Causality Verification: DSR Degradation vs Notification Delay $\\tau$", fontweight="bold", pad=12)
    ax.set_ylim(-2, 105)
    ax.legend(frameon=True, loc="upper right")
    plt.savefig("figure_leadtime_scurve.png", dpi=300)
    plt.close()
    print("[+] Saved figure_leadtime_scurve.png")

# ----------------------------------------------------------------------------
# MAIN EXECUTION
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    df_bench, regime_leads = run_large_scale_benchmarks(n_dags=12, n_jobs=25)
    df_ab = run_ablation_study(n_dags=10, n_jobs=25)
    df_sc = run_lead_time_scurve(n_dags=8, n_jobs=25)
    generate_publication_figures(df_bench, df_ab, df_sc)
    print("\n[+] ALL EXPERIMENTAL PHASES COMPLETE!")
