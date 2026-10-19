#!/usr/bin/env python3
"""
Research Evaluation Pipeline for Dual-Novelty Real-Time DAG Framework:
- Idea 3: Context-Coupled Plaxity (Correlated Environmental Scene Modeling & Bayesian Residual Updates)
- Idea 1: Predict-and-Mitigate (Active RTOS Dynamic Slack Reclamation, DVFS & Emergency Priority Escalation)
Baseline: Toba, Baruah, & Azumi (ACM TECS 2026 / ECRTS 2024)
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

    return {
        "G": G, "topo": topo, "sink": sink, "D": D, "n": len(topo),
        "Fm": Fm, "mu": mu, "sigma": {v: max(1.0, mu[v] / 2.0) for v in topo},
        "wcet": {v: G.nodes[v]["wcet"] for v in topo},
        "cdf": {v: np.cumsum(pmfs[v]) for v in topo},
        "th_time": th_time, "rem_t": rem_t, "cp_wcet": cp[sink],
        "levels": levels
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
# 2. CLOSED-LOOP SIMULATOR (WITH IDEA 3 CONTEXT & IDEA 1 MITIGATION)
# ----------------------------------------------------------------------------
def simulate_closed_loop_job(ctx, c_job_nominal, D, num_cores, bg_tasks,
                             detector_mode="tecs_time",
                             rho_coupling=0.0,
                             enable_mitigation=False,
                             mitigation_boost=0.75):
    """
    Simulates an 8-core NP-EDF schedule with online early detection and dynamic mitigation:
    - detector_mode: 'tecs_time' (Toba baseline) or 'context_coupled' (Idea 3)
    - enable_mitigation: True triggers Idea 1 (Emergency Priority Escalation & DVFS/QoS Acceleration)
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

    completed_residuals = []
    alarmed = [False]
    t_alarm = [None]
    alarm_node = [-1]
    mitigation_active = [False]

    def check_detection_and_dispatch(t):
        # 1. Early Detection Check: At dispatch or when waiting in ready queue
        if not alarmed[0]:
            hat_xi = float(np.mean(completed_residuals)) if len(completed_residuals) > 0 else 0.0
            
            for item in ready:
                if item[2]:  # is_dag
                    vid = item[3]
                    threshold = th_vec[vid]

                    if detector_mode == "tecs_time":
                        # Toba Baseline (Offline lookup assuming nominal independence)
                        t_eff = min(t, D)
                        miss_prob = ctx["Fm"][topo_idx[vid], t_eff]
                    elif detector_mode == "context_coupled":
                        # Idea 3: Online Bayesian Context Estimate
                        delta_rem = rho_coupling * hat_xi * ctx["rem_t"][vid]
                        t_eff = int(np.clip(t + delta_rem, 0, D))
                        miss_prob = ctx["Fm"][topo_idx[vid], t_eff]
                    else:
                        t_eff = min(t, D)
                        miss_prob = ctx["Fm"][topo_idx[vid], t_eff]

                    if miss_prob > (1.0 - threshold):
                        alarmed[0] = True
                        t_alarm[0] = t
                        alarm_node[0] = vid

                        # IDEA 1: DYNAMIC RTOS MITIGATION TRIGGER
                        if enable_mitigation and not mitigation_active[0]:
                            mitigation_active[0] = True
                            # Re-prioritize all ready DAG tasks to EMERGENCY priority (-1)
                            # and apply DVFS/QoS acceleration (mitigation_boost)
                            new_ready = []
                            for it in ready:
                                if it[2]: # is_dag
                                    new_ready.append((-1, it[1], it[2], it[3], max(1, int(round(it[4] * mitigation_boost)))))
                                else:
                                    new_ready.append(it)
                            heapq.heapify(new_ready)
                            ready.clear()
                            ready.extend(new_ready)
                        break

        # 2. Dispatch available cores to ready tasks
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
                    # Update Bayesian residual for Idea 3
                    actual_dur = t - ast[vid]
                    res = (actual_dur - ctx["mu"][vid]) / ctx["sigma"][vid]
                    completed_residuals.append(res)

                    for s in G.successors(vid):
                        in_deg[s] -= 1
                        if in_deg[s] == 0:
                            seq[0] += 1
                            # If mitigation active, dispatch downstream nodes at emergency priority
                            prio = -1 if mitigation_active[0] else D
                            c_s = max(1, int(round(c_job[s] * mitigation_boost))) if mitigation_active[0] else c_job[s]
                            heapq.heappush(ready, (prio, seq[0], True, s, c_s))
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
# 3. COMPREHENSIVE BENCHMARK EVALUATION
# ----------------------------------------------------------------------------
def evaluate_framework(dags_dir="RD-Gen/DAGs", n_dags=8, n_jobs=50, intensity=1.0, seed=42):
    dag_files = sorted(glob.glob(f"{dags_dir}/**/dag_*.yaml", recursive=True))[:n_dags]
    print(f"\n================================================================================")
    print(f"EVALUATING DUAL-NOVELTY FRAMEWORK OVER {len(dag_files)} DAGs ({n_jobs} runs each, intensity={intensity})")
    print(f"================================================================================")

    regimes = [
        ("1. Toba Baseline (Nominal Independent)", "tecs_time", 0.0, False),
        ("2. Toba under Correlated Clutter",        "tecs_time", 0.35, False),
        ("3. Idea 3: Context-Coupled Plaxity",     "context_coupled", 0.35, False),
        ("4. Idea 3 + Idea 1: Predict & Mitigate", "context_coupled", 0.35, True),
    ]

    results = []

    for reg_name, det_mode, rho, mit_on in regimes:
        tp_tot, fp_tot, tn_tot, fn_tot = 0, 0, 0, 0
        lead_list = []
        salvaged_count = 0
        impending_miss_count = 0

        rng = np.random.default_rng(seed)

        for d_idx, yf in enumerate(dag_files):
            ctx = build_dag_context(yf)
            topo, D = ctx["topo"], ctx["D"]

            for job_id in range(n_jobs):
                # Sample environmental scene clutter event (xi > 0)
                if rho > 0.0:
                    xi = rng.normal(1.2, 0.4) if rng.random() < 0.40 else rng.normal(0.0, 0.3)
                else:
                    xi = 0.0

                c_job_nom = {}
                for v in topo:
                    cdf = ctx["cdf"][v]
                    base = min(int(np.searchsorted(cdf, rng.random(), side="right")), len(cdf) - 1)
                    mult = 1.0 + rho * xi
                    c_job_nom[v] = max(1, int(round(base * mult)))

                bg_tasks = make_background(rng, D, intensity)

                # Counterfactual: what happens without mitigation?
                sim_unmit = simulate_closed_loop_job(ctx, c_job_nom, D, num_cores=8, bg_tasks=bg_tasks,
                                                     detector_mode=det_mode, rho_coupling=rho,
                                                     enable_mitigation=False)
                would_miss = sim_unmit["actual_miss"]

                # Run specified regime
                sim = simulate_closed_loop_job(ctx, c_job_nom, D, num_cores=8, bg_tasks=bg_tasks,
                                               detector_mode=det_mode, rho_coupling=rho,
                                               enable_mitigation=mit_on, mitigation_boost=0.75)

                actual_miss = sim["actual_miss"]
                alarmed = sim["alarmed"]

                if would_miss:
                    impending_miss_count += 1
                    if mit_on and not actual_miss:
                        salvaged_count += 1

                if would_miss and alarmed:
                    tp_tot += 1
                    lead_list.append(sim["lead_time"])
                elif not would_miss and alarmed:
                    fp_tot += 1
                elif not would_miss and not alarmed:
                    tn_tot += 1
                else:
                    fn_tot += 1

        tot = len(dag_files) * n_jobs
        miss_rate = (impending_miss_count / tot) * 100
        prec = (tp_tot / (tp_tot + fp_tot) * 100) if (tp_tot + fp_tot) > 0 else 0.0
        rec = (tp_tot / (tp_tot + fn_tot) * 100) if (tp_tot + fn_tot) > 0 else 100.0
        fpr = (fp_tot / (tn_tot + fp_tot) * 100) if (tn_tot + fp_tot) > 0 else 0.0
        avg_lead = float(np.mean(lead_list)) if lead_list else 0.0
        dsr = (salvaged_count / impending_miss_count * 100) if impending_miss_count > 0 else 0.0

        results.append({
            "Regime": reg_name,
            "Impending Miss %": f"{miss_rate:.1f}%",
            "Recall %": f"{rec:.1f}%",
            "Precision %": f"{prec:.1f}%",
            "FPR %": f"{fpr:.1f}%",
            "Lead Time": f"{avg_lead:.1f} ms",
            "Deadline Salvage Rate (DSR)": f"{dsr:.1f}%",
            "Salvaged Count": f"{salvaged_count}/{impending_miss_count}"
        })

    df = pd.DataFrame(results)
    print("\n" + df.to_string(index=False))
    print(f"\n================================================================================")
    
    df.to_csv("novelty_evaluation_results.csv", index=False)
    print("[+] Results saved to novelty_evaluation_results.csv")
    return df

if __name__ == "__main__":
    evaluate_framework()
