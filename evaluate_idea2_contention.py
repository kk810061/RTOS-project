#!/usr/bin/env python3
"""
Comparative Evaluation of Novelty Idea 2: Queue- & Contention-Aware Plaxity
Evaluates whether conditioning plaxity on instantaneous ready-queue backlog
and core contention improves Precision, Lead Time, and Deadline Salvage Rate (DSR).
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

from run_unified_reproducible_evaluation import (
    build_dag_context, make_background, calculate_amdahl_duration,
    wilson_ci, mean_ci
)

def simulate_contention_aware_job(ctx, c_job_nominal, D, num_cores, bg_tasks,
                                  detector_mode="context_coupled",
                                  enable_contention_awareness=False,
                                  enable_mitigation=False,
                                  dvfs_ratio=1.33):
    """
    Enhanced simulator integrating Idea 2:
    t_eff = min(D, t + Delta_mu(xi) + B_contention(t))
    where B_contention is the multi-core queuing backlog ahead of the task.
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

    sigma_0 = 1.0
    accum_prec = 1.0 / (sigma_0 ** 2)
    accum_num = 0.0

    alarmed = [False]
    t_alarm = [None]
    alarm_node = [-1]
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

    def compute_contention_backlog():
        """
        Idea 2: Estimates multi-core queuing interference.
        Total unserviced workload waiting in ready queue + busy cores.
        """
        ready_workload = sum(it[4] for it in ready if not it[2])
        return int(round(ready_workload / num_cores))

    def check_detection_and_dispatch(t):
        if not alarmed[0]:
            hat_xi = get_current_posterior_mean()
            queue_delay = compute_contention_backlog() if enable_contention_awareness else 0
            
            for item in ready:
                if item[2]:  # is_dag
                    vid = item[3]
                    threshold = th_vec[vid]

                    if detector_mode == "tecs_time":
                        t_eff = min(t, D)
                    elif detector_mode == "context_coupled":
                        delta_rem = max(0.0, hat_xi * ctx["rem_sens"][vid])
                        t_eff = int(np.clip(round(t + delta_rem + queue_delay), 0, D))
                    else:
                        t_eff = min(t, D)

                    miss_prob = ctx["Fm"][topo_idx[vid], t_eff]

                    if miss_prob > (1.0 - threshold):
                        alarmed[0] = True
                        t_alarm[0] = t
                        alarm_node[0] = vid

                        if enable_mitigation:
                            trigger_mitigation(t)
                        break

        # Non-preemptive dispatch
        while free and ready:
            _, _, is_dag, vid, dur = heapq.heappop(ready)
            core = free.pop()
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
        "alarm_node": alarm_node[0],
        "t_alarm": t_alarm[0],
        "exit_time": exit_time,
        "deadline": D,
        "lead_time": max(0, D - t_alarm[0]) if (alarmed[0] and t_alarm[0] is not None) else 0,
        "mitigated": mitigation_active[0]
    }

def run_idea2_comparison(n_dags=20, n_jobs=50, seed=42):
    dag_files = sorted(glob.glob("RD-Gen/DAGs/**/dag_*.yaml", recursive=True))
    stride = max(1, len(dag_files) // n_dags)
    selected_dags = dag_files[::stride][:n_dags]

    rng = np.random.default_rng(seed)
    cohort = []

    for d_idx, yf in enumerate(selected_dags):
        ctx = build_dag_context(yf)
        topo, D = ctx["topo"], ctx["D"]

        for j_idx in range(n_jobs):
            is_clutter = (rng.random() < 0.40)
            xi = float(rng.normal(1.3, 0.3)) if is_clutter else float(rng.normal(0.0, 0.25))

            c_job_nom = {}
            for v in topo:
                cdf = ctx["cdf"][v]
                base = min(int(np.searchsorted(cdf, rng.random(), side="right")), len(cdf) - 1)
                mult = 1.0 + ctx["rho"][v] * xi
                c_job_nom[v] = max(1, int(round(base * mult)))

            bg_tasks = make_background(rng, D, intensity=1.0)

            unmit = simulate_contention_aware_job(ctx, c_job_nom, D, num_cores=8, bg_tasks=bg_tasks,
                                                  detector_mode="context_coupled",
                                                  enable_contention_awareness=False,
                                                  enable_mitigation=False)
            would_miss = unmit["actual_miss"]

            cohort.append({
                "ctx": ctx,
                "c_job_nom": c_job_nom,
                "D": D,
                "bg_tasks": bg_tasks,
                "would_miss": would_miss
            })

    total_jobs = len(cohort)
    total_impending = sum(1 for item in cohort if item["would_miss"])

    configs = [
        ("1. Toba Baseline",                        "tecs_time",       False, False),
        ("2. Idea 3: Context-Coupled Plaxity",     "context_coupled", False, False),
        ("3. Ideas 3 + 2: Context + Contention",   "context_coupled", True,  False),
        ("4. Ideas 3 + 1: Context + Mitigation",   "context_coupled", False, True),
        ("5. Triad Novelty (Ideas 3 + 2 + 1)",     "context_coupled", True,  True),
    ]

    records = []
    print("=" * 85)
    print(f"EVALUATING NOVELTY IDEA 2 (CONTENTION-AWARE PLAXITY) ON UNIFIED 1,000-JOB COHORT")
    print("=" * 85)

    for name, det_mode, content_on, mit_on in configs:
        tp, fp, tn, fn = 0, 0, 0, 0
        lead_times = []
        salvaged = 0

        for item in cohort:
            sim = simulate_contention_aware_job(item["ctx"], item["c_job_nom"], item["D"],
                                                num_cores=8, bg_tasks=item["bg_tasks"],
                                                detector_mode=det_mode,
                                                enable_contention_awareness=content_on,
                                                enable_mitigation=mit_on)
            actual_miss = sim["actual_miss"]
            alarmed = sim["alarmed"]
            would_miss = item["would_miss"]

            if would_miss:
                if mit_on and not actual_miss:
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

        rec_p, rec_lo, rec_hi = wilson_ci(tp, tp + fn)
        prec_p, prec_lo, prec_hi = wilson_ci(tp, tp + fp)
        fpr_p, fpr_lo, fpr_hi = wilson_ci(fp, tn + fp)
        dsr_p, dsr_lo, dsr_hi = wilson_ci(salvaged, total_impending)
        lm, l_lo, l_hi = mean_ci(lead_times)

        records.append({
            "Framework Configuration": name,
            "Recall %": f"{rec_p:.1f}%",
            "Precision % [95% CI]": f"{prec_p:.1f}% [{prec_lo:.1f}, {prec_hi:.1f}]",
            "FPR %": f"{fpr_p:.1f}%",
            "Lead Time (ms) [95% CI]": f"{lm:.1f} [{l_lo:.1f}, {l_hi:.1f}]",
            "DSR % [95% CI]": f"{dsr_p:.1f}% [{dsr_lo:.1f}, {dsr_hi:.1f}]",
            "Salvaged Ratio": f"{salvaged} / {total_impending}"
        })

    df = pd.DataFrame(records)
    print("\n" + df.to_string(index=False))
    df.to_csv("idea2_contention_evaluation_results.csv", index=False)
    print("\n[+] Saved idea2_contention_evaluation_results.csv")
    return df

if __name__ == "__main__":
    run_idea2_comparison()
