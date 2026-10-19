#!/usr/bin/env python3
"""
Save as: d:/RTOS/dag_deadline_paper/run_exact_specification.py
Run:     python run_exact_specification.py

Faithful reimplementation of Toba, Baruah & Azumi (ACM TECS 2026):
1. Execution-time model: mu = WCET/3, sigma = mu/2, 0.98*N(mu, sigma^2) + 0.02*N(WCET, (0.2*sigma)^2).
2. Backward plaxity-CDF (Eq. 8-12) using lossless log1p/expm1 lower-tail CDF F_miss[v][t].
3. Variable threshold assignment (Eq. 15-18) with min over successors, th_min = 0.80, th_max = 1.00.
4. 8-Core Non-Preemptive EDF (NP-EDF) discrete-event simulator at 275% utilization.
5. Budget-rescaled drift compensation lookup: AST' = D - (D - AST) / s (clamped, with hysteresis).
6. Reports ALL metrics (Precision, Recall, F1, FPR, Lead Time) and paired t-tests across RD-Gen DAGs.
"""

import glob
import heapq
import yaml
import numpy as np
import pandas as pd
import networkx as nx
from scipy import stats

# ==============================================================================
# 1. PAPER-EXACT EXECUTION-TIME MODEL (mu = WCET/3, sigma = mu/2, 98%/2% Mixture)
# ==============================================================================
def create_exact_paper_pmf(wcet_bins: int) -> np.ndarray:
    wcet_bins = max(6, int(wcet_bins))
    x = np.arange(wcet_bins + 1, dtype=np.float64)
    mu = wcet_bins / 3.0
    sigma = mu / 2.0
    p_main = stats.norm.pdf(x, loc=mu, scale=max(sigma, 1e-3))
    p_tail = stats.norm.pdf(x, loc=wcet_bins, scale=max(0.2 * sigma, 1e-3))
    pmf = 0.98 * p_main + 0.02 * p_tail
    pmf[0] = 0.0
    pmf /= np.sum(pmf)
    return pmf

# ==============================================================================
# 2. EXACT EQUATIONS (8)-(12) AND (15)-(18) (Lossless Lower-Tail & Min-Remaining)
# ==============================================================================
def compute_exact_plaxity_and_thresholds(
    G: nx.DiGraph, node_pmfs: dict, deadline_bins: int, th_min: float = 0.80, th_max: float = 1.00
):
    rev_topo = list(reversed(list(nx.topological_sort(G))))
    lst_pmf_dict = {}
    f_miss_dict = {}  # F_miss[v][t] = P(LST(v) <= t) = 1 - plaxity_surv[v][t]
    t_rem_min = {}
    node_mu = {v: float(np.sum(np.arange(len(pmf)) * pmf)) for v, pmf in node_pmfs.items()}

    for v in rev_topo:
        succs = list(G.successors(v))
        c_v = node_pmfs[v]
        L_c = len(c_v)

        if len(succs) == 0:
            # Eq. 8: Sink node LFT is deterministically D
            lft_pmf = np.zeros(deadline_bins + 1, dtype=np.float64)
            lft_pmf[deadline_bins] = 1.0
            t_rem_min[v] = node_mu[v]
        elif len(succs) == 1:
            u = succs[0]
            lft_pmf = lst_pmf_dict[u].copy()
            t_rem_min[v] = node_mu[v] + t_rem_min[u]
        else:
            # Eq. 9-10: Lossless evaluation of 1 - prod_{u in Succ(v)} (1 - F_miss[u][t])
            log_surv = np.zeros(deadline_bins + 1, dtype=np.float64)
            for u in succs:
                log_surv += np.log1p(-np.minimum(f_miss_dict[u], 1.0 - 1e-16))
            f_lft = -np.expm1(log_surv)
            lft_pmf = np.zeros(deadline_bins + 1, dtype=np.float64)
            lft_pmf[0] = f_lft[0]
            lft_pmf[1:] = np.clip(np.diff(f_lft), 0.0, 1.0)
            # Eq. 17: Remaining workload uses MINIMUM over successors
            t_rem_min[v] = node_mu[v] + min(t_rem_min[u] for u in succs)

        # Eq. 11: Backward convolution LST(v) = LFT(v) - C_v
        corr = np.correlate(lft_pmf, c_v, mode="full")
        zero_idx = L_c - 1
        lst_pmf = np.zeros(deadline_bins + 1, dtype=np.float64)
        lst_pmf[0] = np.sum(corr[:zero_idx + 1])
        end_idx = min(len(corr), zero_idx + deadline_bins + 1)
        lst_pmf[1 : (end_idx - zero_idx)] = corr[zero_idx + 1 : end_idx]

        s_tot = np.sum(lst_pmf)
        if s_tot > 0:
            lst_pmf /= s_tot
        lst_pmf_dict[v] = lst_pmf
        f_miss_dict[v] = np.clip(np.cumsum(lst_pmf), 0.0, 1.0)

    # Eq. 18: Linear interpolation between th_min = 0.80 and th_max = 1.00 using t_rem_min
    max_r = max(t_rem_min.values())
    min_r = min(t_rem_min.values())
    span = max(max_r - min_r, 1e-9)
    tecs_thresholds = {
        v: th_max - (th_max - th_min) * ((r - min_r) / span) for v, r in t_rem_min.items()
    }
    return f_miss_dict, t_rem_min, node_mu, tecs_thresholds

# ==============================================================================
# 3. 8-CORE NON-PREEMPTIVE EDF (NP-EDF) DISCRETE-EVENT SIMULATOR
# ==============================================================================
def simulate_8core_np_edf(
    G: nx.DiGraph, c_job: dict, node_subdeadlines: dict, num_cores: int = 8, bg_tasks: list = None
) -> tuple:
    topo = list(nx.topological_sort(G))
    in_deg_rem = {v: G.in_degree(v) for v in topo}

    events = []
    ready_queue = []
    seq = 0

    for v in topo:
        if in_deg_rem[v] == 0:
            heapq.heappush(ready_queue, (node_subdeadlines[v], seq, True, v, c_job[v]))
            seq += 1

    if bg_tasks:
        for rel_t, bg_d, bg_c in bg_tasks:
            heapq.heappush(events, (rel_t, 1, (bg_d, bg_c, seq), -1))
            seq += 1

    free_cores = list(range(num_cores))
    ast_job, ft_job = {}, {}

    def dispatch(t_now):
        while free_cores and ready_queue:
            _, _, is_dag, item_id, dur = heapq.heappop(ready_queue)
            core = free_cores.pop()
            if is_dag:
                ast_job[item_id] = t_now
            heapq.heappush(events, (t_now + dur, 0, (is_dag, item_id), core))

    dispatch(0)

    while len(ft_job) < len(topo) and events:
        cur_t, ev_type, payload, cid = heapq.heappop(events)
        batch = [(ev_type, payload, cid)]
        while events and events[0][0] == cur_t:
            _, et2, pay2, c2 = heapq.heappop(events)
            batch.append((et2, pay2, c2))

        for et, pay, core_id in batch:
            if et == 0:
                free_cores.append(core_id)
                is_dag, item_id = pay
                if is_dag:
                    ft_job[item_id] = cur_t
                    for succ in G.successors(item_id):
                        in_deg_rem[succ] -= 1
                        if in_deg_rem[succ] == 0:
                            seq += 1
                            heapq.heappush(
                                ready_queue, (node_subdeadlines[succ], seq, True, succ, c_job[succ])
                            )
            else:
                bg_d, bg_c, bg_seq = pay
                heapq.heappush(ready_queue, (bg_d, bg_seq, False, -1, bg_c))

        dispatch(cur_t)

    return ast_job, ft_job

# ==============================================================================
# 4. LOAD RD-GEN 20-NODE TOPOLOGIES
# ==============================================================================
def load_rdgen_20node_dags(limit: int = 30) -> list:
    yaml_files = sorted(
        [f for f in glob.glob("RD-Gen/**/*.yaml", recursive=True) if "sample_config" not in f]
    )
    dags = []
    for yf in yaml_files:
        try:
            with open(yf, "r") as f:
                data = yaml.safe_load(f)
            if isinstance(data, dict) and "nodes" in data:
                G = nx.DiGraph()
                for n in data["nodes"]:
                    nid = int(n["id"])
                    wcet_raw = int(n.get("execution_time", 25))
                    G.add_node(nid, wcet=max(12, wcet_raw * 3))
                for e in data.get("links", data.get("edges", [])):
                    G.add_edge(int(e["source"]), int(e["target"]))
                if len(G.nodes) == 20 and nx.is_directed_acyclic_graph(G):
                    dags.append(G)
                    if len(dags) >= limit:
                        break
        except Exception:
            continue
    return dags

# ==============================================================================
# 5. MAIN EVALUATION: TABLE 7 CHECK + BUDGET-RESCALED DRIFT LOOKUP
# ==============================================================================
def evaluate_job_detectors(
    topo, sink, ast_job, ft_job, deadline_bins, f_miss, tecs_thresh, s_est, methods
):
    miss = int(ft_job[sink] > deadline_bins)
    out = {}
    for m in methods:
        alarmed, alarm_ast = False, deadline_bins
        for v in topo:
            ast_v = ast_job[v]
            if m == "TECS + AST' Rescaled [Ours]":
                # Exact remaining-budget rescaling: AST' = D - (D - AST) / s
                rem_budget_rescaled = (deadline_bins - ast_v) / max(s_est, 1e-6)
                ast_lookup = int(round(np.clip(deadline_bins - rem_budget_rescaled, 0, deadline_bins)))
                th = tecs_thresh[v]
            else:
                ast_lookup = min(ast_v, deadline_bins)
                if m == "1.0 [Yano 2023]":
                    th = 1.00
                elif m == "0.99 [Flat]":
                    th = 0.99
                elif m == "0.95 [ECRTS 2024]":
                    th = 0.95
                else:
                    th = tecs_thresh[v]

            # Trigger alarm when P(LST(v) <= ast_lookup) > 1.0 - th
            if f_miss[v][ast_lookup] > (1.0 - th):
                alarmed = True
                alarm_ast = ast_v
                break
        out[m] = (miss, alarmed, max(0, deadline_bins - alarm_ast))
    return out

def main():
    rng = np.random.default_rng(2026)
    dags = load_rdgen_20node_dags(limit=30)
    print(f"Loaded {len(dags)} RD-Gen 20-node DAGs.")

    t7_methods = ["1.0 [Yano 2023]", "0.99 [Flat]", "0.95 [ECRTS 2024]", "0.8-1.0 (time) [TECS 2026]"]
    eval_methods = ["1.0 [Yano 2023]", "0.95 [ECRTS 2024]", "TECS 0.8-1.0 (time)", "TECS + AST' Rescaled [Ours]"]

    scenarios = {
        "Stationary (1.00x)": (1.00, 1.00),
        "Mild Drift (1.00x -> 1.10x)": (1.00, 1.10),
        "Moderate Drift (1.00x -> 1.15x)": (1.00, 1.15),
    }

    all_results_rows = []

    for scen_name, (s_init, s_drift) in scenarios.items():
        all_methods = list(dict.fromkeys(t7_methods + eval_methods))
        stats_per_dag = {m: {"prec": [], "rec": [], "f1": [], "fpr": [], "lead": []} for m in all_methods}

        for G in dags:
            topo = list(nx.topological_sort(G))
            sink = topo[-1]
            node_pmfs = {v: create_exact_paper_pmf(G.nodes[v]["wcet"]) for v in topo}

            cp_wcet = {}
            for v in topo:
                cp_wcet[v] = max((cp_wcet[u] for u in G.predecessors(v)), default=0) + G.nodes[v]["wcet"]
            vol_wcet = sum(G.nodes[v]["wcet"] for v in topo)

            deadline_bins = int(round(cp_wcet[sink] * 1.05))
            f_miss, t_rem_min, node_mu, tecs_thresh = compute_exact_plaxity_and_thresholds(
                G, node_pmfs, deadline_bins, th_min=0.80, th_max=1.00
            )
            subdeadlines = {v: deadline_bins - int(round(t_rem_min[v] - node_mu[v])) for v in topo}

            # Ring buffer (N=20 jobs) for slowdown estimation s
            ring_buf = np.ones(20, dtype=np.float64)
            rb_head = 0
            counts = {m: {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "leads": []} for m in all_methods}

            # 275% normalized utilization on 8 cores: U_total = 8 * 2.75 = 22.0
            total_wcet_load = 8 * 2.75 * deadline_bins
            bg_wcet_load = max(0.0, total_wcet_load - vol_wcet)

            for job_k in range(200):
                true_s = s_init if job_k < 60 else s_drift
                s_raw = float(np.mean(ring_buf))
                # Hysteresis deadband |s - 1.0| <= 0.03, clamped to [0.85, 1.25]
                s_est = 1.0 if abs(s_raw - 1.0) <= 0.03 else float(np.clip(s_raw, 0.85, 1.25))

                c_job, ratios = {}, []
                for v in topo:
                    base_c = int(rng.choice(len(node_pmfs[v]), p=node_pmfs[v]))
                    c_val = max(1, int(round(base_c * true_s)))
                    c_job[v] = c_val
                    ratios.append(c_val / max(node_mu[v], 1.0))

                ring_buf[rb_head] = float(np.mean(ratios))
                rb_head = (rb_head + 1) % 20

                # Co-running tasks on 8 cores for 275% total normalized WCET utilization
                bg_tasks = []
                w_Gen = 0
                burst_factor = rng.lognormal(mean=0.0, sigma=0.45)
                target_bg_wcet = bg_wcet_load * burst_factor
                while w_Gen < target_bg_wcet:
                    w_b = int(rng.integers(30, 100))
                    if rng.random() < 0.98:
                        dur = max(2, int(round(rng.normal(w_b / 3.0, w_b / 6.0) * true_s)))
                    else:
                        dur = max(2, int(round(rng.normal(w_b, w_b / 30.0) * true_s)))
                    rel = int(rng.integers(0, int(deadline_bins * 0.85)))
                    bg_d = rel + int(rng.integers(dur, int(deadline_bins * 0.40) + dur))
                    bg_tasks.append((rel, bg_d, dur))
                    w_Gen += w_b

                ast_job, ft_job = simulate_8core_np_edf(G, c_job, subdeadlines, num_cores=8, bg_tasks=bg_tasks)
                res_job = evaluate_job_detectors(
                    topo, sink, ast_job, ft_job, deadline_bins, f_miss, tecs_thresh, s_est, all_methods
                )

                for m, (miss, alarmed, lead) in res_job.items():
                    if miss and alarmed:
                        counts[m]["tp"] += 1
                        counts[m]["leads"].append(lead)
                    elif miss and not alarmed:
                        counts[m]["fn"] += 1
                    elif not miss and alarmed:
                        counts[m]["fp"] += 1
                    else:
                        counts[m]["tn"] += 1

            for m in all_methods:
                c = counts[m]
                pr = c["tp"] / (c["tp"] + c["fp"]) if (c["tp"] + c["fp"]) > 0 else 0.0
                rc = c["tp"] / (c["tp"] + c["fn"]) if (c["tp"] + c["fn"]) > 0 else 1.0
                f1 = 2 * pr * rc / (pr + rc) if (pr + rc) > 0 else 0.0
                fpr = c["fp"] / (c["fp"] + c["tn"]) if (c["fp"] + c["tn"]) > 0 else 0.0
                lead_m = float(np.mean(c["leads"])) if c["leads"] else 0.0
                stats_per_dag[m]["prec"].append(pr)
                stats_per_dag[m]["rec"].append(rc)
                stats_per_dag[m]["f1"].append(f1)
                stats_per_dag[m]["fpr"].append(fpr)
                stats_per_dag[m]["lead"].append(lead_m)

        print(f"\n=== SCENARIO: {scen_name} ===")
        print(f"{'Detector':<30} | {'Prec':<12} | {'Recall':<12} | {'F1':<12} | {'FPR':<12} | {'Lead (bins)'}")
        print("-" * 98)
        for m in all_methods:
            s = stats_per_dag[m]
            prec_m, prec_ci = np.mean(s['prec']), 1.96 * np.std(s['prec'], ddof=1) / np.sqrt(len(dags))
            rec_m, rec_ci = np.mean(s['rec']), 1.96 * np.std(s['rec'], ddof=1) / np.sqrt(len(dags))
            f1_m, f1_ci = np.mean(s['f1']), 1.96 * np.std(s['f1'], ddof=1) / np.sqrt(len(dags))
            fpr_m, fpr_ci = np.mean(s['fpr']), 1.96 * np.std(s['fpr'], ddof=1) / np.sqrt(len(dags))
            lead_val = np.mean(s['lead'])
            print(
                f"{m:<30} | {prec_m:.3f}+/-{prec_ci:.3f} "
                f"| {rec_m:.3f}+/-{rec_ci:.3f} "
                f"| {f1_m:.3f}+/-{f1_ci:.3f} "
                f"| {fpr_m:.3f}+/-{fpr_ci:.3f} "
                f"| {lead_val:.1f}"
            )
            all_results_rows.append({
                "Scenario": scen_name,
                "Detector": m,
                "Precision": round(prec_m, 4), "Precision_CI95": round(prec_ci, 4),
                "Recall": round(rec_m, 4), "Recall_CI95": round(rec_ci, 4),
                "F1_Score": round(f1_m, 4), "F1_CI95": round(f1_ci, 4),
                "FPR": round(fpr_m, 4), "FPR_CI95": round(fpr_ci, 4),
                "Lead_Time_bins": round(lead_val, 2)
            })

        # Paired t-tests (Ours vs TECS 0.8-1.0) across the 30 DAGs
        print("\n  Paired t-tests (Ours vs TECS 0.8-1.0):")
        for metric in ["prec", "rec", "f1", "fpr", "lead"]:
            a = np.array(stats_per_dag["TECS + AST' Rescaled [Ours]"][metric])
            b = np.array(stats_per_dag["TECS 0.8-1.0 (time)"][metric])
            if np.allclose(a, b):
                t_val, p_val = 0.0, 1.0
            else:
                t_val, p_val = stats.ttest_rel(a, b)
            print(f"    - {metric.upper():<5}: diff = {np.mean(a - b):+.4f}, t = {t_val:+.3f}, p = {p_val:.4e}")

    pd.DataFrame(all_results_rows).to_csv("exact_specification_results.csv", index=False)
    print("\nExported 'exact_specification_results.csv' successfully.")

if __name__ == "__main__":
    main()
