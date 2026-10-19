#!/usr/bin/env python3
"""
Interactive Web Dashboard & Visualizer for Real-Time DAG Scheduling & Early Deadline Miss Detection
Features Dual Novelty Framework:
- Idea 3: Context-Coupled Plaxity (Correlated Environmental Scene Modeling & Bayesian Residual Updates)
- Idea 1: Predict-and-Mitigate (Active RTOS Dynamic Slack Reclamation, DVFS & Emergency Priority Escalation)
Baseline: Toba, Baruah, & Azumi (ECRTS 2024 / ACM TECS 2026)
"""

import os
import sys
import glob
import json
import heapq
import urllib.parse
import webbrowser
from http.server import HTTPServer, BaseHTTPRequestHandler
import networkx as nx
import numpy as np
import yaml
from scipy import stats

# ----------------------------------------------------------------------------
# 1. CORE ALGORITHMS (TOBA ET AL. ACM TECS 2026 VERBATIM)
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
    ones = {v: 1.0 for v in topo}
    th_time = eq15_18(G, topo, mu, rem_t, th_min=0.8, th_max=1.0)
    th_job = eq15_18(G, topo, ones, rem_j, th_min=0.8, th_max=1.0)

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
            alpha[v] = 0.85     # 85% CPU-bound
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
        "th_time": th_time, "th_job": th_job, "levels": levels,
        "rem_t": rem_t, "cp_wcet": cp[sink],
        "rho": rho, "alpha": alpha, "rem_sens": rem_sens
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
# 2. INSTRUMENTED CLOSED-LOOP SIMULATOR (WITH GANTT TRACES)
# ----------------------------------------------------------------------------
def calculate_amdahl_duration(nom_dur, alpha_v, dvfs_ratio=1.33):
    scaled = nom_dur * (alpha_v / dvfs_ratio + (1.0 - alpha_v))
    return max(1, int(round(scaled)))

def simulate_closed_loop_job(ctx, c_job_nominal, D, num_cores, bg_tasks,
                             detector_mode="tecs_time",
                             rho_coupling=0.0,
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
    ast, ft = {}, {}
    gantt = []

    # Online Bayesian Conjugate State: xi ~ N(mu_0, sigma_0^2)
    sigma_0 = 1.0
    accum_prec = 1.0 / (sigma_0 ** 2)
    accum_num = 0.0

    alarmed = [False]
    t_alarm = [None]
    alarm_node = [-1]
    alarm_core = [-1]
    mitigation_active = [False]

    def get_current_posterior():
        if accum_prec <= 1.0 / (sigma_0 ** 2):
            return 0.0, sigma_0
        p_var = 1.0 / accum_prec
        return accum_num * p_var, float(np.sqrt(p_var))

    def trigger_mitigation(t):
        if not mitigation_active[0]:
            mitigation_active[0] = True
            new_ready = []
            for it in ready:
                prio_key, s_seq, is_dag, vid, dur = it
                if is_dag:
                    # Emergency priority escalation (-1) & Amdahl DVFS
                    dur_boost = calculate_amdahl_duration(dur, ctx["alpha"][vid], dvfs_ratio)
                    new_ready.append((-1, s_seq, is_dag, vid, dur_boost))
                else:
                    new_ready.append(it)
            heapq.heapify(new_ready)
            ready.clear()
            ready.extend(new_ready)

    def check_detection_and_dispatch(t):
        if not alarmed[0]:
            hat_xi, hat_xi_std = get_current_posterior()
            
            for item in ready:
                if item[2]:  # is_dag
                    vid = item[3]
                    threshold = th_vec[vid]

                    if detector_mode == "tecs_time":
                        t_eff = min(t, D)
                        miss_prob = ctx["Fm"][topo_idx[vid], t_eff]
                    elif detector_mode in ("context_coupled", "mitigate"):
                        # Idea 3: Conjugate Bayesian Shift Theorem
                        delta_rem = max(0.0, hat_xi * ctx["rem_sens"][vid])
                        t_eff = int(np.clip(round(t + delta_rem), 0, D))
                        miss_prob = ctx["Fm"][topo_idx[vid], t_eff]
                    elif detector_mode == "tecs_job":
                        t_eff = min(t, D)
                        threshold = ctx["th_job"][vid]
                        miss_prob = ctx["Fm"][topo_idx[vid], t_eff]
                    elif detector_mode == "ecrts_095":
                        t_eff = min(t, D)
                        threshold = 0.95
                        miss_prob = ctx["Fm"][topo_idx[vid], t_eff]
                    else:
                        t_eff = min(t, D)
                        threshold = 1.00
                        miss_prob = ctx["Fm"][topo_idx[vid], t_eff]

                    if miss_prob > (1.0 - threshold):
                        alarmed[0] = True
                        t_alarm[0] = t
                        alarm_node[0] = vid

                        if enable_mitigation:
                            trigger_mitigation(t)
                        break

        # Dispatch available cores to ready tasks (Non-preemptive NP-EDF)
        while free and ready:
            _, _, is_dag, vid, dur = heapq.heappop(ready)
            core = free.pop()
            if is_dag:
                ast[vid] = t
                if alarmed[0] and alarm_node[0] == vid and alarm_core[0] == -1:
                    alarm_core[0] = core

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
                    # Update Conjugate Bayesian Residual
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
    final_hat_xi, final_hat_std = get_current_posterior()

    return {
        "actual_miss": actual_miss,
        "alarmed": alarmed[0],
        "alarm_node": int(alarm_node[0]) if alarmed[0] else None,
        "alarm_core": int(alarm_core[0]) if alarmed[0] else None,
        "t_alarm": int(t_alarm[0]) if (alarmed[0] and t_alarm[0] is not None) else None,
        "exit_time": int(exit_time),
        "deadline": int(D),
        "lead_time": max(0, D - t_alarm[0]) if (alarmed[0] and t_alarm[0] is not None) else 0,
        "mitigated": mitigation_active[0],
        "final_hat_xi": float(final_hat_xi),
        "final_hat_std": float(final_hat_std),
        "gantt": gantt
    }

def run_simulation_suite(ctx, detector_type, intensity=1.0, n_jobs=30, rho_coupling=0.0, enable_mitigation=False, seed=42):
    rng = np.random.default_rng(seed)
    topo, D = ctx["topo"], ctx["D"]

    job_records = []
    tp_count, fp_count, tn_count, fn_count = 0, 0, 0, 0
    salvaged_count = 0
    impending_miss_count = 0
    lead_times = []

    for job_id in range(n_jobs):
        # Sample scene context factor xi
        if rho_coupling > 0.0:
            xi = rng.normal(1.2, 0.4) if rng.random() < 0.40 else rng.normal(0.0, 0.3)
        else:
            xi = 0.0

        c_job_nom = {}
        for v in topo:
            cdf = ctx["cdf"][v]
            base = min(int(np.searchsorted(cdf, rng.random(), side="right")), len(cdf) - 1)
            mult = 1.0 + (ctx["rho"][v] * xi if rho_coupling > 0 else 0.0)
            c_job_nom[v] = max(1, int(round(base * mult)))

        bg_tasks = make_background(rng, D, intensity)

        # Counterfactual: would it miss without mitigation?
        sim_unmit = simulate_closed_loop_job(ctx, c_job_nom, D, num_cores=8, bg_tasks=bg_tasks,
                                             detector_mode=detector_type, rho_coupling=rho_coupling,
                                             enable_mitigation=False)
        would_miss = sim_unmit["actual_miss"]

        # Actual run
        sim = simulate_closed_loop_job(ctx, c_job_nom, D, num_cores=8, bg_tasks=bg_tasks,
                                       detector_mode=detector_type, rho_coupling=rho_coupling,
                                       enable_mitigation=enable_mitigation, dvfs_ratio=1.33)

        actual_miss = sim["actual_miss"]
        alarmed = sim["alarmed"]

        is_salvaged = False
        if would_miss:
            impending_miss_count += 1
            if enable_mitigation and not actual_miss:
                salvaged_count += 1
                is_salvaged = True

        if would_miss and alarmed:
            tp_count += 1
            lead = sim["lead_time"]
            lead_times.append(lead)
            status = "TP (Salvaged!)" if is_salvaged else "TP (Accurate Alarm)"
        elif not would_miss and alarmed:
            fp_count += 1
            lead = 0
            status = "FP (False Alarm)"
        elif not would_miss and not alarmed:
            tn_count += 1
            lead = 0
            status = "TN (Safe On-Time)"
        else:
            fn_count += 1
            lead = 0
            status = "FN (Undetected Miss!)"

        sim["job_id"] = job_id + 1
        sim["would_miss"] = would_miss
        sim["is_salvaged"] = is_salvaged
        sim["status"] = status
        job_records.append(sim)

    total = n_jobs
    miss_rate = (impending_miss_count / total) * 100
    precision = (tp_count / (tp_count + fp_count) * 100) if (tp_count + fp_count) > 0 else 0.0
    recall = (tp_count / (tp_count + fn_count) * 100) if (tp_count + fn_count) > 0 else 100.0
    fpr = (fp_count / (tn_count + fp_count) * 100) if (tn_count + fp_count) > 0 else 0.0
    avg_lead = float(np.mean(lead_times)) if lead_times else 0.0
    dsr = (salvaged_count / impending_miss_count * 100) if impending_miss_count > 0 else 0.0

    return {
        "summary": {
            "total_jobs": total,
            "miss_rate": round(miss_rate, 1),
            "precision": round(precision, 1),
            "recall": round(recall, 1),
            "fpr": round(fpr, 1),
            "avg_lead": round(avg_lead, 1),
            "dsr": round(dsr, 1),
            "salvaged_count": salvaged_count,
            "impending_miss_count": impending_miss_count,
            "tp": tp_count, "fp": fp_count, "tn": tn_count, "fn": fn_count,
            "deadline": D
        },
        "jobs": job_records
    }

# ----------------------------------------------------------------------------
# 3. EMBEDDED DASHBOARD FRONTEND HTML / JS
# ----------------------------------------------------------------------------
DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Autonomous Vehicle RTOS Simulator</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <style>
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
    .custom-scroll::-webkit-scrollbar { width: 6px; height: 6px; }
    .custom-scroll::-webkit-scrollbar-thumb { background: #334155; border-radius: 4px; }
  </style>
</head>
<body class="bg-slate-950 text-slate-100 min-h-screen flex flex-col">

  <!-- TOP NAVIGATION HEADER -->
  <header class="bg-slate-900/90 border-b border-slate-800 px-6 py-4 flex items-center justify-between sticky top-0 z-50 backdrop-blur">
    <div class="flex items-center space-x-3">
      <div class="w-8 h-8 rounded-lg bg-indigo-600 flex items-center justify-center font-bold text-white shadow-lg shadow-indigo-500/30">
        ⚡
      </div>
      <div>
        <h1 class="text-base font-bold tracking-tight text-white flex items-center gap-2">
          Autonomous Vehicle RTOS Simulator
          <span class="text-xs font-medium px-2 py-0.5 rounded-full bg-emerald-950 text-emerald-400 border border-emerald-800">
            Idea 3 + Idea 1 Closed-Loop
          </span>
        </h1>
        <p class="text-xs text-slate-400">8-Core NP-EDF Scheduler & Correlated Plaxity with Active Slack Reclamation</p>
      </div>
    </div>
    <div class="flex items-center space-x-4">
      <span class="inline-flex items-center gap-1.5 text-xs px-2.5 py-1 rounded-full bg-emerald-950 text-emerald-400 border border-emerald-800">
        <span class="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
        DES Engine Online
      </span>
    </div>
  </header>

  <!-- MAIN WORKSPACE -->
  <div class="flex-1 flex overflow-hidden">
    
    <!-- LEFT SIDEBAR: EXPERIMENT CONTROLS (COMPACT WIDTH FOR EXPANDED GANTT) -->
    <aside class="w-64 bg-slate-900 border-r border-slate-800 p-3.5 flex flex-col justify-between overflow-y-auto custom-scroll text-xs">
      <div class="space-y-3">
        <div>
          <h2 class="text-[11px] font-semibold uppercase tracking-wider text-slate-400 mb-2">Simulation Parameters</h2>
          
          <!-- DAG SELECTOR -->
          <div class="space-y-1">
            <label class="text-[11px] font-medium text-slate-300">Target DAG Topology</label>
            <select id="dag-select" class="w-full bg-slate-950 border border-slate-800 text-xs rounded-lg px-2.5 py-1.5 text-slate-200 focus:outline-none focus:border-indigo-500">
              <option value="">Loading DAGs...</option>
            </select>
          </div>

          <!-- DETECTOR SELECTOR -->
          <div class="space-y-1 mt-2.5">
            <label class="text-[11px] font-medium text-slate-300">Framework / Algorithm</label>
            <select id="detector-select" class="w-full bg-slate-950 border border-slate-800 text-xs rounded-lg px-2.5 py-1.5 text-slate-200 focus:outline-none focus:border-indigo-500">
              <option value="mitigate" selected>✨ Predict & Mitigate (Idea 3+1)</option>
              <option value="context_coupled">💡 Context-Coupled (Idea 3)</option>
              <option value="tecs_time">📄 TECS-Time (Toba Baseline)</option>
              <option value="tecs_job">📄 TECS-Job (Job Weighted)</option>
              <option value="ecrts_095">📄 0.95 Flat Threshold</option>
            </select>
          </div>

          <!-- CORRELATION / SCENE CLUTTER SLIDER -->
          <div class="space-y-1 mt-3 p-2.5 rounded-lg bg-indigo-950/30 border border-indigo-900/40">
            <div class="flex justify-between text-[11px]">
              <label class="font-semibold text-indigo-300">Scene Coupling</label>
              <span id="rho-val" class="font-mono text-indigo-400 font-bold">0.35x</span>
            </div>
            <input type="range" id="rho-slider" min="0.0" max="0.5" step="0.05" value="0.35" class="w-full accent-indigo-500 h-1 bg-slate-800 rounded-lg cursor-pointer">
            <div class="flex justify-between text-[9px] text-slate-400 font-mono">
              <span>0.0 (Nominal)</span>
              <span>0.35 (Heavy Clutter)</span>
            </div>
          </div>

          <!-- ACTIVE MITIGATION TOGGLE -->
          <div class="space-y-1.5 mt-2.5 p-2.5 rounded-lg bg-emerald-950/30 border border-emerald-900/40">
            <div class="flex items-center justify-between">
              <label class="text-[11px] font-semibold text-emerald-300">
                RTOS Dynamic Mitigation
              </label>
              <input type="checkbox" id="mit-check" checked class="w-3.5 h-3.5 rounded text-emerald-600 bg-slate-800 border-slate-700 accent-emerald-500 cursor-pointer">
            </div>
            <p class="text-[10px] text-slate-400 leading-tight">
              DVFS core boost + Emergency priority escalation upon alarm.
            </p>
          </div>

          <!-- INTENSITY SLIDER -->
          <div class="space-y-1 mt-3">
            <div class="flex justify-between text-[11px]">
              <label class="font-medium text-slate-300">Contention Load</label>
              <span id="intensity-val" class="font-mono text-indigo-400">1.00x</span>
            </div>
            <input type="range" id="intensity-slider" min="0.25" max="1.50" step="0.25" value="1.00" class="w-full accent-indigo-500 h-1 bg-slate-800 rounded-lg cursor-pointer">
          </div>

          <!-- JOBS SLIDER -->
          <div class="space-y-1 mt-2.5">
            <div class="flex justify-between text-[11px]">
              <label class="font-medium text-slate-300">Sample Stream Size</label>
              <span id="jobs-val" class="font-mono text-indigo-400">40 Jobs</span>
            </div>
            <input type="range" id="jobs-slider" min="10" max="100" step="10" value="40" class="w-full accent-indigo-500 h-1 bg-slate-800 rounded-lg cursor-pointer">
          </div>

          <!-- RANDOMIZE RUN SEEDS TOGGLE -->
          <div class="flex items-center justify-between mt-2.5 pt-2 border-t border-slate-800">
            <label class="text-[10px] text-slate-400 flex items-center gap-1.5 cursor-pointer">
              <input type="checkbox" id="random-seed-check" class="rounded text-indigo-500 accent-indigo-500 cursor-pointer">
              Randomize Run Seed
            </label>
            <span id="seed-badge" class="font-mono text-[9px] text-slate-500">Seed: 42</span>
          </div>
        </div>

        <!-- RUN BUTTON -->
        <button id="run-btn" class="w-full bg-indigo-600 hover:bg-indigo-500 text-white font-medium text-xs py-2 px-3 rounded-lg shadow-md shadow-indigo-600/30 transition flex items-center justify-center gap-2">
          <svg id="run-spinner" class="hidden animate-spin h-3.5 w-3.5 text-white" fill="none" viewBox="0 0 24 24">
            <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
            <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z"></path>
          </svg>
          <span id="run-text">Run 8-Core Simulation</span>
        </button>
      </div>

      <!-- FOOTER -->
      <div class="text-[10px] text-slate-600 font-mono pt-2 border-t border-slate-800/80">
        Engine: 8-Core NP-EDF DES<br>
        20-Node Task Topologies
      </div>
    </aside>

    <!-- RIGHT CONTENT: DASHBOARD & VISUALIZATIONS -->
    <main class="flex-1 flex flex-col overflow-y-auto custom-scroll p-6 space-y-6">

      <!-- METRICS HUD -->
      <div class="grid grid-cols-5 gap-4">
        <!-- Salvage Rate Card (Idea 1 Focus!) -->
        <div class="bg-slate-900/90 border border-emerald-800/80 rounded-xl p-4 flex flex-col justify-between shadow-lg shadow-emerald-950/40">
          <span class="text-xs font-semibold text-emerald-400 uppercase tracking-wide">Deadline Salvage (DSR)</span>
          <div class="mt-2 flex items-baseline gap-2">
            <span id="m-dsr" class="text-2xl font-bold font-mono text-emerald-300">--</span>
            <span id="m-dsr-sub" class="text-xs text-slate-400">averted stops</span>
          </div>
          <div class="w-full bg-slate-800 h-1.5 rounded-full mt-3 overflow-hidden">
            <div id="bar-dsr" class="bg-emerald-400 h-full w-0 transition-all duration-500"></div>
          </div>
        </div>

        <!-- Miss Rate Card -->
        <div class="bg-slate-900/80 border border-slate-800 rounded-xl p-4 flex flex-col justify-between">
          <span class="text-xs text-slate-400">Impending Miss Rate</span>
          <div class="mt-2 flex items-baseline gap-2">
            <span id="m-miss" class="text-2xl font-bold font-mono text-white">--</span>
            <span class="text-xs text-slate-500">without mit</span>
          </div>
          <div class="w-full bg-slate-800 h-1.5 rounded-full mt-3 overflow-hidden">
            <div id="bar-miss" class="bg-rose-500 h-full w-0 transition-all duration-500"></div>
          </div>
        </div>

        <!-- Recall Card -->
        <div class="bg-slate-900/80 border border-slate-800 rounded-xl p-4 flex flex-col justify-between">
          <span class="text-xs text-slate-400">Detection Recall</span>
          <div class="mt-2 flex items-baseline gap-2">
            <span id="m-recall" class="text-2xl font-bold font-mono text-emerald-400">--</span>
            <span class="text-xs text-slate-500">safety margin</span>
          </div>
          <div class="w-full bg-slate-800 h-1.5 rounded-full mt-3 overflow-hidden">
            <div id="bar-recall" class="bg-emerald-500 h-full w-0 transition-all duration-500"></div>
          </div>
        </div>

        <!-- Precision Card -->
        <div class="bg-slate-900/80 border border-slate-800 rounded-xl p-4 flex flex-col justify-between">
          <span class="text-xs text-slate-400">Alarm Precision</span>
          <div class="mt-2 flex items-baseline gap-2">
            <span id="m-prec" class="text-2xl font-bold font-mono text-indigo-400">--</span>
            <span class="text-xs text-slate-500">confidence</span>
          </div>
          <div class="w-full bg-slate-800 h-1.5 rounded-full mt-3 overflow-hidden">
            <div id="bar-prec" class="bg-indigo-500 h-full w-0 transition-all duration-500"></div>
          </div>
        </div>

        <!-- Lead Time Card -->
        <div class="bg-slate-900/80 border border-slate-800 rounded-xl p-4 flex flex-col justify-between">
          <span class="text-xs text-slate-400">Warning Lead Time</span>
          <div class="mt-2 flex items-baseline gap-2">
            <span id="m-lead" class="text-2xl font-bold font-mono text-amber-400">--</span>
            <span class="text-xs text-slate-500">ms advance</span>
          </div>
          <div class="w-full bg-slate-800 h-1.5 rounded-full mt-3 overflow-hidden">
            <div id="bar-lead" class="bg-amber-500 h-full w-0 transition-all duration-500"></div>
          </div>
        </div>
      </div>

      <!-- INTERACTIVE 8-CORE GANTT TIMELINE -->
      <div class="bg-slate-900/80 border border-slate-800 rounded-xl p-5 flex flex-col">
        <div class="flex items-center justify-between mb-3">
          <div>
            <h3 class="text-sm font-semibold text-white flex items-center gap-2">
              8-Core Dispatch Timeline & Active Mitigation
              <span id="gantt-badge" class="text-xs font-mono px-2 py-0.5 rounded bg-slate-800 text-slate-300">Job #1</span>
            </h3>
            <p class="text-xs text-slate-400 mt-0.5">Microsecond Gantt trace showing non-preemptive EDF core occupancy vs deadline</p>
          </div>
          <div class="flex items-center space-x-2">
            <label class="text-xs text-slate-400">Inspect Run:</label>
            <select id="job-picker" class="bg-slate-950 border border-slate-800 text-xs rounded-lg px-2.5 py-1 text-slate-200 focus:outline-none"></select>
          </div>
        </div>

        <!-- SALVAGE BANNER (DYNAMIC) -->
        <div id="salvage-banner" class="hidden mb-3 p-3 rounded-lg bg-emerald-950/70 border border-emerald-800 flex items-center justify-between text-xs text-emerald-200">
          <span class="font-bold flex items-center gap-2">
            🎉 DEADLINE SALVAGED BY DYNAMIC MITIGATION!
          </span>
          <span class="text-[11px] text-emerald-300 font-mono">
            Downstream tasks accelerated via DVFS & Emergency Priority Escalation
          </span>
        </div>

        <!-- GANTT CANVAS CONTAINER -->
        <div class="relative w-full overflow-x-auto custom-scroll bg-slate-950/70 rounded-lg p-3 border border-slate-800/80 min-h-[260px]">
          <canvas id="gantt-canvas" height="260" class="w-full"></canvas>
        </div>

        <!-- GANTT LEGEND -->
        <div class="flex items-center justify-between text-xs text-slate-400 mt-3 pt-2 border-t border-slate-800">
          <div class="flex items-center space-x-4">
            <span class="flex items-center gap-1.5"><span class="w-3 h-3 rounded bg-indigo-500"></span> DAG Tasks</span>
            <span class="flex items-center gap-1.5"><span class="w-3 h-3 rounded bg-slate-700"></span> Background Load</span>
            <span class="flex items-center gap-1.5"><span class="w-3 h-0.5 bg-rose-500 border-dashed"></span> Deadline (D)</span>
            <span class="flex items-center gap-1.5"><span class="w-3 h-3 rounded-full bg-rose-500 border-2 border-white"></span> Alarm Beacon</span>
            <span class="flex items-center gap-1.5"><span class="w-3 h-3 rounded bg-emerald-500"></span> Mitigated Slack</span>
          </div>
          <div id="gantt-lead-info" class="text-xs font-mono text-slate-300"></div>
        </div>
      </div>

      <!-- DAG TOPOLOGY VISUALIZATION -->
      <div class="bg-slate-900/80 border border-slate-800 rounded-xl p-5 flex flex-col">
        <div class="flex items-center justify-between mb-3">
          <div>
            <h3 class="text-sm font-semibold text-white">Autonomous Vehicle Task Graph (20 Nodes)</h3>
            <p class="text-xs text-slate-400 mt-0.5">Directed Acyclic Graph layout: Sensors → Perception → Sensor Fusion → Motion Planner → Actuation Gate</p>
          </div>
          <span id="dag-info-badge" class="text-xs font-mono text-slate-400"></span>
        </div>
        
        <!-- GRAPH CANVAS -->
        <div class="relative w-full bg-slate-950/70 rounded-lg border border-slate-800/80 p-2 min-h-[220px]">
          <canvas id="dag-canvas" height="220" class="w-full"></canvas>
        </div>
      </div>

    </main>
  </div>

  <script>
    let currentDags = [];
    let currentSimData = null;
    let currentDagGraph = null;

    async function init() {
      try {
        const res = await fetch('/api/list_dags');
        const data = await res.json();
        currentDags = data.dags;
        const select = document.getElementById('dag-select');
        select.innerHTML = '';
        currentDags.forEach((d, i) => {
          const opt = document.createElement('option');
          opt.value = d.path;
          opt.textContent = `${d.name} (${d.nodes} nodes, WCET: ${d.wcet}ms)`;
          if (i === 0) opt.selected = true;
          select.appendChild(opt);
        });
        
        await loadDagGraph(select.value);
        runSimulation();
      } catch (e) {
        console.error(e);
      }
    }

    // Sliders
    document.getElementById('rho-slider').addEventListener('input', (e) => {
      document.getElementById('rho-val').textContent = parseFloat(e.target.value).toFixed(2) + 'x';
    });
    document.getElementById('intensity-slider').addEventListener('input', (e) => {
      document.getElementById('intensity-val').textContent = parseFloat(e.target.value).toFixed(2) + 'x';
    });
    document.getElementById('jobs-slider').addEventListener('input', (e) => {
      document.getElementById('jobs-val').textContent = e.target.value + ' Jobs';
    });
    document.getElementById('dag-select').addEventListener('change', async (e) => {
      await loadDagGraph(e.target.value);
      runSimulation();
    });

    async function loadDagGraph(path) {
      try {
        const res = await fetch('/api/dag_graph?path=' + encodeURIComponent(path));
        currentDagGraph = await res.json();
        document.getElementById('dag-info-badge').textContent = `Nodes: ${currentDagGraph.nodes.length} | Edges: ${currentDagGraph.edges.length} | Deadline: ${currentDagGraph.deadline}ms`;
        renderDagGraph(currentDagGraph);
      } catch (e) {
        console.error(e);
      }
    }

    async function runSimulation() {
      const btn = document.getElementById('run-btn');
      const spinner = document.getElementById('run-spinner');
      const runText = document.getElementById('run-text');
      btn.disabled = true;
      spinner.classList.remove('hidden');
      runText.textContent = 'Simulating 8-Core Closed Loop...';

      const det = document.getElementById('detector-select').value;
      const isMit = document.getElementById('mit-check').checked || (det === 'mitigate');
      const isRandom = document.getElementById('random-seed-check').checked;
      const seedVal = isRandom ? Math.floor(Math.random() * 100000) : 42;
      document.getElementById('seed-badge').textContent = isRandom ? `Seed: ${seedVal}` : 'Seed: 42';

      const payload = {
        dag_path: document.getElementById('dag-select').value,
        detector: det,
        intensity: parseFloat(document.getElementById('intensity-slider').value),
        rho_coupling: parseFloat(document.getElementById('rho-slider').value),
        enable_mitigation: isMit,
        n_jobs: parseInt(document.getElementById('jobs-slider').value),
        seed: seedVal
      };

      try {
        const res = await fetch('/api/simulate', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });
        currentSimData = await res.json();
        updateDashboard(currentSimData);
      } catch (e) {
        console.error(e);
      } finally {
        btn.disabled = false;
        spinner.classList.add('hidden');
        runText.textContent = 'Execute Closed-Loop Simulation';
      }
    }

    document.getElementById('run-btn').addEventListener('click', runSimulation);

    function updateDashboard(sim) {
      const s = sim.summary;
      document.getElementById('m-dsr').textContent = s.dsr + '%';
      document.getElementById('m-dsr-sub').textContent = `${s.salvaged_count}/${s.impending_miss_count} averted`;
      document.getElementById('m-miss').textContent = s.miss_rate + '%';
      document.getElementById('m-recall').textContent = s.recall + '%';
      document.getElementById('m-prec').textContent = s.precision + '%';
      document.getElementById('m-lead').textContent = s.avg_lead + ' ms';

      document.getElementById('bar-dsr').style.width = Math.min(100, s.dsr) + '%';
      document.getElementById('bar-miss').style.width = Math.min(100, s.miss_rate) + '%';
      document.getElementById('bar-recall').style.width = Math.min(100, s.recall) + '%';
      document.getElementById('bar-prec').style.width = Math.min(100, s.precision) + '%';
      document.getElementById('bar-lead').style.width = Math.min(100, (s.avg_lead / (s.deadline || 100)) * 100) + '%';

      const picker = document.getElementById('job-picker');
      picker.innerHTML = '';
      sim.jobs.forEach(j => {
        const opt = document.createElement('option');
        opt.value = j.job_id - 1;
        let icon = '✅ Met';
        if (j.is_salvaged) icon = '🎉 SALVAGED!';
        else if (j.actual_miss) icon = j.alarmed ? '🚨 Miss (Alarmed)' : '❌ Miss (Silent!)';
        else if (j.alarmed) icon = '⚠️ False Alarm';

        opt.textContent = `Job #${j.job_id}: ${icon}`;
        picker.appendChild(opt);
      });

      // Default select first salvaged or first missed job
      const salvagedIdx = sim.jobs.findIndex(j => j.is_salvaged);
      const missedIdx = sim.jobs.findIndex(j => j.actual_miss);
      const chosenIdx = salvagedIdx !== -1 ? salvagedIdx : (missedIdx !== -1 ? missedIdx : 0);
      picker.value = chosenIdx;
      renderGantt(sim.jobs[chosenIdx], s.deadline);
    }

    document.getElementById('job-picker').addEventListener('change', (e) => {
      const idx = parseInt(e.target.value);
      if (currentSimData && currentSimData.jobs[idx]) {
        renderGantt(currentSimData.jobs[idx], currentSimData.summary.deadline);
      }
    });

    function renderGantt(job, deadline) {
      const canvas = document.getElementById('gantt-canvas');
      const ctx = canvas.getContext('2d');
      const dpr = window.devicePixelRatio || 1;
      const rect = canvas.getBoundingClientRect();
      canvas.width = rect.width * dpr;
      canvas.height = 260 * dpr;
      ctx.scale(dpr, dpr);

      const w = rect.width;
      const h = 260;
      ctx.clearRect(0, 0, w, h);

      const margin = { left: 65, right: 30, top: 25, bottom: 30 };
      const chartW = w - margin.left - margin.right;
      const chartH = h - margin.top - margin.bottom;

      const maxTime = Math.max(deadline * 1.15, job.exit_time * 1.05);
      const timeToX = (t) => margin.left + (t / maxTime) * chartW;
      const coreH = chartH / 8;

      // Salvage Banner Toggle
      const banner = document.getElementById('salvage-banner');
      if (job.is_salvaged) {
        banner.classList.remove('hidden');
      } else {
        banner.classList.add('hidden');
      }

      // Draw Core Rows
      for (let c = 0; c < 8; c++) {
        const y = margin.top + c * coreH;
        ctx.fillStyle = c % 2 === 0 ? 'rgba(30, 41, 59, 0.4)' : 'rgba(15, 23, 42, 0.4)';
        ctx.fillRect(margin.left, y, chartW, coreH);
        ctx.strokeStyle = 'rgba(51, 65, 85, 0.4)';
        ctx.strokeRect(margin.left, y, chartW, coreH);

        ctx.fillStyle = '#94a3b8';
        ctx.font = '11px monospace';
        ctx.textAlign = 'right';
        ctx.fillText(`Core ${c}`, margin.left - 10, y + coreH / 2 + 4);
      }

      // Draw Tasks
      job.gantt.forEach(t => {
        const x1 = timeToX(t.start);
        const x2 = timeToX(t.end);
        const taskW = Math.max(3, x2 - x1);
        const y = margin.top + t.core * coreH + 3;
        const taskH = coreH - 6;

        if (t.is_dag) {
          ctx.fillStyle = job.is_salvaged && (job.t_alarm !== null && t.start >= job.t_alarm) ? '#10b981' : '#6366f1';
          ctx.beginPath();
          ctx.roundRect(x1, y, taskW, taskH, 3);
          ctx.fill();

          ctx.fillStyle = '#ffffff';
          ctx.font = '10px sans-serif';
          ctx.textAlign = 'center';
          if (taskW > 25) {
            ctx.fillText(`v${t.vid}`, x1 + taskW / 2, y + taskH / 2 + 3);
          }
        } else {
          ctx.fillStyle = '#475569';
          ctx.beginPath();
          ctx.roundRect(x1, y, taskW, taskH, 2);
          ctx.fill();
        }
      });

      // Draw Deadline Line
      const deadX = timeToX(deadline);
      ctx.strokeStyle = '#ef4444';
      ctx.lineWidth = 2;
      ctx.setLineDash([5, 4]);
      ctx.beginPath();
      ctx.moveTo(deadX, margin.top - 10);
      ctx.lineTo(deadX, margin.top + chartH + 10);
      ctx.stroke();
      ctx.setLineDash([]);

      ctx.fillStyle = '#ef4444';
      ctx.font = 'bold 10px monospace';
      ctx.textAlign = 'center';
      ctx.fillText(`DEADLINE D (${deadline}ms)`, deadX, margin.top - 12);

      // Draw Alarm Beacon
      if (job.alarmed && job.t_alarm !== null) {
        const alarmX = timeToX(job.t_alarm);
        const alarmY = margin.top + (job.alarm_core >= 0 ? job.alarm_core * coreH + coreH / 2 : 10);

        // Bold vertical alarm line
        ctx.strokeStyle = '#f59e0b';
        ctx.lineWidth = 2.5;
        ctx.beginPath();
        ctx.moveTo(alarmX, margin.top);
        ctx.lineTo(alarmX, margin.top + chartH);
        ctx.stroke();

        // Pulsing outer beacon ring
        ctx.fillStyle = 'rgba(239, 68, 68, 0.35)';
        ctx.beginPath();
        ctx.arc(alarmX, alarmY, 11, 0, Math.PI * 2);
        ctx.fill();

        // Inner solid beacon dot
        ctx.fillStyle = '#ef4444';
        ctx.beginPath();
        ctx.arc(alarmX, alarmY, 6, 0, Math.PI * 2);
        ctx.fill();
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 2;
        ctx.stroke();

        // High-Visibility Floating Alarm Badge (Solid Red/Amber with Border)
        const tag = `⚠️ ALARM @ ${job.t_alarm}ms (Node ${job.alarm_node})`;
        ctx.font = 'bold 11px sans-serif';
        const tagW = ctx.measureText(tag).width + 18;
        const tagH = 24;

        // Position badge cleanly above or below beacon
        let tagX = Math.min(w - tagW - 10, Math.max(margin.left + 5, alarmX - tagW / 2));
        let tagY = alarmY - 32;
        if (tagY < margin.top) {
          tagY = alarmY + 12;
        }

        // Draw Badge Background with border
        ctx.fillStyle = '#7f1d1d';
        ctx.strokeStyle = '#fef08a';
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.roundRect(tagX, tagY, tagW, tagH, 6);
        ctx.fill();
        ctx.stroke();

        // Little arrow pointing towards beacon
        ctx.fillStyle = '#7f1d1d';
        ctx.beginPath();
        if (tagY < alarmY) {
          ctx.moveTo(alarmX - 5, tagY + tagH);
          ctx.lineTo(alarmX + 5, tagY + tagH);
          ctx.lineTo(alarmX, alarmY - 7);
        } else {
          ctx.moveTo(alarmX - 5, tagY);
          ctx.lineTo(alarmX + 5, tagY);
          ctx.lineTo(alarmX, alarmY + 7);
        }
        ctx.closePath();
        ctx.fill();

        // Bright Yellow Text inside Badge
        ctx.fillStyle = '#fef08a';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(tag, tagX + tagW / 2, tagY + tagH / 2);
        ctx.textBaseline = 'alphabetic'; // reset

        document.getElementById('gantt-lead-info').innerHTML = 
          `<span class="text-amber-400 font-bold">Alarm Fired:</span> ${job.t_alarm}ms | <span class="text-emerald-400 font-bold">Lead Time:</span> ${job.lead_time}ms ahead of deadline`;
      } else {
        document.getElementById('gantt-lead-info').innerHTML = 
          `<span class="text-slate-400 font-bold">Status:</span> ${job.status}`;
      }

      document.getElementById('gantt-badge').textContent = `Job #${job.job_id} [${job.status}]`;
    }

    function renderDagGraph(dag) {
      const canvas = document.getElementById('dag-canvas');
      const ctx = canvas.getContext('2d');
      const dpr = window.devicePixelRatio || 1;
      const rect = canvas.getBoundingClientRect();
      canvas.width = rect.width * dpr;
      canvas.height = 220 * dpr;
      ctx.scale(dpr, dpr);

      const w = rect.width;
      const h = 220;
      ctx.clearRect(0, 0, w, h);

      const numLevels = Math.max(...Object.values(dag.levels)) + 1;
      const levelNodes = {};
      for (let l = 0; l < numLevels; l++) levelNodes[l] = [];
      dag.nodes.forEach(n => {
        const lvl = dag.levels[n.id] || 0;
        levelNodes[lvl].push(n);
      });

      const nodeCoords = {};
      const paddingX = 40;
      const paddingY = 25;
      const colW = (w - 2 * paddingX) / Math.max(1, numLevels - 1);

      for (let l = 0; l < numLevels; l++) {
        const nodesInCol = levelNodes[l];
        const rowH = (h - 2 * paddingY) / Math.max(1, nodesInCol.length);
        nodesInCol.forEach((n, idx) => {
          const x = paddingX + l * colW;
          const y = paddingY + (idx + 0.5) * rowH;
          nodeCoords[n.id] = { x, y };
        });
      }

      // Draw Edges
      ctx.strokeStyle = 'rgba(99, 102, 241, 0.4)';
      ctx.lineWidth = 1.5;
      dag.edges.forEach(e => {
        const p1 = nodeCoords[e.source];
        const p2 = nodeCoords[e.target];
        if (p1 && p2) {
          ctx.beginPath();
          ctx.moveTo(p1.x, p1.y);
          const midX = (p1.x + p2.x) / 2;
          ctx.bezierCurveTo(midX, p1.y, midX, p2.y, p2.x, p2.y);
          ctx.stroke();
        }
      });

      // Draw Nodes
      dag.nodes.forEach(n => {
        const pt = nodeCoords[n.id];
        if (!pt) return;
        const isSink = n.id === dag.sink;
        const isEntry = (dag.levels[n.id] || 0) === 0;

        ctx.fillStyle = isSink ? '#ef4444' : (isEntry ? '#10b981' : '#6366f1');
        ctx.beginPath();
        ctx.arc(pt.x, pt.y, 11, 0, Math.PI * 2);
        ctx.fill();
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 1.5;
        ctx.stroke();

        ctx.fillStyle = '#ffffff';
        ctx.font = 'bold 9px monospace';
        ctx.textAlign = 'center';
        ctx.fillText(n.id, pt.x, pt.y + 3);
      });
    }

    window.addEventListener('resize', () => {
      if (currentSimData && currentSimData.jobs && currentSimData.jobs.length > 0) {
        const picker = document.getElementById('job-picker');
        const chosenIdx = parseInt(picker.value) || 0;
        renderGantt(currentSimData.jobs[chosenIdx], currentSimData.summary.deadline);
      }
      if (currentDagGraph) {
        renderDagGraph(currentDagGraph);
      }
    });

    window.addEventListener('DOMContentLoaded', init);
  </script>
</body>
</html>
"""

# ----------------------------------------------------------------------------
# 4. HTTP SERVER & REST API HANDLER
# ----------------------------------------------------------------------------
class DashboardHandler(BaseHTTPRequestHandler):
    dag_cache = {}

    def do_GET(self):
        url = urllib.parse.urlparse(self.path)
        path = url.path
        query = urllib.parse.parse_qs(url.query)

        if path == "/" or path == "/index.html":
            self.send_response(200)
            self.send_header("Content-type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(DASHBOARD_HTML.encode("utf-8"))
        
        elif path == "/api/list_dags":
            files = sorted(glob.glob("RD-Gen/DAGs/**/dag_*.yaml", recursive=True))
            dags = []
            for f in files[:25]:
                name = os.path.basename(f)
                folder = os.path.basename(os.path.dirname(f))
                dags.append({
                    "path": f.replace("\\", "/"),
                    "name": f"{folder} / {name}",
                    "nodes": 20,
                    "wcet": 240
                })
            self.send_json({"dags": dags})

        elif path == "/api/dag_graph":
            filepath = query.get("path", ["RD-Gen/DAGs/NEN_1_Transfer_1/dag_0.yaml"])[0]
            filepath = os.path.normpath(filepath)
            if filepath not in self.dag_cache:
                self.dag_cache[filepath] = build_dag_context(filepath)
            ctx = self.dag_cache[filepath]
            G = ctx["G"]
            nodes = [{"id": int(v), "wcet": ctx["wcet"][v], "threshold": ctx["th_time"][v]} for v in ctx["topo"]]
            edges = [{"source": int(u), "target": int(v)} for u, v in G.edges()]
            self.send_json({
                "nodes": nodes,
                "edges": edges,
                "sink": ctx["sink"],
                "deadline": ctx["D"],
                "levels": ctx["levels"]
            })

        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == "/api/simulate":
            try:
                length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(length).decode("utf-8")
                data = json.loads(body)

                filepath = data.get("dag_path", "RD-Gen/DAGs/NEN_1_Transfer_1/dag_0.yaml")
                filepath = os.path.normpath(filepath)
                detector = data.get("detector", "mitigate")
                intensity = float(data.get("intensity", 1.0))
                rho_coupling = float(data.get("rho_coupling", 0.35))
                enable_mitigation = bool(data.get("enable_mitigation", True))
                n_jobs = int(data.get("n_jobs", 30))
                seed = int(data.get("seed", 42))

                if filepath not in self.dag_cache:
                    self.dag_cache[filepath] = build_dag_context(filepath)
                ctx = self.dag_cache[filepath]

                results = run_simulation_suite(ctx, detector_type=detector, intensity=intensity,
                                               n_jobs=n_jobs, rho_coupling=rho_coupling,
                                               enable_mitigation=enable_mitigation, seed=seed)
                self.send_json(results)
            except Exception as e:
                self.send_response(500)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def send_json(self, data):
        self.send_response(200)
        self.send_header("Content-type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode("utf-8"))

    def log_message(self, format, *args):
        return

def main():
    port = 8050
    server_address = ("", port)
    try:
        httpd = HTTPServer(server_address, DashboardHandler)
    except OSError:
        port = 8051
        server_address = ("", port)
        httpd = HTTPServer(server_address, DashboardHandler)

    url = f"http://localhost:{port}"
    print(f"\n=======================================================")
    print(f"[+] Real-Time DAG Simulator Dashboard Running!")
    print(f"[+] Open in your browser: {url}")
    print(f"=======================================================\n")
    
    try:
        webbrowser.open(url)
    except Exception:
        pass

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping dashboard server...")
        httpd.server_close()

if __name__ == "__main__":
    main()
