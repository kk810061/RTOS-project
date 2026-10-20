#!/usr/bin/env python3
"""
Refined Phase 1 Validation:
Compare tail miss probabilities P(t + L_v > D) between:
1. Exact Monte Carlo under correlated context xi
2. Shifted Plaxity Lookup F_m(v, t_eff)
3. Nominal unadjusted Lookup F_m(v, t)
"""

import numpy as np
from scipy import stats
import yaml
import glob
import networkx as nx

def evaluate_tail_probabilities():
    dag_files = sorted(glob.glob("RD-Gen/DAGs/**/dag_*.yaml", recursive=True))
    if not dag_files:
        print("No DAG files found.")
        return
        
    with open(dag_files[0], "r") as f:
        data = yaml.safe_load(f)
        
    G = nx.DiGraph()
    for nd in data["nodes"]:
        wcet = max(12, int(nd.get("execution_time", 25)) * 3)
        G.add_node(int(nd["id"]), wcet=wcet)
    for e in data.get("links", data.get("edges", [])):
        G.add_edge(int(e["source"]), int(e["target"]))
        
    topo = list(nx.topological_sort(G))
    sink = [v for v in topo if G.out_degree(v) == 0][0]
    
    levels = {}
    for v in topo:
        preds = list(G.predecessors(v))
        levels[v] = 0 if not preds else max(levels[u] for u in preds) + 1
    max_lvl = max(levels.values()) if levels else 1
    
    # Sensitivity rho_v: perception ~ 0.8-0.9, planning ~ 0.5, control ~ 0.1-0.2
    rho = {v: float(np.clip(0.9 - 0.7 * (levels[v] / max_lvl), 0.1, 0.95)) for v in topo}
    mu = {v: G.nodes[v]["wcet"] / 3.0 for v in topo}
    sigma = {v: mu[v] / 2.0 for v in topo}
    
    # Total nominal critical path
    cp_mu = {}
    for v in topo:
        cp_mu[v] = max((cp_mu[u] for u in G.predecessors(v)), default=0.0) + mu[v]
    D = int(round(cp_mu[sink] * 2.2))  # realistic deadline
    
    target_node = topo[len(topo) // 3]
    desc = list(nx.descendants(G, target_node)) + [target_node]
    subG = G.subgraph(desc)
    
    test_xi = 1.0
    N_SAMPLES = 100000
    rng = np.random.default_rng(42)
    
    # 1. Exact Correlated Monte Carlo
    mc_remaining = np.zeros(N_SAMPLES)
    for s_idx in range(N_SAMPLES):
        durations = {v: max(1.0, rng.normal(mu[v], sigma[v])) * (1.0 + rho[v] * test_xi) for v in desc}
        dist = {v: 0.0 for v in desc}
        for u in nx.topological_sort(subG):
            d_u = dist[u] + durations[u]
            for succ in subG.successors(u):
                if d_u > dist[succ]:
                    dist[succ] = d_u
        mc_remaining[s_idx] = dist[sink]
        
    # 2. Nominal Monte Carlo (xi = 0)
    nom_remaining = np.zeros(N_SAMPLES)
    for s_idx in range(N_SAMPLES):
        durations = {v: max(1.0, rng.normal(mu[v], sigma[v])) for v in desc}
        dist = {v: 0.0 for v in desc}
        for u in nx.topological_sort(subG):
            d_u = dist[u] + durations[u]
            for succ in subG.successors(u):
                if d_u > dist[succ]:
                    dist[succ] = d_u
        nom_remaining[s_idx] = dist[sink]
        
    # Compute downstream path sensitivity delta
    nom_dist = {v: 0.0 for v in desc}
    for u in nx.topological_sort(subG):
        d_u = nom_dist[u] + mu[u]
        for succ in subG.successors(u):
            if d_u > nom_dist[succ]:
                nom_dist[succ] = d_u
                
    delta_mu_rem = 0.0
    curr = target_node
    while True:
        delta_mu_rem += rho[curr] * mu[curr] * test_xi
        succs = list(subG.successors(curr))
        if not succs:
            break
        curr = max(succs, key=lambda s: nom_dist[s])
        
    print(f"Target Node: {target_node}, Downstream Delta mu: {delta_mu_rem:.2f} ms")
    print(f"{'Time t':<10} | {'Exact P(Miss)':<15} | {'Shifted P(Miss)':<16} | {'Nominal P(Miss)':<16} | {'Shift Error':<12}")
    print("-" * 75)
    
    # Test across multiple time steps t
    t_vals = np.linspace(int(D * 0.2), int(D * 0.7), 10)
    max_err = 0.0
    for t in t_vals:
        t = int(t)
        # Exact probability that t + L_v > D
        p_exact = np.mean(t + mc_remaining > D)
        
        # Shifted estimate: probability that (t + delta) + L_v^(0) > D
        t_eff = t + delta_mu_rem
        p_shifted = np.mean(t_eff + nom_remaining > D)
        
        # Nominal unshifted estimate
        p_nom = np.mean(t + nom_remaining > D)
        
        err = abs(p_shifted - p_exact)
        if err > max_err:
            max_err = err
        print(f"{t:<10d} | {p_exact:<15.4f} | {p_shifted:<16.4f} | {p_nom:<16.4f} | {err:<12.4f}")
        
    print(f"\nMax Probability Approximation Error across critical window: {max_err:.4f}")
    if max_err < 0.05:
        print("[PASS] Shifted Plaxity approximates exact correlated tail probability with < 0.05 absolute error!")

if __name__ == "__main__":
    evaluate_tail_probabilities()
