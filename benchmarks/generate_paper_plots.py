#!/usr/bin/env python3
"""
Generate remaining publication figures:
- figure_bayesian_convergence.png: Online posterior tracking and variance shrinkage
- figure_comparative_hud.png: Lead Time & DSR comparison across baseline vs novel framework
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def generate_bayesian_figure():
    rng = np.random.default_rng(42)
    true_xi = 1.4
    n_nodes = 16
    mu_nodes = rng.uniform(25, 75, size=n_nodes)
    sigma_nodes = mu_nodes / 4.0
    # Decaying sensitivity by level
    rho_nodes = np.clip(0.9 - 0.7 * (np.arange(n_nodes) / n_nodes), 0.1, 0.95)
    
    # Ground truth samples
    c_obs = [rng.normal(mu_nodes[i] * (1.0 + rho_nodes[i] * true_xi), sigma_nodes[i]) for i in range(n_nodes)]
    
    sigma_0 = 1.0
    accum_prec = 1.0 / (sigma_0 ** 2)
    accum_num = 0.0
    
    means = [0.0]
    stds = [sigma_0]
    
    for i in range(n_nodes):
        w_i = (rho_nodes[i] * mu_nodes[i]) / sigma_nodes[i]
        y_i = (c_obs[i] - mu_nodes[i]) / sigma_nodes[i]
        accum_prec += w_i ** 2
        accum_num += w_i * y_i
        
        post_var = 1.0 / accum_prec
        post_m = accum_num * post_var
        means.append(post_m)
        stds.append(np.sqrt(post_var))
        
    steps = np.arange(len(means))
    means = np.array(means)
    stds = np.array(stds)
    
    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=300)
    ax.axhline(true_xi, color="#dc2626", linestyle="--", linewidth=2, label=f"True Context $\\xi^* = {true_xi}$")
    ax.plot(steps, means, "o-", color="#2563eb", linewidth=2.2, label="Bayesian Posterior Mean $\\hat{\\mu}_\\xi$")
    ax.fill_between(steps, means - 1.96 * stds, means + 1.96 * stds, color="#3b82f6", alpha=0.25, label="95% Credible Interval ($\\pm 1.96 \\sigma_\\xi$)")
    
    ax.set_xlabel("Number of Completed Nodes ($|\\mathcal{K}|$)", fontweight="bold")
    ax.set_ylabel("Context Estimate $\\xi$", fontweight="bold")
    ax.set_title("Online Conjugate Bayesian Context Tracking ($O(1)$ Incremental Update)", fontweight="bold", pad=12)
    ax.set_xlim(0, n_nodes)
    ax.legend(frameon=True, loc="upper right")
    plt.savefig("figure_bayesian_convergence.png", dpi=300)
    plt.close()
    print("[+] Saved figure_bayesian_convergence.png")

def generate_comparative_hud():
    regimes = ["Toba Baseline\n(Nominal)", "Toba under\nCorrelated Clutter", "Context-Coupled\n(Idea 3)", "Dual Novelty\n(Idea 3 + 1)"]
    leads = [167.4, 154.0, 203.1, 203.1]
    lead_lo = [151.5, 137.0, 173.7, 173.7]
    lead_hi = [183.3, 171.0, 232.6, 232.6]
    dsr = [0.0, 0.0, 0.0, 57.9]
    dsr_lo = [0.0, 0.0, 0.0, 36.3]
    dsr_hi = [0.0, 0.0, 0.0, 76.9]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=300)
    
    # Lead time
    yerr_lead = [[m - lo for m, lo in zip(leads, lead_lo)], [hi - m for m, hi in zip(leads, lead_hi)]]
    bars1 = ax1.bar(regimes, leads, yerr=yerr_lead, capsize=4, color=["#64748b", "#94a3b8", "#38bdf8", "#0284c7"], width=0.55)
    ax1.set_ylabel("Lead Time (ms)", fontweight="bold")
    ax1.set_title("Detection Warning Lead Time (ms)", fontweight="bold")
    for b in bars1:
        h = b.get_height()
        ax1.annotate(f"{h:.1f}ms", (b.get_x() + b.get_width()/2, h + 8), ha="center", fontweight="bold", fontsize=9)
    ax1.set_ylim(0, 270)
    
    # DSR
    yerr_dsr = [[m - lo for m, lo in zip(dsr, dsr_lo)], [hi - m for m, hi in zip(dsr, dsr_hi)]]
    bars2 = ax2.bar(regimes, dsr, yerr=yerr_dsr, capsize=4, color=["#64748b", "#94a3b8", "#64748b", "#10b981"], width=0.55)
    ax2.set_ylabel("Deadline Salvage Rate (%)", fontweight="bold")
    ax2.set_title("Deadline Salvage Rate (DSR %)", fontweight="bold")
    for b in bars2:
        h = b.get_height()
        if h > 0:
            ax2.annotate(f"{h:.1f}%", (b.get_x() + b.get_width()/2, h + 4), ha="center", fontweight="bold", fontsize=9)
    ax2.set_ylim(0, 100)
    
    plt.savefig("figure_comparative_summary.png", dpi=300)
    plt.close()
    print("[+] Saved figure_comparative_summary.png")

if __name__ == "__main__":
    generate_bayesian_figure()
    generate_comparative_hud()
