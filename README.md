# Context-Coupled Plaxity & Non-Preemptive RTOS Slack Reclamation for Autonomous DAGs

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status: Research Prototype](https://img.shields.io/badge/Status-Research%20Prototype-success.svg)]()

A research framework and real-time simulator for **early deadline miss detection and dynamic slack reclamation in autonomous DAG workloads**. 

This repository directly extends the state-of-the-art methodology by **Toba, Baruah, & Azumi** (*"Enhancing Deadline Miss Early Detection for DAG Tasks with Variable Probabilistic Thresholds"*, ACM TECS 2026 / ECRTS 2024), transforming passive deadline monitoring into **active, closed-loop RTOS recovery**.

---

## 📌 Key Novelties & Contributions

1. **Context-Coupled Plaxity (Idea 3):**
   - Breaks the restrictive assumption of independent task execution times under environmental uncertainty.
   - Models shared environmental clutter via a latent state $\xi \in \mathbb{R}$ with task-specific sensitivities $\rho_v$.
   - Implements an online **closed-form Conjugate Normal-Normal Bayesian estimator** updating in $O(1)$ time per task completion.
   - Proves the **Effective Plaxity Shift Theorem** ($t_{\text{eff}} = \min(D, t + \hat{\mu}_\xi \Delta \mu_{\text{rem}}(v))$), achieving an **83.0% error reduction** over baseline tail miss approximations without runtime convolution.

2. **Prediction-Guided Non-Preemptive RTOS Mitigation (Idea 1):**
   - Converts early detection warning lead time into dynamic slack reclamation instead of merely reporting impending failures.
   - **Strict Non-Preemptive NP-EDF:** Running tasks on cores are never interrupted, preserving cache locality and avoiding context-switch jitter.
   - **Ready-Queue Priority Escalation:** Elevates unassigned DAG tasks to emergency priority ($\text{prio} = -1$) ahead of competing Poisson background workloads.
   - **Hardware-Grounded Amdahl DVFS:** Models CPU-bound frequency scaling ($\beta = 1.33\times$) constrained by memory-bus stalls based on NVIDIA Jetson AGX Orin / ARM Cortex-A78AE architectures.
   - **Empirical Performance:** Recovers **61.0%** of impending deadline failures across a unified 1,000-job cohort over 20 diverse RD-Gen DAG topologies.

---

## 🏗️ Repository Architecture

```text
RTOS-main/
├── RD-Gen/                                # Azumi Lab Random DAG Generator (1,000 authentic DAGs)
│   └── DAGs/                              # Parameter sets NEN_1_Transfer_1 through NEN_4_Transfer_5
├── web_gui.py                             # Interactive Web Dashboard (Zero-dependency HTTP + Gantt UI)
├── run_unified_reproducible_evaluation.py # Single-Cohort Master Benchmark (1,000 runs, unified seed)
├── derive_shift_theorem_error_bounds.py   # Shift Theorem mathematical error analysis (100k MC trials)
├── run_research_benchmarks.py             # Multi-regime statistical evaluation engine
├── test_verification_suite.py             # Unit verification tests for paper Equations (8)-(18)
├── test_phase1_bayesian_math.py           # Verification of Bayesian convergence & O(1) updates
├── generate_paper_plots.py                # Publication figure generation script
├── requirements.txt                       # Python dependencies
├── .gitignore                             # Git ignore rules
│
├── Figures & Visual Artifacts:
│   ├── figure_shift_theorem_error_bounds.png      # Tail probability error surface & bounds
│   ├── figure_unified_ablation_matrix.png         # 2x2 Factorial ablation study (N=1,000)
│   ├── figure_unified_leadtime_sensitivity.png    # Warning lead-time latency degradation curve
│   ├── figure_bayesian_convergence.png            # Online posterior tracking and variance shrinkage
│   └── figure_comparative_summary.png             # Baseline vs. Dual Novelty comparison
│
└── Reconciled Benchmark CSVs:
    ├── unified_benchmark_results.csv              # Full multi-regime metrics with 95% Wilson CIs
    ├── unified_ablation_results.csv               # 2x2 Ablation statistics (Priority vs DVFS)
    ├── unified_leadtime_sensitivity.csv           # Injected notification latency sweep (tau = 0-200ms)
    └── shift_theorem_error_bounds.csv             # Monte Carlo empirical error matrix
```

---

## ⚡ Quick Start

### 1. Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd RTOS-main
pip install -r requirements.txt
```

### 2. Launch the Interactive Web Dashboard

Launch the zero-dependency web dashboard:

```bash
python web_gui.py
```

Open your browser at **`http://localhost:8050`** to inspect:
- **Interactive Multi-Core Gantt Chart:** Visualizes 8-core NP-EDF dispatch, Poisson background interference, and the high-visibility **crimson ALARM beacon**.
- **20-Node DAG Visualizer:** Graph layout showing topological levels, WCETs, and variable threshold progressions ($\theta_{\text{min}} \to \theta_{\text{max}}$).
- **Novelty Simulation Modes:** Toggle between baseline, Context-Coupled Plaxity, and Active RTOS Mitigation.

---

## 🧪 Reproducing Experimental Results

### 1. Verify Paper Equations (Baseline Verification)
Verifies Equations (8)-(18) from Toba et al. (ACM TECS 2026):
```bash
python test_verification_suite.py
```

### 2. Verify Shift Theorem Mathematical Bounds (100,000 Monte Carlo Runs)
Computes empirical tail probability errors between exact correlated convolution, first-order shift $t_{\text{eff}}$, and second-order variance scaling:
```bash
python derive_shift_theorem_error_bounds.py
```

### 3. Run the Unified Reproducible Benchmark (1,000-Job Single Cohort)
Evaluates the unified 20-DAG cohort across the comparative benchmark, 2×2 factorial ablation, and warning lead-time latency sweep:
```bash
python run_unified_reproducible_evaluation.py
```

---

## 📊 Summary of Experimental Results

Evaluated across a fixed cohort of **20 authentic 20-node DAGs × 50 jobs = 1,000 total runs** under Poisson background load (100 impending misses):

### 1. Multi-Regime Comparative Benchmark

| Regime | Impending Miss % [95% CI] | Recall % [95% CI] | Precision % [95% CI] | FPR % [95% CI] | Warning Lead Time (ms) [95% CI] | Deadline Salvage Rate (DSR %) [95% CI] |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Toba Baseline (Nominal)** | 10.0% [8.3, 12.0] | 100.0% [96.3, 100.0] | 59.2% [51.6, 66.3] | 7.7% [6.1, 9.6] | 121.9 [111.2, 132.6] | 0.0% (Passive Alarming) |
| **2. Toba under Correlated Clutter** | 10.0% [8.3, 12.0] | 100.0% [96.3, 100.0] | 59.2% [51.6, 66.3] | 7.7% [6.1, 9.6] | 121.9 [111.2, 132.6] | 0.0% (Passive Alarming) |
| **3. Context-Coupled Plaxity (Idea 3)** | 10.0% [8.3, 12.0] | 100.0% [96.3, 100.0] | 58.1% [50.7, 65.3] | 8.0% [6.4, 10.0] | **171.5 [156.9, 186.1]** | 0.0% (Passive Alarming) |
| **4. Dual Novelty (Ideas 3 + 1)** | 10.0% [8.3, 12.0] | 100.0% [96.3, 100.0] | 58.1% [50.7, 65.3] | 8.0% [6.4, 10.0] | **171.5 [156.9, 186.1]** | **61.0% [51.2, 70.0]** (61 / 100) |

> **Lead-Time Advance:** Context-Coupled Plaxity advances warning lead time by **+49.6 ms** ($p < 0.0001$, Welch's $t$-test) compared to the baseline under clutter.

### 2. 2×2 Factorial Ablation Matrix

| Configuration | Ready-Queue Priority | DVFS Frequency | Salvaged / Impending | DSR % [95% Wilson CI] |
| :--- | :---: | :---: | :---: | :---: |
| **None (Passive Alarming)** | Nominal ($D$) | Nominal ($1.0\times$) | 0 / 100 | 0.0% [0.0%, 3.7%] |
| **Amdahl DVFS Only** | Nominal ($D$) | Boosted ($1.33\times$) | 12 / 100 | 12.0% [7.0%, 19.8%] |
| **Priority Escalation Only** | Emergency ($-1$) | Nominal ($1.0\times$) | 32 / 100 | 32.0% [23.7%, 41.7%] |
| **Combined (Dual Mitigate)** | Emergency ($-1$) | Boosted ($1.33\times$) | **61 / 100** | **61.0% [51.2%, 70.0%]** |

### 3. Warning Lead-Time Latency Sensitivity Analysis

| Injected Delay $\tau$ (ms) | Salvaged / Impending | DSR % [95% Wilson CI] | Schedulability Status |
| :---: | :---: | :---: | :--- |
| **0 ms** | **61 / 100** | **61.0% [51.2%, 70.0%]** | Maximum dynamic slack available |
| **25 ms** | 39 / 100 | 39.0% [30.0%, 48.8%] | Slack reduced by 25 ms |
| **50 ms** | 21 / 100 | 21.0% [14.2%, 30.0%] | Approaching non-preemptive blocking bound $B(t)$ |
| **75 ms** | 8 / 100 | 8.0% [4.1%, 15.0%] | Slack exhausted on most critical paths |
| **100 ms** | 3 / 100 | 3.0% [1.0%, 8.5%] | Severe deadline collapse |
| **150 ms** | 2 / 100 | 2.0% [0.6%, 7.0%] | Peripheral short paths only |
| **200 ms** | 0 / 100 | 0.0% [0.0%, 3.7%] | Physical recovery infeasible |

---

## 📜 Academic References

If you build upon or reference this work, please cite:

```bibtex
@article{toba2026enhancing,
  title={Enhancing Deadline Miss Early Detection for DAG Tasks with Variable Probabilistic Thresholds},
  author={Toba, Taiki and Baruah, Sanjoy and Azumi, Takuya},
  journal={ACM Transactions on Embedded Computing Systems},
  year={2026},
  publisher={ACM},
  doi={10.1145/3808694}
}
```

```bibtex
@inproceedings{toba2024early,
  title={Deadline Miss Early Detection for DAG Tasks with Probabilistic Execution Times},
  author={Toba, Taiki and Baruah, Sanjoy and Azumi, Takuya},
  booktitle={36th Euromicro Conference on Real-Time Systems (ECRTS 2024)},
  year={2024}
}
```

---

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
