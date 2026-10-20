# Triad Framework: Context-Coupled Plaxity, NP-EDF Queue Awareness & Active RTOS Mitigation for Autonomous DAGs

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status: Research Framework](https://img.shields.io/badge/Status-Research%20Framework-success.svg)]()

A research framework and real-time simulator for **early deadline miss detection, contention-aware plaxity estimation, and dynamic slack reclamation in autonomous DAG workloads**. 

This repository directly extends the state-of-the-art methodology by **Toba, Baruah, & Azumi** (*"Enhancing Deadline Miss Early Detection for DAG Tasks with Variable Probabilistic Thresholds"*, ACM TECS 2026 / ECRTS 2024), transforming passive deadline monitoring into **active, closed-loop RTOS recovery**.

---

## 📌 Key Novelties & Contributions (The Triad Framework)

| Contribution | Focus Area | Mathematical Formulation | Impact |
| :--- | :--- | :--- | :--- |
| **Idea 3: Context-Coupled Plaxity** | Environmental correlation within DAG | Closed-form conjugate Normal-Normal Bayesian estimator $\hat{\xi}$ + Shift Theorem $t_{\text{eff}} = \min(D, t + \hat{\mu}_\xi \Delta \mu_{\text{rem}}(v))$ | **83.0% error reduction** in tail probability lookup |
| **Idea 2: Queue- & Contention-Aware Plaxity** | Multicore processor contention & waiting time | Node-specific NP-EDF dispatch delay simulation $\widehat{W}_v(t) = \min(H)$ over busy cores $R_c(t)$ and higher-priority ready tasks $\mathcal{H}_v(t)$ | **+88.5 ms** warning lead-time gain; 0 additional false alarms |
| **Idea 1: Prediction-Guided RTOS Mitigation** | Impending deadline failure recovery | Non-preemptive priority escalation ($\text{prio} = -1$) + Amdahl-constrained memory-bus DVFS ($\beta = 1.33\times$) | Surges DSR to **91.7%** across 5,000 author-scale instances |

---

## 🏗️ Repository Architecture

The project is cleanly structured into modular directories:

```text
RTOS-main/
├── README.md                          # Comprehensive project guide & reproduction steps
├── requirements.txt                   # Minimal Python package requirements
├── .gitignore                         # Standard Python/Git ignore rules
├── web_gui.py                         # Interactive Web Dashboard (Dual Modes + Triad Visualizer)
│
├── benchmarks/                        # Scientific benchmark & validation experiment suites
│   ├── run_author_scale_benchmark.py  # 5,000-instance author benchmark (Toba Section 5.3)
│   ├── run_rigorous_validation_suite.py # 7-way ablation, 6,000-run sweep, 100-job telemetry
│   ├── run_unified_reproducible_evaluation.py # 1,000-job unified single cohort evaluation
│   ├── run_research_benchmarks.py     # Multi-regime statistical evaluation engine
│   ├── derive_shift_theorem_error_bounds.py # Shift Theorem mathematical bounds (100k MC trials)
│   ├── research_phase_idea2_deep_dive.py # Idea 2 queue delay deep dive
│   ├── evaluate_idea2_contention.py   # Idea 2 contention analysis
│   └── generate_paper_plots.py        # Publication figure generator
│
├── figures/                           # Publication-ready plots & visualizations (PNG)
│   ├── figure_author_scale_5000_benchmark.png # 5,000-instance comparative DSR & lead time
│   ├── figure_robust_contention_sweep.png     # 6,000-run contention load sweep with Wilson CIs
│   ├── figure_shift_theorem_error_bounds.png  # Error surface & bounds of Shift Theorem
│   ├── figure_unified_ablation_matrix.png     # 7-way comparative ablation matrix
│   ├── figure_unified_leadtime_sensitivity.png# Lead-time latency degradation curve
│   ├── figure_bayesian_convergence.png        # Online posterior tracking and variance shrinkage
│   └── figure_comparative_summary.png         # Multi-regime summary chart
│
├── results/                           # Experimental data & telemetry outputs (CSV)
│   ├── author_scale_5000_benchmark_results.csv # Full 5,000-instance benchmark metrics
│   ├── unified_7way_ablation_and_confusion_matrices.csv # Full 7-way confusion matrices
│   ├── all_100_impending_misses_telemetry.csv  # Complete per-job telemetry for all 100 misses
│   ├── dispatch_delay_error_distribution.csv   # N=26,048 queue delay error distribution
│   ├── robust_contention_sweep_1000runs.csv    # 6,000-run load sweep dataset
│   ├── shift_theorem_error_bounds.csv          # Shift theorem error bounds
│   └── ...
│
├── tests/                             # Analytical verification & unit tests
│   ├── test_verification_suite.py     # Equations (8)-(18) verification on chain/branch DAGs
│   ├── test_phase1_bayesian_math.py   # Bayesian conjugate model convergence validation
│   └── test_phase1_tail_prob.py       # Tail probability approximation validation
│
├── papers/                            # Primary literature & extracted research notes
│   ├── toba_ecrts2024.pdf             # Original ECRTS 2024 paper
│   ├── 3808694.pdf                    # ACM TECS 2026 journal paper
│   ├── extracted_paper_equations_and_eval.txt
│   └── extracted_paper_text.txt
│
├── RD-Gen/                            # Authentic DAG generation benchmarks (1,000 YAML DAGs)
│   └── DAGs/                          # Parameter sets NEN_1_Transfer_1 through NEN_4_Transfer_5
│
└── archive/                           # Legacy and preliminary research prototypes
    └── archive_legacy_ast_drift/      # Legacy AST drift experiments
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

Open your browser at **`http://localhost:8050`** to access:
- **Mode A: Interactive Single-DAG Mode:**
  - Microsecond 8-core NP-EDF Gantt timeline with dedicated top-header alarm beacon.
  - Interactive sliders for contention load ($0.25\times - 1.50\times$), scene clutter $\xi$, and sample size.
  - DAG node telemetry inspector showing ready time, dispatch start time, actual wait vs. estimated $\widehat{W}_v(t)$, and Bayesian shift $\Delta \mu_{\text{rem}}(v)$.
- **Mode B: Author-Scale Benchmark Mode:**
  - Complete 5,000-instance evaluation replicating Toba et al. Section 5.3 with live re-run capability.

---

## 🧪 Reproducing Experimental Results

### 1. Verify Paper Equations (Unit Test Suite)
Verifies Equations (8)-(18) from Toba et al. (ACM TECS 2026):
```bash
python tests/test_verification_suite.py
```

### 2. Run Author-Scale Benchmark (5,000 Instances)
Executes 500 randomly generated DAGs 10 times across all 7 configurations:
```bash
python benchmarks/run_author_scale_benchmark.py 500 10
```

### 3. Run Rigorous 7-Way Factorial Ablation & Contention Sweep
Evaluates the 1,000-job cohort, error distributions ($N=26,048$), and 6,000-run load sweep:
```bash
python benchmarks/run_rigorous_validation_suite.py
```

### 4. Verify Shift Theorem Mathematical Bounds (100,000 Monte Carlo Runs)
```bash
python benchmarks/derive_shift_theorem_error_bounds.py
```

---

## 📊 Summary of Experimental Results

### Author-Scale Multi-DAG Benchmark ($N = 5{,}000$ Instances, 500 DAGs × 10 Runs)

Replicating Toba et al. Section 5.3 experimental protocol (424 impending misses identified):

| Configuration | Context (Id 3) | Queue (Id 2) | Mitigate (Id 1) | Recall [95% CI] | Precision [95% CI] | Lead Time (ms) [95% CI] | DSR % [95% CI] | Salvaged Ratio |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Toba Baseline** | No | No | No | 100.0% [99.1, 100.0] | 75.0% [71.3, 78.4] | 117.8 [112.3, 123.2] | **0.0%** [0.0, 0.9] | 0 / 424 |
| **2. Idea 2 Alone** | No | Yes | No | 100.0% [99.1, 100.0] | 75.0% [71.3, 78.4] | 210.7 [199.7, 221.7] | **0.0%** [0.0, 0.9] | 0 / 424 |
| **3. Idea 3 Alone** | Yes | No | No | 100.0% [99.1, 100.0] | 72.7% [69.0, 76.2] | 172.5 [165.9, 179.1] | **0.0%** [0.0, 0.9] | 0 / 424 |
| **4. Ideas 3 + 2** | Yes | Yes | No | 100.0% [99.1, 100.0] | 72.7% [69.0, 76.2] | 267.0 [258.0, 276.0] | **0.0%** [0.0, 0.9] | 0 / 424 |
| **5. Ideas 2 + 1** | No | Yes | Yes | 100.0% [99.1, 100.0] | 75.0% [71.3, 78.4] | 210.7 [199.7, 221.7] | **70.8%** [66.3, 74.9] | 300 / 424 |
| **6. Ideas 3 + 1** | Yes | No | Yes | 100.0% [99.1, 100.0] | 72.7% [69.0, 76.2] | 172.5 [165.9, 179.1] | **67.0%** [62.4, 71.3] | 284 / 424 |
| **7. Triad Framework** | **Yes** | **Yes** | **Yes** | **100.0%** [99.1, 100.0] | **72.7%** [69.0, 76.2] | **267.0** [258.0, 276.0] | **91.7%** [88.7, 94.0] | **389 / 424** |

---

## 📜 Academic References

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
