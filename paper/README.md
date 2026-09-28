# IEEE Conference Paper: TGNN-NCO

This directory contains the complete source code, figures, styles, and bibliography for the 8+ page IEEE-format double-column conference paper:

> **"Dynamic AI-Based Placement Optimization on the Cloud Continuum: A Spatio-Temporal Neural Combinatorial Approach with State-Aware Migration"**

---

## Directory Contents

- **`main.tex`**: The primary LaTeX document, fully adhering to standard IEEE double-column conference format (`IEEEtran.cls`), containing all 10 required sections, mathematical derivations, algorithms, tables, and discussions.
- **`references.bib`**: Programmatically verified BibTeX citations with zero hallucinations.
- **`IEEEtran.cls`**: Official IEEE Transactions / Conference document class (CTAN v1.8b).
- **`figures/`**: Publication-ready vector graphics (PDF and high-resolution PNG):
  - `fig1_feasibility_rate.pdf`: In-distribution feasibility comparison.
  - `fig2_inference_time_ood.pdf`: Out-of-distribution latency scalability vs. $N \in [20, 100]$.
  - `fig3_optimality_gap.pdf`: Empirical CDF of optimality gap vs. exact GEKKO MINLP.
  - `fig4_training_convergence.pdf`: PPO policy training convergence curves (reward & feasibility).
  - `fig5_ablation_study.pdf`: Architectural and masking ablation study.
- **`Makefile`**: Standard compilation automation (`make all`, `make clean`).

---

## How to Compile

### 1. Overleaf
1. Create a new Overleaf project ("Upload Project").
2. Zip this `paper/` directory and upload the archive.
3. Set the compiler to `pdfLaTeX` and compile `main.tex`.

### 2. Local Linux / macOS with TeX Live
```bash
cd paper/
make
```
Or manually:
```bash
pdflatex main
bibtex main
pdflatex main
pdflatex main
```

---

## Implementation Gaps & Boundary Notes
The paper explicitly documents the following architectural boundaries (in comments at the top of `main.tex` and in Section VIII):
1. **Simulation vs. Bare-Metal Hardware:** Evaluated on the high-fidelity Gymnasium simulation environment with Waxman topologies and Ornstein-Uhlenbeck stochastic load processes; physical bare-metal Kubernetes deployment is prepared via dry-run API interfaces rather than an in-situ multi-datacenter deployment.
2. **Inter-Domain Federation Signaling:** Cross-operator federation protocols (GSMA OP East-West interface, 3GPP SBA NFs per Dalgitsis et al., 2024) are mathematically integrated into the state migration and routing formulation, but real multi-operator BGP signaling traces were not executed in the test harness.
3. **State Migration Model:** Modeled via the analytical dirty-memory pre-copy transfer model ($T_{\text{mig}} = \frac{\text{dirty\_ratio} \times \text{RAM}}{\text{BW}}$); post-copy, userfaultfd snapshotting, and decoupled external state stores (e.g., Redis/Couchbase) are analyzed as future extensions.
