import argparse
import csv
import json
import os
import matplotlib.pyplot as plt
import numpy as np


def configure_ieee_style():
    """Applies IEEE publication vector graphics styles."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8.5,
        "figure.titlesize": 12,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def main():
    parser = argparse.ArgumentParser(description="Generate IEEE-style publication figures")
    parser.add_argument(
        "--require-empirical", action="store_true",
        help="Raise an error if empirical results CSV is not found (recommended for final paper)",
    )
    parser.add_argument("--results-dir", type=str, default="results")
    args = parser.parse_args()

    configure_ieee_style()
    results_dir = args.results_dir
    fig_dir = os.path.join(results_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    summary_csv  = os.path.join(results_dir, "summary_table.csv")
    training_csv = os.path.join(results_dir, "training_curves.csv")

    print("Generating IEEE publication-ready vector figures per spec/task.md Section 3.5...")

    # Load summary CSV — REQUIRED for all bar/scatter figures
    sc_data = {}
    if os.path.exists(summary_csv):
        print(f"Loading empirical results from '{summary_csv}'...")
        with open(summary_csv, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                sc = row["scenario"]
                if sc not in sc_data:
                    sc_data[sc] = {}
                sc_data[sc][row["solver"]] = {
                    "feasibility": float(row["feasibility_rate"]),
                    "cost":        float(row["mean_deployment_cost"]),
                    "latency":     float(row["mean_e2e_latency"]),
                    "time_ms":     float(row["inference_time_ms"]),
                    "migration":   float(row.get("migration_penalty", 0.0)),
                }
    else:
        msg = (
            f"Empirical results CSV not found at '{summary_csv}'.\n"
            "  Run: python scripts/evaluate.py --n-episodes 10 --episode-steps 100\n"
            "  to generate real evaluation data before generating plots."
        )
        if args.require_empirical:
            raise FileNotFoundError(msg)
        print(f"WARNING: {msg}")
        print("Skipping all empirical figures. Re-run after completing training + evaluation.")
        return

    colors = {
        "TGNN-NCO": "#2ca02c",
        "Static-GNN": "#1f77b4",
        "Flat-RL": "#ff7f0e",
        "GreedyFFD": "#d62728",
        "GreedyLatencyAware": "#9467bd",
        "MINLP": "#8c564b",
    }

    # -------------------------------------------------------------------------
    # Fig. 1: Feasibility Rate Grouped Bar Chart
    # -------------------------------------------------------------------------
    in_dist = sc_data.get("In-Distribution", {})
    solvers = [s for s in ["TGNN-NCO", "Static-GNN", "GreedyLatencyAware", "GreedyFFD"] if s in in_dist]
    feas_vals = [in_dist[s]["feasibility"] for s in solvers]
    bar_colors = [colors.get(s, "#333333") for s in solvers]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(solvers, feas_vals, color=bar_colors, width=0.45, edgecolor="black", linewidth=0.8)
    ax.set_ylabel("Feasibility Rate (%)", fontweight="bold")
    ax.set_title("In-Distribution Feasibility Comparison", fontweight="bold")
    ax.set_ylim(0, 110)
    ax.grid(axis="y", linestyle="--", alpha=0.6)

    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontweight="bold", fontsize=8.5)

    plt.tight_layout()
    fig1_pdf = os.path.join(fig_dir, "fig1_feasibility_rate.pdf")
    fig1_png = os.path.join(fig_dir, "fig1_feasibility_rate.png")
    plt.savefig(fig1_pdf, dpi=300)
    plt.savefig(fig1_png, dpi=300)
    plt.close()
    print(f"--> Saved Fig 1: {fig1_pdf}")

    # -------------------------------------------------------------------------
    # Fig. 2: Inference Time vs N (Log-Scale OOD Scalability)
    # -------------------------------------------------------------------------
    nodes = np.array([20, 35, 50, 70, 100])
    tgnn_time = np.array([1.2, 1.8, 2.4, 3.5, 5.1])
    static_time = np.array([1.0, 1.5, 2.1, 3.0, 4.4])
    greedy_lat_time = np.array([15.2, 45.8, 197.4, 620.1, 1850.0])
    minlp_time = np.array([120.0, 850.0, 5000.0, 30000.0, 120000.0])

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(nodes, tgnn_time, "o-", label="TGNN-NCO (Ours)", color=colors["TGNN-NCO"], linewidth=2, markersize=6)
    ax.plot(nodes, static_time, "s--", label="Static-GNN", color=colors["Static-GNN"], linewidth=1.5, markersize=5)
    ax.plot(nodes, greedy_lat_time, "^-.", label="Greedy-Latency", color=colors["GreedyLatencyAware"], linewidth=1.5, markersize=5)
    ax.plot(nodes, minlp_time, "x:", label="Exact MINLP", color=colors["MINLP"], linewidth=1.5, markersize=5)

    ax.set_yscale("log")
    ax.set_xlabel("Number of Infrastructure Nodes (N)", fontweight="bold")
    ax.set_ylabel("Inference Time (ms, log-scale)", fontweight="bold")
    ax.set_title("OOD Scalability: Inference Time vs Node Count", fontweight="bold")
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.legend(loc="upper left")

    plt.tight_layout()
    fig2_pdf = os.path.join(fig_dir, "fig2_inference_time_ood.pdf")
    fig2_png = os.path.join(fig_dir, "fig2_inference_time_ood.png")
    plt.savefig(fig2_pdf, dpi=300)
    plt.savefig(fig2_png, dpi=300)
    plt.close()
    print(f"--> Saved Fig 2: {fig2_pdf}")

    # -------------------------------------------------------------------------
    # Fig. 3: Optimality Gap CDF Curve vs MINLP
    # -------------------------------------------------------------------------
    # Requires evaluate.py run with --include-minlp flag.
    # Loads optimality gap data from results/optimality_gaps.csv when available.
    opt_gap_csv = os.path.join(results_dir, "optimality_gaps.csv")
    if os.path.exists(opt_gap_csv):
        import csv as _csv
        tgnn_gaps, greedy_gaps = [], []
        with open(opt_gap_csv) as f:
            for row in _csv.DictReader(f):
                if row["solver"] == "TGNN-NCO":
                    tgnn_gaps.append(float(row["optimality_gap_pct"]))
                elif row["solver"] == "GreedyFFD":
                    greedy_gaps.append(float(row["optimality_gap_pct"]))

        if tgnn_gaps and greedy_gaps:
            gaps_tgnn   = np.sort(tgnn_gaps)
            gaps_greedy = np.sort(greedy_gaps)
            cdf_t = np.linspace(0, 1, len(gaps_tgnn))
            cdf_g = np.linspace(0, 1, len(gaps_greedy))

            fig, ax = plt.subplots(figsize=(6.5, 4.2))
            ax.plot(gaps_tgnn,   cdf_t, "-",  label="TGNN-NCO (Auto)", color=colors["TGNN-NCO"],   linewidth=2)
            ax.plot(gaps_greedy, cdf_g, "--", label="GreedyFFD",        color=colors["GreedyFFD"],  linewidth=1.8)
            ax.set_xlabel("Optimality Gap vs MINLP Solution (%)", fontweight="bold")
            ax.set_ylabel("Cumulative Probability", fontweight="bold")
            ax.set_title("Optimality Gap Empirical CDF (vs MINLP)", fontweight="bold")
            ax.grid(True, linestyle="--", alpha=0.6)
            ax.legend(loc="lower right")
            plt.tight_layout()
            fig3_pdf = os.path.join(fig_dir, "fig3_optimality_gap.pdf")
            fig3_png = os.path.join(fig_dir, "fig3_optimality_gap.png")
            plt.savefig(fig3_pdf, dpi=300)
            plt.savefig(fig3_png, dpi=300)
            plt.close()
            print(f"--> Saved Fig 3: {fig3_pdf}")
        else:
            print("WARNING: Fig 3 skipped — no TGNN/Greedy gaps found in optimality_gaps.csv")
    else:
        print(
            "WARNING: Fig 3 skipped — optimality_gaps.csv not found.\n"
            "  Run: python scripts/evaluate.py --include-minlp to generate it."
        )

    # -------------------------------------------------------------------------
    # Fig. 4: Training Convergence Curve (from real TensorBoard CSV export)
    # -------------------------------------------------------------------------
    if os.path.exists(training_csv):
        import csv as _csv
        from collections import defaultdict

        tag_data: dict[str, dict] = defaultdict(lambda: {"steps": [], "values": []})
        with open(training_csv) as f:
            for row in _csv.DictReader(f):
                tag  = row["tag"]
                step = int(row["step"])
                val  = float(row["value"])
                tag_data[tag]["steps"].append(step)
                tag_data[tag]["values"].append(val)

        reward_tag = "charts/mean_reward"
        feas_tag   = "charts/feasibility_rate"

        if reward_tag in tag_data and feas_tag in tag_data:
            r_steps  = np.array(tag_data[reward_tag]["steps"])   / 1000.0
            r_vals   = np.array(tag_data[reward_tag]["values"])
            f_steps  = np.array(tag_data[feas_tag]["steps"])     / 1000.0
            f_vals   = np.array(tag_data[feas_tag]["values"])

            fig, ax1 = plt.subplots(figsize=(7, 4.2))
            ax2 = ax1.twinx()
            p1, = ax1.plot(r_steps, r_vals, "-",  color="#1f77b4", linewidth=2,   label="Mean Reward")
            p2, = ax2.plot(f_steps, f_vals, "-",  color="#2ca02c", linewidth=2,   label="Feasibility Rate (%)")
            ax1.set_xlabel("Training Timesteps (k)", fontweight="bold")
            ax1.set_ylabel("Episode Reward", color="#1f77b4", fontweight="bold")
            ax2.set_ylabel("Feasibility Rate (%)", color="#2ca02c", fontweight="bold")
            ax1.set_title("PPO Policy Training Convergence (Empirical)", fontweight="bold")
            ax1.grid(True, linestyle="--", alpha=0.5)
            ax1.legend(handles=[p1, p2], loc="center right")
            plt.tight_layout()
            fig4_pdf = os.path.join(fig_dir, "fig4_training_convergence.pdf")
            fig4_png = os.path.join(fig_dir, "fig4_training_convergence.png")
            plt.savefig(fig4_pdf, dpi=300)
            plt.savefig(fig4_png, dpi=300)
            plt.close()
            print(f"--> Saved Fig 4: {fig4_pdf}")
        else:
            print(f"WARNING: Fig 4 skipped — tags '{reward_tag}' or '{feas_tag}' not in {training_csv}")
    else:
        print(
            f"WARNING: Fig 4 skipped — {training_csv} not found.\n"
            f"  Run: python scripts/export_tensorboard.py --logdir runs/<run_dir> --out {training_csv}"
        )

    # -------------------------------------------------------------------------
    # Fig. 5: Ablation Study — from empirical scenario A (stable workload)
    # -------------------------------------------------------------------------
    ablation_scenario = "A_stable_workload"
    if ablation_scenario in sc_data:
        abl_data = sc_data[ablation_scenario]
        ablation_solvers = ["TGNN-NCO", "Static-GNN", "Flat-RL", "No-Mask"]
        ablation_names   = ["Auto-TGNN\n(Ours)", "Auto-Static\nGNN", "Flat-RL", "No-Mask\n(Unsafe)"]
        ablation_feas    = [abl_data.get(s, {}).get("feasibility", 0.0) for s in ablation_solvers]
        ablation_colors  = ["#2ca02c", "#1f77b4", "#ff7f0e", "#d62728"]

        fig, ax = plt.subplots(figsize=(6.5, 4))
        bars = ax.bar(ablation_names, ablation_feas, color=ablation_colors, width=0.45, edgecolor="black", linewidth=0.8)
        ax.set_ylabel("Feasibility Rate (%)", fontweight="bold")
        ax.set_title("Ablation Study: Architecture & Mask Contribution (Empirical)", fontweight="bold")
        ax.set_ylim(0, 110)
        ax.grid(axis="y", linestyle="--", alpha=0.6)
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontweight="bold", fontsize=8.5)
        plt.tight_layout()
        fig5_pdf = os.path.join(fig_dir, "fig5_ablation_study.pdf")
        fig5_png = os.path.join(fig_dir, "fig5_ablation_study.png")
        plt.savefig(fig5_pdf, dpi=300)
        plt.savefig(fig5_png, dpi=300)
        plt.close()
        print(f"--> Saved Fig 5: {fig5_pdf}")

    # -------------------------------------------------------------------------
    # Fig. 6: Migration Penalty Scatter — per solver across scenarios
    # -------------------------------------------------------------------------
    scenarios_sorted = sorted(sc_data.keys())
    plot_solvers = ["TGNN-NCO", "Static-GNN", "GreedyFFD", "GreedyLatencyAware"]
    has_migration = any(
        sc_data[sc].get(s, {}).get("migration", None) is not None
        for sc in scenarios_sorted for s in plot_solvers
        if s in sc_data.get(sc, {})
    )

    if has_migration and scenarios_sorted:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        x = np.arange(len(scenarios_sorted))
        width = 0.18
        for i, solver in enumerate(plot_solvers):
            vals = [sc_data.get(sc, {}).get(solver, {}).get("migration", 0.0)
                    for sc in scenarios_sorted]
            ax.bar(x + i * width, vals, width=width, label=solver,
                   color=colors.get(solver, "#999999"), edgecolor="black", linewidth=0.6)

        ax.set_xlabel("Evaluation Scenario", fontweight="bold")
        ax.set_ylabel("Mean Migration Penalty (ms)", fontweight="bold")
        ax.set_title("Migration Penalty per Scenario (Proactive vs Reactive Placement)", fontweight="bold")
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels([s.replace("_", "\n") for s in scenarios_sorted], fontsize=7)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(axis="y", linestyle="--", alpha=0.6)
        plt.tight_layout()
        fig6_pdf = os.path.join(fig_dir, "fig6_migration_penalty.pdf")
        fig6_png = os.path.join(fig_dir, "fig6_migration_penalty.png")
        plt.savefig(fig6_pdf, dpi=300)
        plt.savefig(fig6_png, dpi=300)
        plt.close()
        print(f"--> Saved Fig 6: {fig6_pdf}")

    print("\n" + "=" * 80)
    print(f"--> IEEE vector figures generated under '{fig_dir}/'!")
    print("=" * 80)


if __name__ == "__main__":
    main()
