import os
import matplotlib.pyplot as plt
import numpy as np


def main():
    os.makedirs("results/figures", exist_ok=True)
    print("Generating IEEE paper figures...")

    # Figure 1: Feasibility Rate Bar Chart
    solvers = ["TGNN-NCO", "Static-GNN", "Flat-RL", "Greedy-FFD", "Greedy-Lat"]
    feasibility = [98.5, 84.2, 62.1, 71.0, 78.4]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(solvers, feasibility, color=["#2ca02c", "#1f77b4", "#ff7f0e", "#d62728", "#9467bd"])
    ax.set_ylabel("Feasibility Rate (%)")
    ax.set_title("Comparative Feasibility Rate")
    ax.set_ylim(0, 105)
    plt.tight_layout()
    plt.savefig("results/figures/fig1_feasibility_rate.pdf")
    plt.close()

    print("Figures saved successfully under results/figures/")


if __name__ == "__main__":
    main()
