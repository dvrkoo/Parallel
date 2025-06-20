import pandas as pd
import matplotlib.pyplot as plt
import os

# Load the summary CSV
df = pd.read_csv("results/summary.csv")

# Ensure output directory exists
output_dir = "plots"
os.makedirs(output_dir, exist_ok=True)

# Get unique k and n_samples values
k_list = sorted(df["k"].unique())
n_list = sorted(df["n_samples"].unique())

# Plot: for each k, speedup vs threads with lines for each n_samples
for k in k_list:
    plt.figure()
    subset_k = df[df["k"] == k]
    for n in n_list:
        sub = subset_k[subset_k["n_samples"] == n]
        if sub.empty:
            continue
        plt.plot(sub["threads"], sub["speedup"], marker="o", label=f"n={n}")
    plt.title(f"Parallel Speedup vs Threads (k={k})")
    plt.xlabel("Number of Threads")
    plt.ylabel("Speedup")
    plt.legend(title="n_samples")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/speedup_k_{k}.png", dpi=150)
    plt.show()

# Insightful plot for sequential case:
# Plot runtime vs. n_samples for different k (log-log scale recommended)
plt.figure()
for k in k_list:
    # select one representative time per (n_samples, k): all seq_time are same across threads
    sub = df[df["k"] == k].groupby("n_samples")["seq_time"].mean().reset_index()
    plt.plot(sub["n_samples"], sub["seq_time"], marker="o", label=f"k={k}")

plt.xscale("log")
plt.yscale("log")
plt.title("Sequential Runtime vs. Dataset Size (log-log)")
plt.xlabel("Number of Samples (log scale)")
plt.ylabel("Sequential Time (s, log scale)")
plt.legend(title="k")
plt.grid(True, which="both", linestyle="--")
plt.tight_layout()
plt.savefig(f"{output_dir}/sequential_scaling.png", dpi=150)
plt.show()
