import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np  # <--- FIX 1: Import numpy

# Load the summary CSV
try:
    df = pd.read_csv("results/final_summary.csv")
except FileNotFoundError:
    print("Error: 'results/summary.csv' not found.")
    print("Please run your C++ benchmark program first to generate the results.")
    exit()


# Ensure output directory exists
output_dir = "plots"
os.makedirs(output_dir, exist_ok=True)

# Get unique k and n_samples values
k_list = sorted(df["k"].unique())
n_list = sorted(df["n_samples"].unique())

# ==============================================================================
# PLOT 1: Parallel Speedup vs. Threads
# ==============================================================================
print("Generating speedup plots...")
for k in k_list:
    plt.figure(figsize=(10, 6))
    subset_k = df[df["k"] == k]
    for n in n_list:
        sub = subset_k[subset_k["n_samples"] == n]
        if sub.empty:
            continue
        plt.plot(
            sub["threads"], sub["speedup"], marker="o", linestyle="-", label=f"n={n}"
        )
    plt.title(f"Parallel Speedup vs. Threads (k={k})")
    plt.xlabel("Number of Threads")
    plt.ylabel("Speedup (Sequential Time / Parallel Time)")
    plt.axhline(y=1, color="gray", linestyle="--", label="Sequential Baseline")
    plt.legend(title="n_samples")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/speedup_k_{k}.png", dpi=150)
    plt.close()

# ==============================================================================
# PLOT 2: Raw Execution Time vs. Threads
# ==============================================================================
print("Generating execution time plots...")
for k in k_list:
    plt.figure(figsize=(10, 7))
    subset_k = df[df["k"] == k]

    # <--- FIX 2: Use `np` directly instead of `pd.np`
    colors = plt.cm.viridis(np.linspace(0, 1, len(n_list)))

    for i, n in enumerate(n_list):
        sub = subset_k[subset_k["n_samples"] == n]
        if sub.empty:
            continue

        plt.plot(
            sub["threads"],
            sub["tp_time"],
            marker="o",
            linestyle="-",
            label=f"n={n}",
            color=colors[i],
        )

        seq_time_val = sub["seq_time"].iloc[0]
        plt.axhline(y=seq_time_val, color=colors[i], linestyle="--")

    plt.yscale("log")

    plt.title(
        f"Execution Time vs. Threads (k={k})\n(Dashed lines represent sequential time)"
    )
    plt.xlabel("Number of Threads")
    plt.ylabel("Execution Time (seconds, log scale)")
    plt.legend(title="n_samples")
    plt.grid(True, which="both", linestyle="--")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/execution_time_k_{k}.png", dpi=150)
    plt.close()

# ==============================================================================
# PLOT 3: Sequential Scaling
# ==============================================================================
print("Generating sequential scaling plot...")
plt.figure(figsize=(10, 6))
for k in k_list:
    sub = df[df["k"] == k].groupby("n_samples")["seq_time"].mean().reset_index()
    plt.plot(sub["n_samples"], sub["seq_time"], marker="o", label=f"k={k}")

plt.xscale("log")
plt.yscale("log")
plt.title("Sequential Runtime vs. Dataset Size (log-log scale)")
plt.xlabel("Number of Samples (log scale)")
plt.ylabel("Sequential Time (seconds, log scale)")
plt.legend(title="k")
plt.grid(True, which="both", linestyle="--")
plt.tight_layout()
plt.savefig(f"{output_dir}/sequential_scaling.png", dpi=150)
plt.close()

print(f"\nAll plots have been saved to the '{output_dir}/' directory.")
