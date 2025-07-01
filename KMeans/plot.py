import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

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
# PLOT 1: Parallel Speedup vs. Threads (Unchanged)
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
# PLOT 2: Raw Execution Time vs. Threads (REVISED as requested)
# This version creates a separate plot for EACH combination of n_samples and k.
# ==============================================================================
print("Generating execution time plots for each individual run...")

# Iterate through each combination of n_samples and k
for n in n_list:
    for k in k_list:
        # Filter the dataframe for the specific run (e.g., n=1000 and k=5)
        sub = df[(df["n_samples"] == n) & (df["k"] == k)]

        # Skip if this combination doesn't exist in the data
        if sub.empty:
            continue

        # Create a new figure for this specific plot
        plt.figure(figsize=(10, 6))

        # Plot parallel execution time
        plt.plot(
            sub["threads"],
            sub["tp_time"],
            marker="o",
            linestyle="-",
            color="blue",
            label="Parallel Time",
        )

        # Plot sequential time as a dashed horizontal line for reference
        # We take the first value since it's the same for all thread counts in this subset.
        seq_time_val = sub["seq_time"].iloc[0]
        plt.axhline(
            y=seq_time_val, color="red", linestyle="--", label="Sequential Time"
        )

        # Use a logarithmic scale for the Y-axis to see the trend clearly
        plt.yscale("log")

        # Set a specific title for this run
        plt.title(f"Execution Time vs. Threads (n_samples={n}, k={k})")
        plt.xlabel("Number of Threads")
        plt.ylabel("Execution Time (seconds, log scale)")
        plt.legend()
        plt.grid(True, which="both", linestyle="--")
        plt.tight_layout()

        # Save the figure with a unique, descriptive name
        plt.savefig(f"{output_dir}/execution_time_n{n}_k{k}.png", dpi=150)
        plt.close()  # Close the plot to free up memory


# ==============================================================================
# PLOT 3: Sequential Scaling (Unchanged)
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
