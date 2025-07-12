import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import sys


# --- Plotting Function for Block Size Benchmark ---
def plot_blocksize_performance(csv_path="output/benchmark_blocksize.csv"):
    """
    Reads the block size benchmark CSV and plots GPU execution time vs. threads per block.
    """
    print(f"--- Generating Block Size Performance Plot from '{csv_path}' ---")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: The file '{csv_path}' was not found.")
        return

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(12, 7))
    kernels = df["Kernel"].unique()

    # Create a custom categorical x-axis label for clarity
    df["BlockLabel"] = (
        df["BlockSize"] + "\n(" + df["ThreadsPerBlock"].astype(str) + " thr)"
    )

    # Use seaborn for easier plotting with hues
    sns.lineplot(
        data=df,
        x="BlockLabel",
        y=df["AvgGPUTime"] * 1000,
        hue="Kernel",
        marker="o",
        markersize=8,
        ax=ax,
        palette="viridis",
        sort=False,
    )

    ax.set_title(
        f'GPU Kernel Performance vs. Block Size\n(Resolution: {df["Resolution"].iloc[0]})',
        fontsize=16,
        pad=20,
    )
    ax.set_xlabel(
        "Block Dimensions (Total Threads per Block)", fontsize=12, labelpad=15
    )
    ax.set_ylabel("Average GPU Execution Time (ms)", fontsize=12, labelpad=15)
    plt.xticks(rotation=45, ha="right")
    ax.legend(title="Kernel Type", fontsize=10)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.tight_layout()

    output_filename = "blocksize_performance.png"
    plt.savefig(output_filename, dpi=300)
    print(f"Plot saved as '{output_filename}'")


# --- Plotting Function for Scaling Benchmark ---
def plot_scaling_performance(csv_path="output/benchmark_scaling.csv"):
    """
    Reads the scaling benchmark CSV and plots GPU execution time vs. utilization fraction.
    """
    print(f"--- Generating Scaling Performance Plot from '{csv_path}' ---")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: The file '{csv_path}' was not found.")
        return

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(12, 7))

    # Utilization Fraction is the natural x-axis
    df["Utilization %"] = df["UtilizationFraction"] * 100

    sns.lineplot(
        data=df,
        x="Utilization %",
        y=df["AvgGPUTime"] * 1000,
        hue="Kernel",
        marker="o",
        markersize=8,
        ax=ax,
        palette="plasma",
    )

    ax.set_title(
        f'GPU Strong Scaling Performance\n(Fixed Resolution: {df["Resolution"].iloc[0]})',
        fontsize=16,
        pad=20,
    )
    ax.set_xlabel("GPU Utilization via Grid Size Limit (%)", fontsize=12, labelpad=15)
    ax.set_ylabel("Average GPU Execution Time (ms)", fontsize=12, labelpad=15)

    # Use a log scale for the y-axis if values are very different, but linear is fine here.
    # ax.set_yscale('log')

    # Format x-axis as percentages
    ax.xaxis.set_major_formatter(mticker.PercentFormatter())

    ax.legend(title="Kernel Type")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.tight_layout()

    output_filename = "scaling_performance.png"
    plt.savefig(output_filename, dpi=300)
    print(f"Plot saved as '{output_filename}'")


# --- Plotting Function for Throughput Benchmark ---
def plot_throughput_performance(csv_path="output/benchmark_throughput.csv"):
    """
    Reads the throughput benchmark CSV and plots Speedup vs. Image Resolution (Total Pixels).
    """
    print(f"--- Generating Throughput Performance Plot from '{csv_path}' ---")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: The file '{csv_path}' was not found.")
        return

    # --- FIX: Calculate total pixels directly from the 'Resolution' column ---
    # This is much more robust than parsing directory names.
    def get_pixels_from_col(res_str):
        try:
            parts = res_str.lower().split("x")
            if len(parts) == 2:
                return int(parts[0]) * int(parts[1])
            return 0
        except:
            return 0

    df["TotalPixels"] = df["Resolution"].apply(get_pixels_from_col)

    # Sort the dataframe by the number of pixels to ensure the lines on the plot
    # connect in the correct order (from smallest to largest resolution).
    df = df.sort_values("TotalPixels").reset_index(drop=True)

    sns.set_theme(style="darkgrid")
    fig, ax = plt.subplots(figsize=(12, 7))

    sns.lineplot(
        data=df,
        x="TotalPixels",
        y="Speedup",
        hue="Kernel",
        marker="o",
        markersize=8,
        ax=ax,
        palette="magma",
    )

    ax.set_title(
        "CPU vs. GPU Speedup Across Different Image Resolutions", fontsize=16, pad=20
    )
    ax.set_xlabel("Image Size (Total Pixels)", fontsize=12, labelpad=15)
    ax.set_ylabel("Speedup Factor (CPU Time / GPU Time)", fontsize=12, labelpad=15)

    # Use a log scale for the x-axis to better visualize different resolutions
    ax.set_xscale("log")

    # Format the x-axis with commas for readability (e.g., 2,000,000)
    ax.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, p: format(int(x), ","))
    )
    plt.xticks(rotation=30)

    ax.legend(title="Kernel Type")
    plt.tight_layout()

    output_filename = "throughput_performance.png"
    plt.savefig(output_filename, dpi=300)
    print(f"Plot saved as '{output_filename}'")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot_all_benchmarks.py <mode>")
        print("Modes:")
        print("  --blocksize    (default)")
        print("  --scaling")
        print("  --throughput")
        # Default to blocksize if no argument is given
        plot_blocksize_performance()
    else:
        mode = sys.argv[1]
        if mode == "--blocksize":
            plot_blocksize_performance()
        elif mode == "--scaling":
            plot_scaling_performance()
        elif mode == "--throughput":
            plot_throughput_performance()
        else:
            print(f"Error: Unknown mode '{mode}'")
