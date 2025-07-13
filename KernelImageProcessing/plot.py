import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import os
import argparse  # Using argparse for robust CLI handling


def generate_kernelsize_plots(csv_path, output_dir):
    """
    Generates two plots for the kernel size benchmark.
    1. GPU Time vs. Kernel Size
    2. Speedup vs. Kernel Size
    """
    print(f"\n--- Generating Plots for Mode: KERNELSIZE ---")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: The file '{csv_path}' was not found.")
        return

    # --- Plot 1: GPU Time vs. Kernel Size ---
    fig1, ax1 = plt.subplots(figsize=(12, 7))
    sns.set_theme(style="darkgrid", palette="coolwarm")
    sns.lineplot(
        data=df,
        x="KernelSize",
        y=df["AvgGPUTime"] * 1000,
        hue="KernelType",
        marker="o",
        markersize=8,
        ax=ax1,
    )

    ax1.set_title("GPU Execution Time vs. Kernel Size", fontsize=16, pad=20)
    ax1.set_xlabel("Kernel Dimension (N in NxN)", fontsize=12, labelpad=15)
    ax1.set_ylabel("Average GPU Execution Time (ms)", fontsize=12, labelpad=15)
    ax1.legend(title="Kernel Type")
    ax1.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.tight_layout()

    time_plot_filename = os.path.join(output_dir, "kernelsize_vs_gputime.png")
    plt.savefig(time_plot_filename, dpi=300)
    plt.close(fig1)
    print(f"Plot saved as '{time_plot_filename}'")

    # --- Plot 2: Speedup vs. Kernel Size ---
    fig2, ax2 = plt.subplots(figsize=(12, 7))
    sns.set_theme(style="darkgrid", palette="viridis")
    sns.lineplot(
        data=df,
        x="KernelSize",
        y="Speedup",
        hue="KernelType",
        marker="s",
        markersize=8,
        ax=ax2,
    )

    ax2.set_title("Speedup vs. Kernel Size", fontsize=16, pad=20)
    ax2.set_xlabel("Kernel Dimension (N in NxN)", fontsize=12, labelpad=15)
    ax2.set_ylabel("Speedup Factor (CPU Time / GPU Time)", fontsize=12, labelpad=15)
    ax2.legend(title="Kernel Type")
    ax2.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.tight_layout()

    speedup_plot_filename = os.path.join(output_dir, "kernelsize_vs_speedup.png")
    plt.savefig(speedup_plot_filename, dpi=300)
    plt.close(fig2)
    print(f"Plot saved as '{speedup_plot_filename}'")


def create_separate_plot(
    df, x_col, y_col, x_label, y_label, title, output_filename, mode
):
    """
    Creates and saves a single, well-formatted plot for one kernel.
    """
    # --- ANNOTATION FIX ---
    # We'll calculate the offset based on the axis range for consistent spacing.
    # This requires a two-pass approach: create plot, get range, then add text.

    sns.set_theme(style="whitegrid", palette="rocket")
    fig, ax = plt.subplots(figsize=(11, 6.5))

    sns.lineplot(data=df, x=x_col, y=y_col, marker="o", markersize=8, ax=ax, lw=2.5)

    # --- ANNOTATION FIX - PASS 1: Get axis range ---
    ymin, ymax = ax.get_ylim()
    y_range = ymax - ymin
    y_pos_offset = y_range * 0.025  # Use 2.5% of the total y-axis range as the offset

    # Reset index to safely iterate and compare rows
    df = df.reset_index(drop=True)
    for index, row in df.iterrows():
        va = "bottom"
        final_offset = y_pos_offset
        # Compare with the previous row to avoid text overlap
        if index > 0 and row[y_col] < df.iloc[index - 1][y_col]:
            va = "top"
            final_offset = -y_pos_offset * 1.2  # Make downward offset slightly larger

        ax.text(
            row[x_col],
            row[y_col] + final_offset,
            f" {row[y_col]:.1f}x",
            va=va,
            ha="center",
            fontsize=9.5,
            fontweight="bold",
        )

    # --- Formatting ---
    ax.set_title(title, fontsize=16, pad=20)
    ax.set_xlabel(x_label, fontsize=12, labelpad=15)
    ax.set_ylabel(y_label, fontsize=12, labelpad=15)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)

    if mode == "blocksize" or (mode == "throughput" and len(df[x_col].unique()) < 8):
        plt.xticks(rotation=30, ha="right")

    if mode == "scaling":
        ax.xaxis.set_major_formatter(mticker.PercentFormatter())

    plt.tight_layout()
    plt.savefig(output_filename, dpi=300)
    plt.close(fig)
    print(f"Plot saved as '{output_filename}'")


def create_combined_plot(
    df, x_col, y_col, x_label, y_label, title, output_filename, mode, hue_col
):
    """
    Creates and saves a single plot with all kernels combined.
    Annotations are disabled here to avoid clutter.
    """
    sns.set_theme(style="darkgrid", palette="viridis")
    fig, ax = plt.subplots(figsize=(12, 7))

    sns.lineplot(
        data=df, x=x_col, y=y_col, hue=hue_col, marker="o", markersize=8, ax=ax, lw=2
    )

    ax.set_title(title, fontsize=16, pad=20)
    ax.set_xlabel(x_label, fontsize=12, labelpad=15)
    ax.set_ylabel(y_label, fontsize=12, labelpad=15)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    ax.legend(title="Kernel Type")

    if mode == "blocksize" or (mode == "throughput" and len(df[x_col].unique()) < 8):
        plt.xticks(rotation=30, ha="right")

    if mode == "scaling":
        ax.xaxis.set_major_formatter(mticker.PercentFormatter())

    plt.tight_layout()
    plt.savefig(output_filename, dpi=300)
    plt.close(fig)
    print(f"Plot saved as '{output_filename}'")


def generate_plots(mode, style, csv_path):
    """
    Main plotting dispatcher. Reads data and calls the appropriate plotting function.
    """
    print(
        f"\n--- Generating Plots for Mode: {mode.upper()} | Style: {style.upper()} ---"
    )
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: The file '{csv_path}' was not found.")
        return

    output_dir = f"plots_{mode}"
    os.makedirs(output_dir, exist_ok=True)

    y_col = "Speedup"
    y_label = "Speedup Factor (CPU Time / GPU Time)"
    kernel_col_name = "KernelType" if mode == "kernelsize" else "Kernel"

    # --- DEFINE PLOT PARAMETERS BASED ON MODE ---

    if mode == "blocksize":
        df["x_axis_label"] = (
            df["BlockSize"] + "\n(" + df["ThreadsPerBlock"].astype(str) + " thr)"
        )
        x_col, x_label = "x_axis_label", "Block Dimensions (Total Threads per Block)"
        title_prefix = "Block Size vs. Speedup"

    elif mode == "scaling":
        df["x_axis_label"] = df["UtilizationFraction"] * 100
        x_col, x_label = "x_axis_label", "GPU Utilization (%)"
        title_prefix = "Strong Scaling vs. Speedup"

    elif mode == "throughput":

        def get_pixels(res_str):
            parts = str(res_str).lower().split("x")
            return int(parts[0]) * int(parts[1]) if len(parts) == 2 else 0

        df["TotalPixels"] = df["Resolution"].apply(get_pixels)
        df = df[df["TotalPixels"] > 0].sort_values("TotalPixels")
        df["x_axis_label"] = (
            df["Resolution"]
            + "\n("
            + (df["TotalPixels"] / 1e6).round(1).astype(str)
            + " MP)"
        )
        x_col, x_label = "x_axis_label", "Image Resolution (Megapixels)"
        title_prefix = "Throughput vs. Speedup"

    elif mode == "kernelsize":
        df = df.sort_values("KernelSize")
        df["x_axis_label"] = (
            df["KernelSize"].astype(str) + "x" + df["KernelSize"].astype(str)
        )
        x_col = "x_axis_label"
        x_label = "Kernel Size (N x N)"  # The axis label is now simpler.
        title_prefix = "Kernel Size vs. Speedup"
        # --- END OF FIX ---

    else:
        print(f"Invalid mode '{mode}'")
        return

    # --- PLOTTING LOGIC (No changes needed here) ---
    if style == "separate":
        for kernel in df[kernel_col_name].unique():
            kernel_df = df[df[kernel_col_name] == kernel]
            title = f"{title_prefix} for {kernel} Kernel"
            output_filename = os.path.join(
                output_dir, f"{kernel}_{mode}_vs_speedup.png"
            )
            # Assuming create_separate_plot is defined as in the previous step
            create_separate_plot(
                kernel_df, x_col, y_col, x_label, y_label, title, output_filename, mode
            )
    elif style == "combined":
        title = f"Combined {title_prefix} Analysis"
        output_filename = os.path.join(output_dir, f"combined_{mode}_vs_speedup.png")
        # Assuming create_combined_plot is defined and takes hue_col
        create_combined_plot(
            df,
            x_col,
            y_col,
            x_label,
            y_label,
            title,
            output_filename,
            mode,
            hue_col=kernel_col_name,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate plots from CUDA convolution benchmark data.",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "mode",
        choices=["all", "blocksize", "scaling", "throughput", "kernelsize"],
        help="The type of benchmark data to plot.",
    )

    parser.add_argument(
        "--style",
        choices=["separate", "combined"],
        default="separate",
        help="Plotting style:\n"
        "  separate: Generate one PNG file per kernel (default).\n"
        "  combined: Generate one PNG file with all kernels on the same plot.",
    )

    args = parser.parse_args()

    csv_map = {
        "blocksize": "output/benchmark_blocksize.csv",
        "scaling": "output/benchmark_scaling.csv",
        "throughput": "output/benchmark_throughput.csv",
        "kernelsize": "output/benchmark_kernelsize.csv",
    }

    if args.mode == "all":
        for mode, csv_file in csv_map.items():
            generate_plots(mode, args.style, csv_file)
    elif args.mode in csv_map:
        generate_plots(args.mode, args.style, csv_map[args.mode])
    else:
        print(f"Error: Invalid mode '{args.mode}'")
