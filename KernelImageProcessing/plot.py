import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import sys
import os
import argparse


def create_plot(
    df, x_col, y_col, x_label, y_label, title, output_filename, hue_col=None
):
    """
    Generic plotting function for creating and saving a single, well-formatted plot.
    Can create plots with or without a hue for combined/separate styles.
    """
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(12, 7))

    sns.lineplot(
        data=df, x=x_col, y=y_col, hue=hue_col, marker="o", markersize=8, ax=ax, lw=2.5
    )

    ax.set_title(title, fontsize=16, pad=20)
    ax.set_xlabel(x_label, fontsize=12, labelpad=15)
    ax.set_ylabel(y_label, fontsize=12, labelpad=15)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)

    if hue_col:
        ax.legend(title="Kernel Type")

    # Rotate x-axis labels if they are long strings
    if df[x_col].dtype == "object":
        plt.xticks(rotation=30, ha="right")

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

    # --- Setup plot parameters based on mode ---
    y_col = "Speedup"
    y_label = "Speedup Factor (CPU Time / GPU Time)"
    kernel_col_name = "KernelType" if mode == "kernelsize" else "Kernel"

    if mode == "blocksize":
        df["x_axis_label"] = (
            df["BlockSize"].astype(str)
            + "\n("
            + df["ThreadsPerBlock"].astype(str)
            + " thr)"
        )
        x_col, x_label = "x_axis_label", "Block Dimensions"
        title_prefix = "Block Size vs. Speedup"

    elif mode == "scaling":
        df["x_axis_label"] = df["UtilizationFraction"] * 100
        x_col, x_label = "x_axis_label", "GPU Utilization (%)"
        title_prefix = "Strong Scaling vs. Speedup"

    elif mode == "throughput":
        # Simple and direct: use the 'Resolution' column for the x-axis
        # To ensure correct order, we create a categorical type.
        df["TotalPixels"] = df["Resolution"].apply(
            lambda res: int(res.split("x")[0]) * int(res.split("x")[1])
        )
        df = df.sort_values("TotalPixels")
        x_col, x_label = "Resolution", "Image Resolution"
        title_prefix = "Throughput vs. Speedup"

    elif mode == "kernelsize":
        df = df.sort_values("KernelSize")
        df["x_axis_label"] = (
            df["KernelSize"].astype(str) + "x" + df["KernelSize"].astype(str)
        )
        x_col, x_label = "x_axis_label", "Kernel Size (N x N)"
        title_prefix = "Kernel Size vs. Speedup"

    else:
        print(f"Invalid mode '{mode}'")
        return

    # --- Plotting Dispatch ---
    filename_suffix = (
        "_shared_mem"
        if "Optimization" in df.columns and df["Optimization"].iloc[0] == "Shared"
        else ""
    )
    output_dir = f"plots_{mode}"
    os.makedirs(output_dir, exist_ok=True)

    if style == "separate":
        for kernel in df[kernel_col_name].unique():
            kernel_df = df[df[kernel_col_name] == kernel]
            title = f"{title_prefix} for {kernel} Kernel"
            output_filename = os.path.join(
                output_dir, f"{kernel}_{mode}_vs_speedup{filename_suffix}.png"
            )
            create_plot(
                kernel_df, x_col, y_col, x_label, y_label, title, output_filename
            )
    elif style == "combined":
        title = f"Combined {title_prefix} Analysis"
        output_filename = os.path.join(
            output_dir, f"combined_{mode}_vs_speedup{filename_suffix}.png"
        )
        create_plot(
            df,
            x_col,
            y_col,
            x_label,
            y_label,
            title,
            output_filename,
            hue_col=kernel_col_name,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate plots from CUDA convolution benchmark data."
    )
    parser.add_argument(
        "mode", choices=["all", "blocksize", "scaling", "throughput", "kernelsize"]
    )
    parser.add_argument("--style", choices=["separate", "combined"], default="separate")
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
