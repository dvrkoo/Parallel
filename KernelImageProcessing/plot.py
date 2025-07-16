import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import sys
import os
import argparse


# In generate_plots.py


def create_plot(
    df,
    x_col,
    y_col,
    x_label,
    y_label,
    title,
    output_filename,
    mode,
    hue_col=None,
    style_col=None,
):
    """
    Generic plotting function for creating and saving a single, well-formatted plot.
    Handles single-line, multi-line (hue), and comparison (style) plots.
    """
    # --- Setup Plot Style ---
    # Use a different theme for combined plots to make colors stand out.
    if hue_col or style_col:
        sns.set_theme(style="darkgrid", palette="viridis")
    else:
        sns.set_theme(style="whitegrid", palette="rocket")

    fig, ax = plt.subplots(figsize=(12, 7))

    # --- Plotting Call ---
    sns.lineplot(
        data=df,
        x=x_col,
        y=y_col,
        hue=hue_col,
        style=style_col,
        marker="o",
        markersize=8,
        ax=ax,
        lw=2.5,
    )

    # --- Formatting ---
    ax.set_title(title, fontsize=16, pad=20)
    ax.set_xlabel(x_label, fontsize=12, labelpad=15)
    ax.set_ylabel(y_label, fontsize=12, labelpad=15)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)

    # --- LEGEND FIX ---
    # Add a legend if multiple lines are being drawn (i.e., if hue or style is used).
    if hue_col or style_col:
        # Move the legend outside the plot area for clarity
        ax.legend(title="Legend", bbox_to_anchor=(1.02, 1), loc="upper left")
        # Adjust layout to make space for the legend
        plt.tight_layout(rect=[0, 0, 0.85, 1])
    else:
        # If there's no hue or style, there's no legend to draw.
        # Use a standard tight_layout.
        plt.tight_layout()

    # --- Axis Tick Formatting ---
    if df[x_col].dtype == "object":
        # Don't rotate if there are too many labels, let matplotlib decide
        if len(df[x_col].unique()) < 10:
            plt.xticks(rotation=30, ha="right")

    if mode == "scaling":
        ax.xaxis.set_major_formatter(mticker.PercentFormatter())

    # --- Save and Close ---
    plt.savefig(output_filename, dpi=300)
    plt.close(fig)
    print(f"Plot saved as '{output_filename}'")


def get_plot_params(df, mode):
    """Helper to prepare dataframe and get plot parameters."""
    y_col = "Speedup"
    y_label = "Speedup Factor (CPU Time / GPU Time)"
    kernel_col = "KernelType" if mode == "kernelsize" else "Kernel"

    if mode == "blocksize":
        df["x_axis_label"] = (
            df["BlockSize"].astype(str)
            + " ("
            + df["ThreadsPerBlock"].astype(str)
            + " thr)"
        )
        x_col, x_label = "x_axis_label", "Block Dimensions"
        title_prefix = "Block Size vs. Speedup"
    elif mode == "throughput":
        df["TotalPixels"] = df["Resolution"].apply(
            lambda res: int(res.split("x")[1]) * int(res.split("x")[1])
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
        return None, None, None, None, None, None, None

    return df, x_col, y_col, x_label, y_label, title_prefix, kernel_col


# (create_plot and get_plot_params helper functions remain the same as the previous version)


def run_single_file_plotter(mode, style, csv_path):
    """
    The original logic for plotting a single CSV file for blocksize and kernelsize.
    """
    print(f"\n--- Generating Plots for Mode: {mode.upper()} from '{csv_path}' ---")
    try:
        df = pd.read_csv(csv_path)
        if df.empty:
            print(f"Warning: CSV file '{csv_path}' is empty. Skipping.")
            return
    except (FileNotFoundError, pd.errors.EmptyDataError):
        print(f"Warning: Could not read or file is empty: '{csv_path}'. Skipping.")
        return

    df, x_col, y_col, x_label, y_label, title_prefix, kernel_col = get_plot_params(
        df, mode
    )
    if not x_col:
        return

    output_dir = f"plots_{mode}"
    os.makedirs(output_dir, exist_ok=True)

    # Filename suffix is simpler here as there's no stock/shared distinction
    filename_suffix = (
        "_shared"
        if "Optimization" in df.columns and df["Optimization"].iloc[0] == "Shared"
        else ""
    )

    if style == "separate":
        for kernel in df[kernel_col].unique():
            kernel_df = df[df[kernel_col] == kernel]
            title = f"{title_prefix} for {kernel} Kernel"
            output_filename = os.path.join(
                output_dir, f"{kernel}_{mode}{filename_suffix}.png"
            )
            create_plot(
                kernel_df, x_col, y_col, x_label, y_label, title, output_filename, mode
            )
    elif style == "combined":
        title = f"Combined {title_prefix} Analysis"
        output_filename = os.path.join(
            output_dir, f"combined_{mode}{filename_suffix}.png"
        )
        create_plot(
            df,
            x_col,
            y_col,
            x_label,
            y_label,
            title,
            output_filename,
            mode,
            hue_col=kernel_col,
        )


def run_throughput_comparison_plotter():
    """
    Specialized logic ONLY for the throughput benchmark to find and compare
    stock vs. shared CSVs.
    """
    mode = "throughput"
    print(f"\n--- Generating Comparison Plot for Mode: {mode.upper()} ---")
    stock_csv = f"output/stock_benchmark_{mode}.csv"
    shared_csv = f"output/shared_benchmark_{mode}.csv"

    # Check for existence of both files
    if not os.path.exists(stock_csv) or not os.path.exists(shared_csv):
        print(
            f"Info: To generate a comparison plot, both '{stock_csv}' and '{shared_csv}' must exist."
        )
        print("Skipping comparison plot.")
        return

    try:
        df_stock = pd.read_csv(stock_csv)
        df_shared = pd.read_csv(shared_csv)
        if df_stock.empty or df_shared.empty:
            print(
                "Warning: One of the comparison CSVs is empty. Skipping comparison plot."
            )
            return
        df_combined = pd.concat([df_stock, df_shared], ignore_index=True)
    except Exception as e:
        print(f"Error reading or combining CSV files for comparison: {e}")
        return

    df_combined, x_col, y_col, x_label, y_label, title_prefix, kernel_col = (
        get_plot_params(df_combined, mode)
    )
    if not x_col:
        return

    output_dir = "plots_comparison"
    os.makedirs(output_dir, exist_ok=True)
    title = "Comparison: Throughput vs. Speedup (Global vs. Shared Memory)"
    output_filename = os.path.join(output_dir, "comparison_throughput.png")

    create_plot(
        df_combined,
        x_col,
        y_col,
        x_label,
        y_label,
        title,
        output_filename,
        mode,
        hue_col=kernel_col,
        style_col="Optimization",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate plots from CUDA convolution benchmark data."
    )
    parser.add_argument(
        "mode",
        choices=["all", "throughput", "blocksize", "kernelsize"],
        help="The benchmark mode to plot.",
    )
    parser.add_argument(
        "--style",
        choices=["separate", "combined"],
        default="separate",
        help="Plot style for single-file modes (blocksize, kernelsize).",
    )
    args = parser.parse_args()

    # --- NEW MAIN DISPATCHER LOGIC ---
    if args.mode == "all":
        # For 'all', run all modes with their specific logic
        run_single_file_plotter(
            "blocksize", args.style, "output/benchmark_blocksize.csv"
        )
        run_single_file_plotter(
            "kernelsize", args.style, "output/benchmark_kernelsize.csv"
        )
        # Throughput is special: it plots individual files AND the comparison
        run_single_file_plotter(
            "throughput", args.style, "output/stock_benchmark_throughput.csv"
        )
        run_single_file_plotter(
            "throughput", args.style, "output/shared_benchmark_throughput.csv"
        )
        run_throughput_comparison_plotter()

    elif args.mode == "throughput":
        # If just throughput is selected, do all three potential plots for it
        run_single_file_plotter(
            "throughput", args.style, "output/stock_benchmark_throughput.csv"
        )
        run_single_file_plotter(
            "throughput", args.style, "output/shared_benchmark_throughput.csv"
        )
        run_throughput_comparison_plotter()

    elif args.mode in ["blocksize", "kernelsize"]:
        # For other modes, just run the simple single-file plotter
        csv_path = f"output/benchmark_{args.mode}.csv"
        run_single_file_plotter(args.mode, args.style, csv_path)
    else:
        print(f"Error: Unknown mode '{args.mode}'")
