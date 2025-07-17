import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
import argparse


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
    Generic plotting function with improved legend handling for comparison plots.
    """
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(14, 8))

    palette = "viridis" if hue_col else "rocket"

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
        palette=palette,
    )

    ax.set_title(title, fontsize=18, pad=20)
    ax.set_xlabel(x_label, fontsize=13, labelpad=15)
    ax.set_ylabel(y_label, fontsize=13, labelpad=15)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)

    if style_col and hue_col:
        handles, labels = ax.get_legend_handles_labels()

        ax.get_legend().remove()  # Remove the default, ambiguous legend

        custom_lines = []
        legend_labels = []

        # Group by kernel, then by optimization
        for kernel_name in df[hue_col].unique():
            kernel_df = df[df[hue_col] == kernel_name]
            for opt_name in df[style_col].unique():
                opt_df = kernel_df[kernel_df[style_col] == opt_name]
                if not opt_df.empty:
                    # Get the color and linestyle used by seaborn for this combo
                    line = ax.get_lines()[len(custom_lines)]
                    custom_lines.append(
                        plt.Line2D(
                            [0],
                            [0],
                            color=line.get_color(),
                            lw=2.5,
                            linestyle=line.get_linestyle(),
                        )
                    )
                    legend_labels.append(f"{kernel_name} ({opt_name})")

        ax.legend(
            handles=custom_lines,
            labels=legend_labels,
            title="Legend",
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
        )

    elif hue_col or style_col:
        # This is a simpler plot with only hue or only style
        ax.legend(title="Legend", bbox_to_anchor=(1.02, 1), loc="upper left")
    # --- END OF LEGEND LOGIC ---

    if df[x_col].dtype == "object" and len(df[x_col].unique()) < 10:
        plt.xticks(rotation=30, ha="right")

    # Adjust layout to make space for the legend
    plt.tight_layout(rect=[0, 0, 0.82, 1])

    plt.savefig(output_filename, dpi=300)
    plt.close(fig)
    print(f"  Plot saved: {output_filename}")


def get_plot_params(df, mode):
    """Helper to prepare dataframe and get plot parameters."""
    y_col = "Speedup"
    y_label = "Speedup Factor (CPU/GPU)"
    kernel_col = "KernelType" if mode == "kernelsize" else "Kernel"
    title_prefix = f"{mode.capitalize()} vs. Speedup"

    if mode == "blocksize":
        df["x_axis_label"] = (
            df["BlockSize"] + " (" + df["ThreadsPerBlock"].astype(str) + " thr)"
        )
        x_col, x_label = "x_axis_label", "Block Dimensions"
    elif mode == "throughput":
        df["TotalPixels"] = df["Resolution"].apply(
            lambda res: int(res.split("x")[0]) * int(res.split("x")[1])
        )
        df = df.sort_values("TotalPixels")
        x_col, x_label = "Resolution", "Image Resolution"
    elif mode == "kernelsize":
        df = df.sort_values("KernelSize")
        df["x_axis_label"] = (
            df["KernelSize"].astype(str) + "x" + df["KernelSize"].astype(str)
        )
        x_col, x_label = "x_axis_label", "Kernel Size (N x N)"
    else:
        return [None] * 7  # Return None for all values

    return df, x_col, y_col, x_label, y_label, title_prefix, kernel_col


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate plots from CUDA convolution benchmark data."
    )
    parser.add_argument(
        "mode", choices=["all", "throughput", "blocksize", "kernelsize"]
    )
    parser.add_argument(
        "--style",
        choices=["separate", "combined"],
        default="separate",
        help="Plotting style:\n"
        "separate: One plot per kernel & optimization type, plus combined plots for each opt type.\n"
        "combined: One plot per kernel comparing optimizations, plus one master comparison plot.",
    )
    args = parser.parse_args()

    modes_to_run = (
        ["throughput", "blocksize", "kernelsize"] if args.mode == "all" else [args.mode]
    )

    for mode in modes_to_run:
        print(
            f"\n--- Processing Mode: {mode.upper()} | Style: {args.style.upper()} ---"
        )

        stock_csv = f"output/stock_benchmark_{mode}.csv"
        shared_csv = f"output/shared_benchmark_{mode}.csv"

        df_stock, df_shared = None, None
        if os.path.exists(stock_csv):
            try:
                df_stock = pd.read_csv(stock_csv)
            except pd.errors.EmptyDataError:
                print(f"Warning: File is empty: '{stock_csv}'")
        if os.path.exists(shared_csv):
            try:
                df_shared = pd.read_csv(shared_csv)
            except pd.errors.EmptyDataError:
                print(f"Warning: File is empty: '{shared_csv}'")

        if df_stock is None and df_shared is None:
            print(f"No valid data files found for mode '{mode}'. Skipping.")
            continue

        # --- PLOT GENERATION ---

        if args.style == "separate":
            # Create subdirectories for clarity
            output_dir_stock = f"plots_{mode}/stock"
            output_dir_shared = f"plots_{mode}/shared"
            os.makedirs(output_dir_stock, exist_ok=True)
            os.makedirs(output_dir_shared, exist_ok=True)

            # Plot individual stock results
            if df_stock is not None:
                df_plot, x_col, y_col, x_label, y_label, title_prefix, kernel_col = (
                    get_plot_params(df_stock, mode)
                )
                for kernel in df_plot[kernel_col].unique():
                    kernel_df = df_plot[df_plot[kernel_col] == kernel]
                    title = f"{title_prefix} for {kernel} Kernel (Global Memory)"
                    filename = os.path.join(
                        output_dir_stock, f"{kernel}_{mode}_stock.png"
                    )
                    create_plot(
                        kernel_df, x_col, y_col, x_label, y_label, title, filename, mode
                    )
                # Plot combined stock results
                title = f"Combined {title_prefix} (Global Memory)"
                filename = os.path.join(output_dir_stock, f"combined_{mode}_stock.png")
                create_plot(
                    df_plot,
                    x_col,
                    y_col,
                    x_label,
                    y_label,
                    title,
                    filename,
                    mode,
                    hue_col=kernel_col,
                )

            # Plot individual shared results
            if df_shared is not None:
                df_plot, x_col, y_col, x_label, y_label, title_prefix, kernel_col = (
                    get_plot_params(df_shared, mode)
                )
                for kernel in df_plot[kernel_col].unique():
                    kernel_df = df_plot[df_plot[kernel_col] == kernel]
                    title = f"{title_prefix} for {kernel} Kernel (Shared Memory)"
                    filename = os.path.join(
                        output_dir_shared, f"{kernel}_{mode}_shared.png"
                    )
                    create_plot(
                        kernel_df, x_col, y_col, x_label, y_label, title, filename, mode
                    )
                # Plot combined shared results
                title = f"Combined {title_prefix} (Shared Memory)"
                filename = os.path.join(
                    output_dir_shared, f"combined_{mode}_shared.png"
                )
                create_plot(
                    df_plot,
                    x_col,
                    y_col,
                    x_label,
                    y_label,
                    title,
                    filename,
                    mode,
                    hue_col=kernel_col,
                )

        elif args.style == "combined":
            if df_stock is None or df_shared is None:
                print(
                    f"Warning: Both stock and shared CSVs needed for 'combined' style. Skipping {mode}."
                )
                continue

            output_dir = f"plots_comparison/{mode}"
            os.makedirs(output_dir, exist_ok=True)

            df_combined = pd.concat([df_stock, df_shared], ignore_index=True)
            df_plot, x_col, y_col, x_label, y_label, title_prefix, kernel_col = (
                get_plot_params(df_combined, mode)
            )

            # Plot comparison for each kernel
            for kernel in df_plot[kernel_col].unique():
                kernel_df = df_plot[df_plot[kernel_col] == kernel]
                title = f"Comparison: {title_prefix} for {kernel} Kernel"
                filename = os.path.join(output_dir, f"comparison_{mode}_{kernel}.png")
                create_plot(
                    kernel_df,
                    x_col,
                    y_col,
                    x_label,
                    y_label,
                    title,
                    filename,
                    mode,
                    style_col="Optimization",
                )

            # Plot master comparison with everything
            title = f"Master Comparison: {title_prefix}"
            filename = os.path.join(output_dir, f"master_comparison_{mode}.png")
            create_plot(
                df_plot,
                x_col,
                y_col,
                x_label,
                y_label,
                title,
                filename,
                mode,
                hue_col=kernel_col,
                style_col="Optimization",
            )
