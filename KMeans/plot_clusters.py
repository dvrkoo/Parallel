# plot_clusters.py

import argparse
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import os


def plot_3d_clusters(n_samples, k, mode, threads=None, save_fig=False, no_show=False):
    """
    Loads data, centroids, and assignments to generate a 3D plot of the clustering
    from multiple viewing angles.

    Args:
        n_samples (int): The number of data samples used.
        k (int): The number of clusters (k) used.
        mode (str): The execution mode, either 'seq' or 'par'.
        threads (int, optional): The number of threads used, required if mode is 'par'.
        save_fig (bool): If True, saves the plot to a file.
        no_show (bool): If True, does not display the plot window.
    """
    print(
        f"--- Plotting for n={n_samples}, k={k}, mode='{mode}'"
        + (f", threads={threads}" if threads else "")
        + " ---"
    )

    # --- 1. Construct File Paths ---
    data_file = f"data/{n_samples}_3.csv"

    if mode == "seq":
        base_name = f"results/plots/centroids_seq_n{n_samples}_k{k}.csv"
        assignments_file = f"results/plots/assignments_seq_n{n_samples}_k{k}.csv"
    elif mode == "par":
        if threads is None:
            print("Error: --threads must be specified for parallel mode ('par').")
            sys.exit(1)
        base_name = f"results/plots/centroids_par_n{n_samples}_k{k}_t{threads}.csv"
        assignments_file = (
            f"results/plots/assignments_par_n{n_samples}_k{k}_t{threads}.csv"
        )
    else:
        print(f"Error: Invalid mode '{mode}'. Choose 'seq' or 'par'.")
        sys.exit(1)

    centroids_file = base_name

    # --- 2. Check if files exist ---
    required_files = [data_file, centroids_file, assignments_file]
    for f in required_files:
        if not os.path.exists(f):
            print(f"Error: Required file not found at '{f}'")
            print(
                "Please ensure you have run the C++ program to generate the result files."
            )
            sys.exit(1)

    # --- 3. Load Data using Pandas ---
    print("Loading data files...")
    points_df = pd.read_csv(data_file)
    centroids_df = pd.read_csv(centroids_file, header=None)
    assignments_df = pd.read_csv(assignments_file, header=None)

    points = points_df.values
    centroids = centroids_df.values
    assignments = assignments_df[0].values

    # --- 4. Create the Multi-Angle 3D Plot ---
    print("Generating multi-angle 3D plot...")

    # Define the different camera angles we want to see
    views = [
        {"elev": 30, "azim": -60, "title": "Perspective 1"},
        {"elev": 90, "azim": 0, "title": "Top-Down (X-Y Plane)"},
        {"elev": 0, "azim": 0, "title": "Front View (X-Z Plane)"},
        {"elev": 0, "azim": -90, "title": "Side View (Y-Z Plane)"},
    ]

    # Create a figure with a 2x2 grid of 3D subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 14), subplot_kw={"projection": "3d"})

    # Set a main title for the entire figure
    main_title = f"K-Means Clustering (n={n_samples}, k={k}, mode='{mode}'"
    if threads:
        main_title += f", threads={threads})"
    else:
        main_title += ")"
    fig.suptitle(main_title, fontsize=20)

    # Loop through each subplot and view angle
    for ax, view in zip(axes.flat, views):
        # Plot the data points
        ax.scatter(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            c=assignments,
            cmap="viridis",
            s=10,
            alpha=0.5,
            zorder=1,  # **CHANGE**: Render points at a lower layer
        )

        # Plot the centroids
        ax.scatter(
            centroids[:, 0],
            centroids[:, 1],
            centroids[:, 2],
            c="red",
            marker="X",
            s=250,
            edgecolor="black",
            label="Centroids",
            depthshade=False,  # Make sure the color is pure red
            zorder=10,  # **CHANGE**: Render centroids on the top layer
        )

        # Set the camera angle for this specific subplot
        ax.view_init(elev=view["elev"], azim=view["azim"])

        # Set labels and title for the subplot
        ax.set_title(view["title"], fontsize=14)
        ax.set_xlabel("Feature 1")
        ax.set_ylabel("Feature 2")
        ax.set_zlabel("Feature 3")
        ax.legend()
        ax.grid(True)

    # --- 5. Finalize and Show/Save Plot ---
    # Adjust layout to prevent titles from overlapping
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # rect makes space for suptitle

    if save_fig:
        output_filename = f"results/plots/plot_n{n_samples}_k{k}_{mode}"
        if threads:
            output_filename += f"_t{threads}"
        output_filename += ".png"

        plt.savefig(output_filename, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {output_filename}")

    if not no_show:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot K-Means clustering results in 3D from multiple angles."
    )
    parser.add_argument(
        "--n", type=int, required=True, help="Number of samples (e.g., 10000)."
    )
    parser.add_argument(
        "--k", type=int, required=True, help="Number of clusters (e.g., 5)."
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["seq", "par"],
        help="Mode of execution ('seq' or 'par').",
    )
    parser.add_argument(
        "--t", type=int, help="Number of threads (required for 'par' mode)."
    )
    parser.add_argument(
        "--save", action="store_true", help="Save the plot to a PNG file."
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not display the interactive plot window.",
    )

    args = parser.parse_args()

    plot_3d_clusters(
        n_samples=args.n,
        k=args.k,
        mode=args.mode,
        threads=args.t,
        save_fig=args.save,
        no_show=args.no_show,
    )
