import argparse
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import os


def plot_3d_clusters(n_samples, k, mode, threads=None, save_fig=False, no_show=False):
    """
    Loads data, centroids, and assignments to generate a 3D plot of the clustering.

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
    # Original data points (has a header)
    points_df = pd.read_csv(data_file)
    # Centroids (no header)
    centroids_df = pd.read_csv(centroids_file, header=None)
    # Assignments (no header, single column)
    assignments_df = pd.read_csv(assignments_file, header=None)

    # Convert to NumPy arrays for plotting
    points = points_df.values
    centroids = centroids_df.values
    assignments = assignments_df[
        0
    ].values  # Get the first (and only) column as a 1D array

    # --- 4. Create the 3D Plot ---
    print("Generating 3D plot...")
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")

    # Plot the data points, colored by their cluster assignment
    # The 'c=assignments' argument tells matplotlib to use the cluster IDs for coloring
    scatter_points = ax.scatter(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        c=assignments,
        cmap="viridis",  # A nice color map
        s=10,  # Size of the points
        alpha=0.6,  # Transparency for dense plots
        label="Data Points",
    )

    # Plot the centroids, making them larger and distinct
    ax.scatter(
        centroids[:, 0],
        centroids[:, 1],
        centroids[:, 2],
        c="red",  # A single, distinct color
        marker="X",  # A different marker shape
        s=250,  # Much larger size
        edgecolor="black",  # Black edge for visibility
        label="Centroids",
    )

    # --- 5. Finalize and Show/Save Plot ---
    title = f"K-Means Clustering (n={n_samples}, k={k}, mode='{mode}'"
    if threads:
        title += f", threads={threads})"
    else:
        title += ")"

    ax.set_title(title, fontsize=16)
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.set_zlabel("Feature 3")
    ax.legend()
    ax.grid(True)

    if save_fig:
        output_filename = f"results/plots/plot_n{n_samples}_k{k}_{mode}"
        if threads:
            output_filename += f"_t{threads}"
        output_filename += ".png"

        plt.savefig(output_filename, dpi=300)
        print(f"Plot saved to {output_filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot K-Means clustering results in 3D."
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
