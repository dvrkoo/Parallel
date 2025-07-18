# CUDA Image Convolution Benchmark Suite

This repository contains a C++ and CUDA-based benchmark suite for analyzing the performance of 2D image convolution on the CPU versus the GPU. The project demonstrates a systematic approach to GPU optimization, starting from a naive parallel implementation and progressing to a highly-tuned kernel using shared memory and empirically determined launch parameters.

The suite is designed to be a comprehensive tool for performance analysis, capable of generating reproducible test data, running multiple benchmark scenarios, and producing clear, plottable CSV results.

## Features

- **CPU vs. GPU Performance:** Directly compares a sequential CPU implementation against a parallel CUDA implementation.
- **Two GPU Kernel Implementations:**
  - **Naive Global Memory:** A straightforward kernel where each thread reads directly from global memory.
  - **Optimized Shared Memory:** A high-performance kernel using the tiling technique to maximize data reuse and memory bandwidth.
- **Multiple Benchmark Modes:**
  - `throughput`: Measures speedup across various image resolutions.
  - `kernelsize`: Analyzes performance as the convolution kernel size increases.
  - `blocksize`: Determines the optimal CUDA thread block dimensions for each filter.
- **Reproducible Data Generation:** Automatically generates and caches consistent test images, ensuring scientific reproducibility.
- **Flexible Controls:** Command-line flags (`--shared`, `--tuned`) allow for precise control over which optimization strategies to enable for any given benchmark run.
- **Automated Plotting:** A companion Python script (`plot.py`) reads the generated CSV files and produces professional, publication-quality plots to visualize the results.

## Getting Started

### Prerequisites

- A CUDA-enabled NVIDIA GPU
- NVIDIA CUDA Toolkit (nvcc compiler)
- CMake (version 3.10 or higher)
- OpenCV (version 4.x recommended)
- Python 3.x with `pandas`, `matplotlib`, and `seaborn` for plotting:
  ```bash
  pip install pandas matplotlib seaborn
  ```

### Building the Project

The project uses a standard CMake build process.

1.  **Clone the repository:**

    ```bash
    git clone <https://github.com/dvrkoo/Parallel/>
    cd <repo-directory/KernelImageProcessing>
    ```

2.  **Create a build directory:**

    ```bash
    mkdir build
    cd build
    ```

3.  **Run CMake and build:**
    ```bash
    cmake ..
    make
    ```
    This will create the executable `KernelApp` in the `build` directory.

## Running the Benchmarks

The `KernelApp` executable is controlled via command-line arguments. The general syntax is:

`./KernelApp [flags] <mode>`

### Benchmark Modes

- `--throughput`: (Most common) Benchmarks performance across multiple image sizes (512x512 to 4096x4096).
- `--kernelsize`: Benchmarks performance for different kernel sizes (3x3 to 21x21) on a fixed 1440p image.
- `--blocksize`: Benchmarks performance across various thread block dimensions to find the optimal configuration.

### Optional Flags

- `--shared`: Enables the optimized shared memory CUDA kernel. If omitted, the naive global memory kernel is used.
- `--tuned`: (Only for `--throughput` mode) Uses the empirically determined optimal block size for each filter instead of the default 16x16.

### Example Workflow

This workflow will generate all the data needed for a full analysis.

1.  **Run Stock (Global Memory) Benchmarks:**

    ```bash
    # Throughput with fixed 16x16 blocks
    ./KernelApp --throughput
    # -> output/stock_fixed_benchmark_throughput.csv

    # Blocksize analysis for the global kernel
    ./KernelApp --blocksize
    # -> output/stock_benchmark_blocksize.csv
    ```

2.  **Run Optimized (Shared Memory) Benchmarks:**

    ```bash
    # Throughput with fixed 16x16 blocks
    ./KernelApp --throughput --shared
    # -> output/shared_fixed_benchmark_throughput.csv

    # Throughput with tuned block sizes (peak performance)
    ./KernelApp --throughput --shared --tuned
    # -> output/shared_tuned_benchmark_throughput.csv
    ```

**Note:** The first time you run a benchmark, the program will generate and save test images to a `generated_images/` directory. Subsequent runs will load these cached images for consistency.

## Visualizing the Results

The `plot.py` script automatically finds the relevant CSV files and generates comparison plots.

### Usage

`python plot.py <mode>`

### Plotting Modes

- `throughput`: Generates a comparison of Global vs. Shared memory speedup across different image resolutions.
- `blocksize`: Generates a comparison of performance for different block sizes.
- `kernelsize`: Generates a comparison of performance for different filter sizes.
- `all`: Runs all of the above plotting modes.

### Example

To generate the key throughput comparison plot after running the C++ benchmarks:

```bash
python plot.py throughput
```
