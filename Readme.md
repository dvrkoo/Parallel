# C++ Parallel Computing Projects

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains a collection of C++ projects focused on implementing and analyzing parallel algorithms. Each project explores a different computational problem and compares the performance of a sequential implementation against a parallel version using OpenMP for the KMeans and using Cuda for the KernelImageProcessing.

## Repository Content

This repository contains two main exercises:

### 1. K-Means Clustering

- **Folder:** `KMeans/`
- **Description:** An implementation of the K-Means clustering algorithm. This project analyzes the performance trade-offs between sequential and parallel execution based on dataset size and the number of clusters.
- **[➡️ Go to the K-Means README for full details](./KMeans/Readme.md)**

### 2. Kernel-Based Image Processing

- **Folder:** `KernelImageProcessing/`
- **Description:** Applies common image processing filters (convolution kernels) like blur and edge detection. This project demonstrates the speedup achieved by parallelizing per-pixel operations on large images.
- **[➡️ Go to the Image Processing README for full details](./KernelImageProcessing/Readme.md)**

## Getting Started

Each project is self-contained and includes its own `CMakeLists.txt` for building, along with a detailed `README.md` explaining its specific usage and findings.

To build and run an exercise:

1.  Navigate into the project directory:

    ```sh
    cd KMeans
    # or
    cd KernelImageProcessing
    ```

2.  Follow the instructions in that project's specific README.md file.

## Core Technologies

- Language: C++ (17 or later)
- Parallelism: OpenMP, CUDA
- Build Systems: Cmake
