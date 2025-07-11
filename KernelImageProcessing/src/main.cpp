#include "CpuConvolution.h"
#include "GpuConvolution.h"
#include "Kernels.h"
#include <algorithm>
#include <chrono>
#include <cuda_runtime.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

const int NUM_AVERAGING_RUNS = 30;

struct Test {
  std::string name;
  cv::Mat kern;
};

struct ResolutionDir {
  std::filesystem::path path;
  std::string name;
  long total_pixels;
  int width;
  int height;
  bool operator<(const ResolutionDir &other) const {
    return total_pixels < other.total_pixels;
  }
};

// Forward declarations
void run_scaling_benchmark(std::vector<ResolutionDir> &resolution_dirs,
                           const std::vector<Test> &tests);
void run_throughput_benchmark(std::vector<ResolutionDir> &resolution_dirs,
                              const std::vector<Test> &tests);
void run_blocksize_benchmark(const std::vector<ResolutionDir> &resolution_dirs,
                             const std::vector<Test> &tests);

// ===================================================================
//                        CORE TIMING UTILITIES
// ===================================================================

double timeCpu(const cv::Mat &img, const cv::Mat &kern) {
  CpuConvolution conv(kern);
  auto start = std::chrono::high_resolution_clock::now();
  conv.apply(img);
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration<double>(end - start).count();
}

double timeGpu(GpuConvolution &conv, const cv::Mat &img, const dim3 &blockDim,
               int maxGridDimX = 0) {
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  conv.apply(img, blockDim, maxGridDimX);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  return milliseconds / 1000.0;
}

// ===================================================================
//                          MAIN DISPATCHER
// ===================================================================

int main(int argc, char **argv) {
  if (argc < 3) {
    std::cerr << "Usage: KernelApp <mode> <root_image_directory>\n";
    std::cerr << "Modes:\n";
    std::cerr << "  --scaling      : Finds highest resolution subdir and runs "
                 "scaling tests.\n";
    std::cerr << "  --throughput   : Iterates through all subdirs and runs "
                 "throughput tests.\n";
    std::cerr << "  --blocksize    : Finds highest resolution subdir and tests "
                 "different block sizes.\n";
    return 1;
  }

  std::string mode = argv[1];
  std::filesystem::path root_image_dir(argv[2]);

  if (!std::filesystem::is_directory(root_image_dir)) {
    std::cerr << "Error: Provided path is not a directory: " << argv[2] << "\n";
    return 1;
  }

  std::filesystem::create_directories("output");

  std::vector<Test> tests = {
      {"PrewittX", cv::Mat(3, 3, CV_32F, Kernels::PrewittX().data())},
      {"Gauss5x5", cv::Mat(5, 5, CV_32F, Kernels::Gaussian5x5().data())},
      {"Gauss7x7", cv::Mat(7, 7, CV_32F, Kernels::Gaussian7x7().data())}};

  std::cout << "Discovering image resolutions from subdirectories...\n";
  std::vector<ResolutionDir> discovered_dirs;
  for (const auto &entry :
       std::filesystem::directory_iterator(root_image_dir)) {
    if (entry.is_directory()) {
      cv::Mat first_image;
      for (const auto &img_entry :
           std::filesystem::directory_iterator(entry.path())) {
        if (img_entry.is_regular_file()) {
          first_image = cv::imread(img_entry.path().string());
          if (!first_image.empty())
            break;
        }
      }
      if (!first_image.empty()) {
        int w = first_image.cols;
        int h = first_image.rows;
        discovered_dirs.push_back({entry.path(),
                                   entry.path().filename().string(),
                                   (long)w * h, w, h});
        std::cout << "  - Found '" << entry.path().filename().string()
                  << "' with resolution " << w << "x" << h << "\n";
      }
    }
  }

  if (discovered_dirs.empty()) {
    std::cerr << "Error: No directories with valid images found in " << argv[2]
              << "\n";
    return 1;
  }

  if (mode == "--scaling") {
    run_scaling_benchmark(discovered_dirs, tests);
  } else if (mode == "--throughput") {
    run_throughput_benchmark(discovered_dirs, tests);
  } else if (mode == "--blocksize") {
    run_blocksize_benchmark(discovered_dirs, tests);
  } else {
    std::cerr << "Unknown mode: " << mode << "\n";
    return 1;
  }

  return 0;
}

// ===================================================================
//                  BENCHMARK IMPLEMENTATIONS
// ===================================================================

void run_scaling_benchmark(std::vector<ResolutionDir> &resolution_dirs,
                           const std::vector<Test> &tests) {
  std::cout << "\n--- Running GPU Scaling Benchmark ---\n";
  std::sort(resolution_dirs.begin(), resolution_dirs.end());
  const ResolutionDir &highest_res_dir = resolution_dirs.back();
  std::cout << "Using highest resolution directory for test: '"
            << highest_res_dir.name << "'\n\n";

  cv::Mat base_image;
  for (const auto &entry :
       std::filesystem::directory_iterator(highest_res_dir.path)) {
    if (entry.is_regular_file()) {
      base_image = cv::imread(entry.path().string(), cv::IMREAD_COLOR);
      if (!base_image.empty())
        break;
    }
  }
  if (base_image.empty())
    return;

  std::ofstream csv("output/benchmark_scaling.csv");
  csv << "Resolution,Kernel,AvgCPUTime,AvgGPUTime,UtilizationFraction,"
         "ActiveGridDimX,Speedup\n";
  std::vector<double> scaling_fractions = {0.05, 0.1, 0.25, 0.5, 0.75, 1.0};
  dim3 defaultBlockDim(16, 16);

  for (const auto &t : tests) {
    std::cout << "Benchmarking Kernel: " << t.name << "\n";
    cv::Mat imageToProcess;
    if (t.name.find("Prewitt") != std::string::npos)
      cv::cvtColor(base_image, imageToProcess, cv::COLOR_BGR2GRAY);
    else
      imageToProcess = base_image;

    double total_cpu_time = 0.0;
    for (int i = 0; i < NUM_AVERAGING_RUNS; ++i)
      total_cpu_time += timeCpu(imageToProcess, t.kern);
    double avg_cpu_time = total_cpu_time / NUM_AVERAGING_RUNS;

    GpuConvolution gpu_conv(t.kern);
    gpu_conv.apply(imageToProcess, defaultBlockDim, 1);
    cudaDeviceSynchronize();

    int fullGridDimX =
        (imageToProcess.cols + defaultBlockDim.x - 1) / defaultBlockDim.x;
    for (double fraction : scaling_fractions) {
      int activeGridDimX =
          std::max(1, static_cast<int>(fullGridDimX * fraction));
      double total_gpu_time = 0.0;
      for (int i = 0; i < NUM_AVERAGING_RUNS; ++i)
        total_gpu_time +=
            timeGpu(gpu_conv, imageToProcess, defaultBlockDim, activeGridDimX);
      double avg_gpu_time = total_gpu_time / NUM_AVERAGING_RUNS;
      double speedup = (avg_gpu_time > 0) ? (avg_cpu_time / avg_gpu_time) : 0;

      csv << highest_res_dir.name << "," << t.name << "," << avg_cpu_time << ","
          << avg_gpu_time << "," << fraction << "," << activeGridDimX << ","
          << speedup << "\n";
    }
  }
  csv.close();
}

void run_throughput_benchmark(std::vector<ResolutionDir> &resolution_dirs,
                              const std::vector<Test> &tests) {
  std::cout << "\n--- Running Throughput Benchmark ---\n";
  std::ofstream csv("output/benchmark_throughput.csv");
  csv << "ResolutionName,Resolution,Kernel,AvgCPUTime,AvgGPUTime,Speedup\n";
  dim3 defaultBlockDim(16, 16);

  for (const auto &dir : resolution_dirs) {
    std::cout << "Processing Directory: '" << dir.name << "'\n";
    cv::Mat base_image;
    for (const auto &entry : std::filesystem::directory_iterator(dir.path)) {
      if (entry.is_regular_file()) {
        base_image = cv::imread(entry.path().string(), cv::IMREAD_COLOR);
        if (!base_image.empty())
          break;
      }
    }
    if (base_image.empty())
      continue;

    for (const auto &t : tests) {
      cv::Mat imageToProcess;
      if (t.name.find("Prewitt") != std::string::npos)
        cv::cvtColor(base_image, imageToProcess, cv::COLOR_BGR2GRAY);
      else
        imageToProcess = base_image;

      double total_cpu_time = 0.0;
      for (int i = 0; i < NUM_AVERAGING_RUNS; ++i)
        total_cpu_time += timeCpu(imageToProcess, t.kern);
      double avg_cpu_time = total_cpu_time / NUM_AVERAGING_RUNS;

      GpuConvolution gpu_conv(t.kern);
      gpu_conv.apply(imageToProcess, defaultBlockDim, 1);
      cudaDeviceSynchronize();

      double total_gpu_time = 0.0;
      for (int i = 0; i < NUM_AVERAGING_RUNS; ++i)
        total_gpu_time += timeGpu(gpu_conv, imageToProcess, defaultBlockDim, 0);
      double avg_gpu_time = total_gpu_time / NUM_AVERAGING_RUNS;
      double speedup = (avg_gpu_time > 0) ? avg_cpu_time / avg_gpu_time : 0;

      csv << dir.name << "," << dir.width << "x" << dir.height << "," << t.name
          << "," << avg_cpu_time << "," << avg_gpu_time << "," << speedup
          << "\n";
    }
  }
  csv.close();
}

void run_blocksize_benchmark(const std::vector<ResolutionDir> &resolution_dirs,
                             const std::vector<Test> &tests) {
  std::cout << "\n--- Running Block Size Benchmark ---\n";

  auto dirs_copy = resolution_dirs;
  std::sort(dirs_copy.begin(), dirs_copy.end());
  const ResolutionDir &highest_res_dir = dirs_copy.back();
  std::cout << "Using highest resolution directory for test: '"
            << highest_res_dir.name << "'\n\n";

  cv::Mat base_image;
  for (const auto &entry :
       std::filesystem::directory_iterator(highest_res_dir.path)) {
    if (entry.is_regular_file()) {
      base_image = cv::imread(entry.path().string(), cv::IMREAD_COLOR);
      if (!base_image.empty())
        break;
    }
  }
  if (base_image.empty())
    return;

  std::ofstream csv("output/benchmark_blocksize.csv");
  csv << "Resolution,Kernel,BlockSize,ThreadsPerBlock,AvgCPUTime,AvgGPUTime,"
         "Speedup\n";

  std::vector<dim3> block_sizes = {dim3(8, 8),  dim3(16, 8),  dim3(16, 16),
                                   dim3(32, 8), dim3(32, 16), dim3(32, 32)};

  for (const auto &t : tests) {
    std::cout << "Benchmarking Kernel: " << t.name << "\n";
    cv::Mat imageToProcess;
    if (t.name.find("Prewitt") != std::string::npos)
      cv::cvtColor(base_image, imageToProcess, cv::COLOR_BGR2GRAY);
    else
      imageToProcess = base_image;

    double total_cpu_time = 0.0;
    for (int i = 0; i < NUM_AVERAGING_RUNS; ++i)
      total_cpu_time += timeCpu(imageToProcess, t.kern);
    double avg_cpu_time = total_cpu_time / NUM_AVERAGING_RUNS;

    GpuConvolution gpu_conv(t.kern);
    gpu_conv.apply(imageToProcess, dim3(8, 8), 1);
    cudaDeviceSynchronize();

    for (const auto &block_dim : block_sizes) {
      double total_gpu_time = 0.0;
      for (int i = 0; i < NUM_AVERAGING_RUNS; ++i) {
        total_gpu_time += timeGpu(gpu_conv, imageToProcess, block_dim, 0);
      }
      double avg_gpu_time = total_gpu_time / NUM_AVERAGING_RUNS;
      double speedup = avg_gpu_time > 0 ? avg_cpu_time / avg_gpu_time : 0;
      std::string block_str =
          std::to_string(block_dim.x) + "x" + std::to_string(block_dim.y);
      unsigned int threads_per_block = block_dim.x * block_dim.y * block_dim.z;

      std::cout << "  Block Size: " << block_str
                << " | Avg. GPU Time: " << avg_gpu_time
                << "s | Speedup: " << speedup << "x\n";
      csv << highest_res_dir.name << "," << t.name << "," << block_str << ","
          << threads_per_block << "," << avg_cpu_time << "," << avg_gpu_time
          << "," << speedup << "\n";
    }
    std::cout << "\n";
  }
  csv.close();
}
