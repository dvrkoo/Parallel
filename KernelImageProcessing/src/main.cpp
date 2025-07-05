#include "CpuConvolution.h"
#include "GpuConvolution.h"
#include "Kernels.h"
#include <chrono>
#include <cuda_runtime.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

// The number of times to run each test to get an average
const int NUM_AVERAGING_RUNS = 30;

// Define a structure to hold test information
struct Test {
  std::string name;
  cv::Mat kern;
};

// Forward declarations for our two benchmark functions
void run_scaling_benchmark(
    const std::vector<std::filesystem::path> &resolution_dirs,
    const std::vector<Test> &tests);
void run_throughput_benchmark(
    const std::vector<std::filesystem::path> &resolution_dirs,
    const std::vector<Test> &tests);

double timeCpu(const cv::Mat &img, const cv::Mat &kern) {
  CpuConvolution conv(kern);
  auto start = std::chrono::high_resolution_clock::now();
  conv.apply(img);
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration<double>(end - start).count();
}

double timeGpu(GpuConvolution &conv, const cv::Mat &img, int maxGridDimX = 0) {
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  conv.apply(img, maxGridDimX);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  return milliseconds / 1000.0;
}

int main(int argc, char **argv) {
  if (argc < 3) {
    std::cerr << "Usage: KernelApp <mode> <root_image_directory>\n";
    std::cerr << "       <root_image_directory> should contain subdirectories "
                 "named by resolution (e.g., '1920x1080').\n";
    std::cerr << "Modes:\n";
    std::cerr << "  --scaling      : Finds highest resolution subdir and runs "
                 "scaling tests.\n";
    std::cerr << "  --throughput   : Iterates through all resolution subdirs "
                 "and runs throughput tests.\n";
    return 1;
  }

  // --- 1. Argument Parsing and Setup ---
  std::string mode = argv[1];
  std::filesystem::path root_image_dir(argv[2]);

  if (!std::filesystem::is_directory(root_image_dir)) {
    std::cerr << "Error: Provided path is not a directory: " << argv[2] << "\n";
    return 1;
  }

  std::filesystem::create_directories("output");

  // --- 2. Define Kernels for Testing ---
  auto prewittX_data = Kernels::PrewittX();
  auto gauss5_data = Kernels::Gaussian5x5();
  auto gauss7_data = Kernels::Gaussian7x7();

  std::vector<Test> tests = {
      {"PrewittX", cv::Mat(3, 3, CV_32F, prewittX_data.data())},
      {"Gauss5x5", cv::Mat(5, 5, CV_32F, gauss5_data.data())},
      {"Gauss7x7", cv::Mat(7, 7, CV_32F, gauss7_data.data())}};

  // --- 3. Find all resolution subdirectories ---
  std::vector<std::filesystem::path> resolution_dirs;
  for (const auto &entry :
       std::filesystem::directory_iterator(root_image_dir)) {
    if (entry.is_directory()) {
      resolution_dirs.push_back(entry.path());
    }
  }

  if (resolution_dirs.empty()) {
    std::cerr << "Error: No resolution subdirectories found in " << argv[2]
              << "\n";
    return 1;
  }

  // --- 4. Dispatch to the Correct Benchmark Function ---
  if (mode == "--scaling") {
    run_scaling_benchmark(resolution_dirs, tests);
  } else if (mode == "--throughput") {
    run_throughput_benchmark(resolution_dirs, tests);
  } else {
    std::cerr << "Unknown mode: " << mode << "\n";
    return 1;
  }

  return 0;
}

/**
 * @brief Finds the highest resolution directory and runs scaling tests on an
 * image from it.
 */
void run_scaling_benchmark(
    const std::vector<std::filesystem::path> &resolution_dirs,
    const std::vector<Test> &tests) {
  std::cout << "\n--- Running GPU Scaling Benchmark ---\n";
  std::cout << "Finding highest resolution images to test architectural "
               "bottlenecks...\n";

  // Find the directory with the highest resolution
  std::filesystem::path highest_res_dir;
  long max_pixels = 0;
  for (const auto &dir : resolution_dirs) {
    std::string dirname = dir.filename().string();
    int w = 0, h = 0;
    if (sscanf(dirname.c_str(), "%dx%d", &w, &h) == 2) {
      if ((long)w * h > max_pixels) {
        max_pixels = (long)w * h;
        highest_res_dir = dir;
      }
    }
  }

  if (highest_res_dir.empty()) {
    std::cerr << "Error: Could not find any valid resolution-named "
                 "subdirectories (e.g., '1920x1080').\n";
    return;
  }

  std::cout << "Using highest resolution directory for test: "
            << highest_res_dir.filename().string() << "\n\n";

  // Find the first valid image in that directory to use for the benchmark
  cv::Mat base_image;
  for (const auto &entry :
       std::filesystem::directory_iterator(highest_res_dir)) {
    std::string ext = entry.path().extension().string();
    if (ext == ".jpg" || ext == ".png" || ext == ".bmp" || ext == ".jpeg") {
      base_image = cv::imread(entry.path().string(), cv::IMREAD_COLOR);
      if (!base_image.empty()) {
        std::cout << "Using image: " << entry.path().filename().string()
                  << "\n\n";
        break; // Found an image, stop searching
      }
    }
  }

  if (base_image.empty()) {
    std::cerr << "Error: No loadable images found in " << highest_res_dir
              << "\n";
    return;
  }

  std::ofstream csv("output/benchmark_scaling.csv");
  csv << "Resolution,Kernel,AvgCPUTime,AvgGPUTime,UtilizationFraction,"
         "ActiveGridDimX,Speedup\n";

  std::vector<double> scaling_fractions = {0.05, 0.1, 0.25, 0.5, 0.75, 1.0};

  for (const auto &t : tests) {
    std::cout << "Benchmarking Kernel: " << t.name << " (averaging "
              << NUM_AVERAGING_RUNS << " runs)\n";
    cv::Mat imageToProcess;
    if (t.name.find("Prewitt") != std::string::npos) {
      cv::cvtColor(base_image, imageToProcess, cv::COLOR_BGR2GRAY);
    } else {
      imageToProcess =
          base_image; // Use the image at its native high resolution
    }

    double total_cpu_time = 0.0;
    for (int i = 0; i < NUM_AVERAGING_RUNS; ++i)
      total_cpu_time += timeCpu(imageToProcess, t.kern);
    double avg_cpu_time = total_cpu_time / NUM_AVERAGING_RUNS;
    std::cout << "  Avg. CPU Time: " << avg_cpu_time << "s (for comparison)\n";

    GpuConvolution gpu_conv(t.kern);
    gpu_conv.apply(imageToProcess, 1); // Warm-up
    cudaDeviceSynchronize();

    int fullGridDimX = (imageToProcess.cols + 15) / 16;
    for (double fraction : scaling_fractions) {
      int activeGridDimX =
          std::max(1, static_cast<int>(fullGridDimX * fraction));
      double total_gpu_time = 0.0;
      for (int i = 0; i < NUM_AVERAGING_RUNS; ++i)
        total_gpu_time += timeGpu(gpu_conv, imageToProcess, activeGridDimX);
      double avg_gpu_time = total_gpu_time / NUM_AVERAGING_RUNS;
      double speedup = (avg_gpu_time > 0) ? (avg_cpu_time / avg_gpu_time) : 0;

      std::cout << "  GPU Util: " << (fraction * 100) << "%"
                << " | GridDimX: " << activeGridDimX
                << " | Avg. Time: " << avg_gpu_time << "s"
                << " | Speedup: " << speedup << "x\n";

      csv << highest_res_dir.filename().string() << "," << t.name << ","
          << avg_cpu_time << "," << avg_gpu_time << "," << fraction << ","
          << activeGridDimX << "," << speedup << "\n";
    }
    std::cout << "\n";
  }
  csv.close();
  std::cout
      << "Scaling benchmark data saved to 'output/benchmark_scaling.csv'\n";
}

/**
 * @brief Iterates through all resolution directories and runs throughput tests
 * at full GPU utilization.
 */
void run_throughput_benchmark(
    const std::vector<std::filesystem::path> &resolution_dirs,
    const std::vector<Test> &tests) {
  std::cout << "\n--- Running Throughput Benchmark ---\n";
  std::cout << "Analyzes speedup across all available resolutions at full GPU "
               "utilization.\n\n";
  std::ofstream csv("output/benchmark_throughput.csv");
  csv << "Resolution,Kernel,AvgCPUTime,AvgGPUTime,Speedup\n";

  for (const auto &dir : resolution_dirs) {
    std::string resolution_str = dir.filename().string();
    std::cout
        << "===========================================================\n";
    std::cout << "Processing Resolution: " << resolution_str << " (averaging "
              << NUM_AVERAGING_RUNS << " runs)\n";

    // Find the first valid image in this directory
    cv::Mat base_image;
    for (const auto &entry : std::filesystem::directory_iterator(dir)) {
      std::string ext = entry.path().extension().string();
      if (ext == ".jpg" || ext == ".png" || ext == ".bmp" || ext == ".jpeg") {
        base_image = cv::imread(entry.path().string(), cv::IMREAD_COLOR);
        if (!base_image.empty())
          break;
      }
    }

    if (base_image.empty()) {
      std::cerr << "Warning: No loadable images found in " << dir
                << ". Skipping.\n";
      continue;
    }

    for (const auto &t : tests) {
      cv::Mat imageToProcess;
      if (t.name.find("Prewitt") != std::string::npos) {
        cv::cvtColor(base_image, imageToProcess, cv::COLOR_BGR2GRAY);
      } else {
        imageToProcess = base_image;
      }

      double total_cpu_time = 0.0;
      for (int i = 0; i < NUM_AVERAGING_RUNS; ++i)
        total_cpu_time += timeCpu(imageToProcess, t.kern);
      double avg_cpu_time = total_cpu_time / NUM_AVERAGING_RUNS;

      GpuConvolution gpu_conv(t.kern);
      gpu_conv.apply(imageToProcess, 1); // Warm-up
      cudaDeviceSynchronize();

      double total_gpu_time = 0.0;
      for (int i = 0; i < NUM_AVERAGING_RUNS; ++i)
        total_gpu_time += timeGpu(gpu_conv, imageToProcess, 0);
      double avg_gpu_time = total_gpu_time / NUM_AVERAGING_RUNS;
      double speedup = (avg_gpu_time > 0) ? avg_cpu_time / avg_gpu_time : 0;

      std::cout << "  " << t.name << " | Avg. CPU: " << avg_cpu_time << "s"
                << " | Avg. GPU: " << avg_gpu_time << "s"
                << " | Speedup: " << speedup << "x\n";

      csv << resolution_str << "," << t.name << "," << avg_cpu_time << ","
          << avg_gpu_time << "," << speedup << "\n";
    }
    std::cout << "\n";
  }
  csv.close();
  std::cout << "Throughput benchmark data saved to "
               "'output/benchmark_throughput.csv'\n";
}
