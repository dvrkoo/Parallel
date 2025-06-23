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

using Clock = std::chrono::high_resolution_clock;

// Helper function to time CPU convolution
double timeSeq(const cv::Mat &img, const cv::Mat &kern) {
  CpuConvolution conv(kern);
  auto start = Clock::now();
  conv.apply(img);
  return std::chrono::duration<double>(Clock::now() - start).count();
}

// Helper function to time GPU convolution
double timeGpu(const cv::Mat &img, const cv::Mat &kern) {
  GpuConvolution conv(kern);
  auto start = Clock::now();
  conv.apply(img);
  return std::chrono::duration<double>(Clock::now() - start).count();
}

int main(int argc, char **argv) {
  if (argc < 2) {
    std::cerr << "Usage: KernelApp <image_path>\n";
    return 1;
  }

  // --- 1. Setup Environment ---
  std::filesystem::create_directories("output");
  std::ofstream csv("output/benchmark.csv");
  csv << "ImageSize,Kernel,CPUTime,GPUTime,Speedup,TotalCores\n";

  // Query GPU properties
  int devCount = 0;
  cudaGetDeviceCount(&devCount);
  int totalCores = 0;
  if (devCount > 0) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int coresPerSM = 128; // Adjust for GPU architecture (e.g., 128 for Ampere)
    if (prop.major < 8)
      coresPerSM = 64;
    totalCores = prop.multiProcessorCount * coresPerSM;
    std::cout << "GPU Device: " << prop.name
              << " | SMs=" << prop.multiProcessorCount
              << " | TotalCores~=" << totalCores << "\n";
  }

  // --- 2. Load Input Image ---
  cv::Mat img = cv::imread(argv[1], cv::IMREAD_COLOR);
  if (img.empty()) {
    std::cerr << "Failed to load image: " << argv[1] << "\n";
    return 1;
  }

  // --- 3. Define Kernels and Image Sizes for Testing ---
  // Store the kernel data in local variables to ensure their lifetime.
  auto prewittX_data = Kernels::PrewittX();
  auto prewittY_data = Kernels::PrewittY();
  auto gauss5_data = Kernels::Gaussian5x5();
  auto gauss7_data = Kernels::Gaussian7x7();

  struct Test {
    std::string name;
    cv::Mat kern;
  };

  std::vector<Test> tests = {
      {"PrewittX_Vertical", cv::Mat(3, 3, CV_32F, prewittX_data.data())},
      {"PrewittY_Horizontal", cv::Mat(3, 3, CV_32F, prewittY_data.data())},
      {"Gauss5x5", cv::Mat(5, 5, CV_32F, gauss5_data.data())},
      {"Gauss7x7", cv::Mat(7, 7, CV_32F, gauss7_data.data())}};

  std::vector<int> sizes = {256, 512, 1024, 2048, 4096};

  // --- 4. Main Benchmark Loop ---
  for (int s : sizes) {
    cv::Mat rimg_color;
    cv::resize(img, rimg_color, cv::Size(s, s), 0, 0, cv::INTER_AREA);
    std::cout << "Benchmarking " << s << "x" << s << " images...\n";

    for (const auto &t : tests) {
      cv::Mat imageToProcess;
      bool isEdgeFilter = (t.name.find("Prewitt") != std::string::npos);

      // Prepare the correct input format for the filter
      if (isEdgeFilter) {
        cv::cvtColor(rimg_color, imageToProcess, cv::COLOR_BGR2GRAY);
      } else {
        imageToProcess = rimg_color;
      }

      // Benchmark performance
      double t_cpu = timeSeq(imageToProcess, t.kern);
      double t_gpu = timeGpu(imageToProcess, t.kern);
      double speedup = (t_gpu > 0) ? (t_cpu / t_gpu) : 0;
      std::cout << "  " << t.name << " | CPU: " << t_cpu << "s"
                << " | GPU: " << t_gpu << "s"
                << " | Speedup: " << speedup << "x\n";

      csv << s << "," << t.name << "," << t_cpu << "," << t_gpu << ","
          << speedup << "," << totalCores << "\n";

      // --- 5. Generate and Save Output Images with Correct Visualization ---
      CpuConvolution seq(t.kern);
      GpuConvolution gpu(t.kern);
      cv::Mat outSeq_float = seq.apply(imageToProcess);
      cv::Mat outGpu_float = gpu.apply(imageToProcess);

      cv::Mat outSeq_vis, outGpu_vis;

      // Use different visualization strategies based on the filter type
      if (isEdgeFilter) {
        // For Prewitt, use convertScaleAbs for high-contrast magnitude.
        // This takes the absolute value of the gradients and scales them,
        // resulting in a clear black-and-white edge map.
        cv::convertScaleAbs(outSeq_float, outSeq_vis);
        cv::convertScaleAbs(outGpu_float, outGpu_vis);
      } else {
        // For Gaussian blur, normalize to preserve the color information.
        cv::normalize(outSeq_float, outSeq_vis, 0, 255, cv::NORM_MINMAX,
                      CV_8UC(imageToProcess.channels()));
        cv::normalize(outGpu_float, outGpu_vis, 0, 255, cv::NORM_MINMAX,
                      CV_8UC(imageToProcess.channels()));
      }

      std::string cpu_path =
          "output/" + t.name + "_" + std::to_string(s) + "_cpu.png";
      std::string gpu_path =
          "output/" + t.name + "_" + std::to_string(s) + "_gpu.png";
      cv::imwrite(cpu_path, outSeq_vis);
      cv::imwrite(gpu_path, outGpu_vis);
    }
  }

  csv.close();
  std::cout << "Benchmark data and images saved to 'output/'\n";
  return 0;
}
