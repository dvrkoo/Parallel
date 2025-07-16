// BenchmarkSuite.h

#ifndef BENCHMARK_SUITE_H
#define BENCHMARK_SUITE_H

#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

// Forward declare the GpuConvolution class to avoid including its full header
// here. This is a good practice to reduce compilation dependencies.
class GpuConvolution;

// A simple struct to hold the definition of a single test case.
struct Test {
  std::string name;
  cv::Mat kern;
};

// A class to encapsulate the logic for running a single test configuration.
class BenchmarkRunner {
public:
  // The result of a single benchmark run (averaged over NUM_TIMING_RUNS).
  struct Result {
    double avg_cpu_time = 0.0;
    double avg_gpu_time = 0.0;
    double speedup = 0.0;
  };

  // Static method to run a test and return the result.
  static Result run(const cv::Mat &image, const Test &test,
                    const dim3 &block_dim, bool use_shared);

private:
  // Helper to prepare the image (e.g., convert to grayscale if needed).
  static cv::Mat prepare_image(const cv::Mat &base_image, const Test &test);
};

// The main class that orchestrates the entire benchmark process.
class BenchmarkSuite {
public:
  BenchmarkSuite(int argc, char **argv);
  void run();

private:
  void parse_args(int argc, char **argv);
  void define_tests();
  cv::Mat getOrGenerateImage(int width, int height,
                             const std::string &name_suffix = "");

  // The individual benchmark modes.
  void run_throughput();
  void run_blocksize();
  void run_kernelsize();

  std::string mode_;
  bool use_shared_mem_ = false;
  std::vector<Test> tests_;
};

#endif // BENCHMARK_SUITE_H
