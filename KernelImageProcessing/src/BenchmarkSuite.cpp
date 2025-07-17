#include "BenchmarkSuite.h"
#include "CpuConvolution.h"
#include "GpuConvolution.h"
#include "Kernels.h"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

// --- Configuration Constants ---
const int NUM_TIMING_RUNS = 5;
const int NUM_STATISTICAL_IMAGES = 3;

// ===================================================================
//                       Core Timing Utilities
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
//                   BenchmarkRunner Implementation
// ===================================================================

cv::Mat BenchmarkRunner::prepare_image(const cv::Mat &base_image,
                                       const Test &test) {
  if (test.name.find("Prewitt") != std::string::npos &&
      base_image.channels() > 1) {
    cv::Mat gray;
    cv::cvtColor(base_image, gray, cv::COLOR_BGR2GRAY);
    return gray;
  }
  return base_image;
}

BenchmarkRunner::Result BenchmarkRunner::run(const cv::Mat &image,
                                             const Test &test,
                                             const dim3 &block_dim,
                                             bool use_shared) {
  Result r;
  cv::Mat imageToProcess = prepare_image(image, test);

  for (int i = 0; i < NUM_TIMING_RUNS; ++i) {
    r.avg_cpu_time += timeCpu(imageToProcess, test.kern);
  }
  r.avg_cpu_time /= NUM_TIMING_RUNS;

  GpuConvolution gpu_conv(test.kern, use_shared);
  gpu_conv.apply(imageToProcess, block_dim, 1);
  cudaDeviceSynchronize();

  for (int i = 0; i < NUM_TIMING_RUNS; ++i) {
    r.avg_gpu_time += timeGpu(gpu_conv, imageToProcess, block_dim, 0);
  }
  r.avg_gpu_time /= NUM_TIMING_RUNS;

  r.speedup = (r.avg_gpu_time > 0) ? r.avg_cpu_time / r.avg_gpu_time : 0;
  return r;
}

// ===================================================================
//                   BenchmarkSuite Implementation
// ===================================================================

BenchmarkSuite::BenchmarkSuite(int argc, char **argv) {
  parse_args(argc, argv);
  define_tests();
}

void BenchmarkSuite::run() {
  std::filesystem::create_directories("output");
  if (mode_ == "--throughput")
    run_throughput();
  else if (mode_ == "--blocksize")
    run_blocksize();
  else if (mode_ == "--kernelsize")
    run_kernelsize();
  else
    std::cerr << "Unknown mode: " << mode_ << "\n";
}

void BenchmarkSuite::parse_args(int argc, char **argv) {
  std::vector<char *> filtered_argv;
  for (int i = 0; i < argc; ++i) {
    if (strcmp(argv[i], "--shared") == 0) {
      use_shared_mem_ = true;
    } else {
      filtered_argv.push_back(argv[i]);
    }
  }

  if (use_shared_mem_) {
    std::cout << ">> OPTIMIZATION: Shared Memory Kernel ENABLED <<\n\n";
  }

  if (filtered_argv.size() < 2) {
    std::cerr << "Usage: KernelApp [flags] <mode>\n";
    std::cerr << "Modes: --throughput, --blocksize, --kernelsize\n";
    std::cerr << "Flags: --shared\n";
    exit(1);
  }
  mode_ = filtered_argv[1];
}

void BenchmarkSuite::define_tests() {
  auto prewittX_data = Kernels::PrewittX();
  auto gauss5_data = Kernels::Gaussian5x5();
  auto gauss7_data = Kernels::Gaussian7x7();
  auto log_data = Kernels::LaplacianOfGaussian5x5();
  auto sharpen_data = Kernels::Sharpen();

  tests_ = {{"PrewittX", cv::Mat(3, 3, CV_32F, prewittX_data.data())},
            {"Gauss5x5", cv::Mat(5, 5, CV_32F, gauss5_data.data())},
            {"Gauss7x7", cv::Mat(7, 7, CV_32F, gauss7_data.data())},
            {"LoG5x5", cv::Mat(5, 5, CV_32F, log_data.data())},
            {"Sharpen3x3", cv::Mat(3, 3, CV_32F, sharpen_data.data())}};
}

cv::Mat BenchmarkSuite::getOrGenerateImage(int width, int height,
                                           const std::string &name_suffix) {
  std::string resolution_str =
      std::to_string(width) + "x" + std::to_string(height);
  std::filesystem::path image_path =
      "generated_images/" + resolution_str + name_suffix + ".tiff";
  std::filesystem::create_directories(image_path.parent_path());

  if (std::filesystem::exists(image_path)) {
    return cv::imread(image_path.string(), cv::IMREAD_COLOR);
  }

  std::cout << "Generating new random image: " << image_path << std::endl;
  cv::Mat new_img(height, width, CV_8UC3);
  cv::RNG rng(12345);
  rng.fill(new_img, cv::RNG::UNIFORM, 0, 256);
  cv::imwrite(image_path.string(), new_img, {cv::IMWRITE_TIFF_COMPRESSION, 1});
  return new_img;
}

void BenchmarkSuite::run_throughput() {
  std::cout << "\n--- Running Throughput Benchmark ---\n";
  std::string filename = "output/";
  if (use_shared_mem_) {
    filename += "shared_benchmark_throughput.csv";
  } else {
    filename += "stock_benchmark_throughput.csv";
  }
  std::ofstream csv(filename);
  csv << "Resolution,Kernel,AvgCPUTime,AvgGPUTime,Speedup,Optimization\n";
  std::string opt_type = use_shared_mem_ ? "Shared" : "Global";

  std::vector<int> sizes = {512, 1024, 1920, 2560, 3840, 4096};

  for (int size : sizes) {
    std::cout << "--- Benchmarking " << size << "x" << size << " Images ---"
              << std::endl;
    std::map<std::string, BenchmarkRunner::Result> accumulated_results;

    for (int i = 0; i < NUM_STATISTICAL_IMAGES; ++i) {
      cv::Mat image =
          getOrGenerateImage(size, size, "_img" + std::to_string(i));
      for (const auto &test : tests_) {
        auto result =
            BenchmarkRunner::run(image, test, dim3(16, 16), use_shared_mem_);
        accumulated_results[test.name].avg_cpu_time += result.avg_cpu_time;
        accumulated_results[test.name].avg_gpu_time += result.avg_gpu_time;
      }
    }

    for (const auto &test : tests_) {
      auto &total = accumulated_results[test.name];
      total.avg_cpu_time /= NUM_STATISTICAL_IMAGES;
      total.avg_gpu_time /= NUM_STATISTICAL_IMAGES;
      total.speedup = (total.avg_gpu_time > 0)
                          ? total.avg_cpu_time / total.avg_gpu_time
                          : 0;

      std::cout << "  " << std::setw(12) << std::left << test.name
                << " | Speedup: " << total.speedup << "x\n";
      csv << size << "x" << size << "," << test.name << ","
          << total.avg_cpu_time << "," << total.avg_gpu_time << ","
          << total.speedup << "," << opt_type << "\n";
    }
  }
  csv.close();
}

void BenchmarkSuite::run_blocksize() {
  std::cout << "\n--- Running Block Size Benchmark ---\n";
  std::cout << "Analyzes performance by varying threads-per-block on a "
               "fixed-size image.\n";
  std::cout << "Using " << NUM_STATISTICAL_IMAGES
            << " unique images for averaging.\n\n";

  const int w = 1920, h = 1920;
  std::string resolution_str = std::to_string(w) + "x" + std::to_string(h);

  std::string filename = "output/";
  if (use_shared_mem_) {
    filename += "shared_benchmark_blocksize.csv";
  } else {
    filename += "stock_benchmark_blocksize.csv";
  }
  std::ofstream csv(filename);
  csv << "Resolution,Kernel,BlockSize,ThreadsPerBlock,AvgCPUTime,AvgGPUTime,"
         "Speedup,Optimization\n";

  std::vector<dim3> block_sizes = {dim3(8, 8),  dim3(16, 8),  dim3(16, 16),
                                   dim3(32, 8), dim3(32, 16), dim3(32, 32)};

  for (const auto &test : tests_) {
    std::cout << "Benchmarking Kernel: " << test.name << std::endl;

    // Use a map to accumulate results for each block size configuration.
    // The key is the block size as a string (e.g., "16x16").
    std::map<std::string, BenchmarkRunner::Result> accumulated_results;

    for (int i = 0; i < NUM_STATISTICAL_IMAGES; ++i) {
      cv::Mat image = getOrGenerateImage(w, h, "_img" + std::to_string(i));

      for (const auto &block_dim : block_sizes) {
        // Determine if shared memory can be used for this specific block size.
        bool can_use_shared =
            use_shared_mem_ && (block_dim.x == 16 && block_dim.y == 16);

        auto result =
            BenchmarkRunner::run(image, test, block_dim, can_use_shared);

        std::string block_str =
            std::to_string(block_dim.x) + "x" + std::to_string(block_dim.y);
        accumulated_results[block_str].avg_cpu_time += result.avg_cpu_time;
        accumulated_results[block_str].avg_gpu_time += result.avg_gpu_time;
      }
    }

    // Final averaging and writing to CSV for this kernel
    std::cout << "  Final Results for " << test.name << ":" << std::endl;
    for (const auto &block_dim : block_sizes) {
      std::string block_str =
          std::to_string(block_dim.x) + "x" + std::to_string(block_dim.y);
      auto &total = accumulated_results[block_str];
      total.avg_cpu_time /= NUM_STATISTICAL_IMAGES;
      total.avg_gpu_time /= NUM_STATISTICAL_IMAGES;
      total.speedup = (total.avg_gpu_time > 0)
                          ? total.avg_cpu_time / total.avg_gpu_time
                          : 0;

      unsigned int threads_per_block = block_dim.x * block_dim.y;
      bool can_use_shared =
          use_shared_mem_ && (block_dim.x == 16 && block_dim.y == 16);
      std::string opt_type = can_use_shared ? "Shared" : "Global";

      std::cout << "    Block Size: " << std::setw(7) << std::left << block_str
                << " | " << std::setw(17)
                << ("(" + std::to_string(threads_per_block) + " thr / " +
                    opt_type + ")")
                << " | Speedup: " << total.speedup << "x\n";

      csv << resolution_str << "," << test.name << "," << block_str << ","
          << threads_per_block << "," << total.avg_cpu_time << ","
          << total.avg_gpu_time << "," << total.speedup << "," << opt_type
          << "\n";
    }
    std::cout << std::endl;
  }
  csv.close();
  std::cout << "\nBlock size benchmark data saved to "
               "'output/benchmark_blocksize.csv'\n";
}

void BenchmarkSuite::run_kernelsize() {
  std::cout << "\n--- Running Kernel Size Benchmark ---\n";
  std::cout << "Analyzes performance as the N in an NxN kernel increases.\n";
  std::cout << "Using " << NUM_STATISTICAL_IMAGES
            << " unique images for averaging.\n\n";

  const int w = 1920, h = 1920;
  std::string resolution_str = std::to_string(w) + "x" + std::to_string(h);

  std::string filename = "output/";
  if (use_shared_mem_) {
    filename += "shared_benchmark_kernelsize.csv";
  } else {
    filename += "stock_benchmark_kernelsize.csv";
  }
  std::ofstream csv(filename);
  std::cout << "Saving results to: " << filename << std::endl;
  // --- END FILENAME FIX ---

  csv << "Resolution,KernelType,KernelSize,AvgCPUTime,AvgGPUTime,Speedup,"
         "Optimization\n";
  std::string opt_type = use_shared_mem_ ? "Shared" : "Global";

  std::vector<int> kernel_sizes = {3, 5, 7, 9, 11, 15, 21};
  dim3 blockDim(16, 16); // A fixed, reasonable block size for this comparison

  for (int size : kernel_sizes) {
    std::cout << "--- Benchmarking " << size << "x" << size << " Kernels ---"
              << std::endl;

    // Generate the kernels for this size
    cv::Mat gauss_kernel = cv::getGaussianKernel(size, 0, CV_32F);
    gauss_kernel = gauss_kernel * gauss_kernel.t();
    cv::Mat box_kernel(size, size, CV_32F, cv::Scalar(1.0 / (size * size)));
    std::vector<Test> current_tests = {{"Gaussian", gauss_kernel},
                                       {"Box", box_kernel}};

    // Use maps to accumulate results for each kernel type (Gaussian, Box)
    std::map<std::string, BenchmarkRunner::Result> accumulated_results;

    for (int i = 0; i < NUM_STATISTICAL_IMAGES; ++i) {
      cv::Mat image = getOrGenerateImage(w, h, "_img" + std::to_string(i));
      for (const auto &test : current_tests) {
        // Shared memory can be used here because the block size is always 16x16
        auto result =
            BenchmarkRunner::run(image, test, blockDim, use_shared_mem_);
        accumulated_results[test.name].avg_cpu_time += result.avg_cpu_time;
        accumulated_results[test.name].avg_gpu_time += result.avg_gpu_time;
      }
    }

    // Final averaging and writing to CSV for this kernel size
    std::cout << "  Final Results for " << size << "x" << size << ":"
              << std::endl;
    for (const auto &test : current_tests) {
      auto &total = accumulated_results[test.name];
      total.avg_cpu_time /= NUM_STATISTICAL_IMAGES;
      total.avg_gpu_time /= NUM_STATISTICAL_IMAGES;
      total.speedup = (total.avg_gpu_time > 0)
                          ? total.avg_cpu_time / total.avg_gpu_time
                          : 0;

      std::cout << "    " << std::setw(10) << std::left << test.name
                << " | Speedup: " << total.speedup << "x\n";

      csv << resolution_str << "," << test.name << "," << size << ","
          << total.avg_cpu_time << "," << total.avg_gpu_time << ","
          << total.speedup << "," << opt_type << "\n";
    }
  }
  csv.close();
  std::cout << "\nKernel size benchmark data saved to "
               "'output/benchmark_kernelsize.csv'\n";
}
