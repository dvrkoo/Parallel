#include "BenchmarkSuite.h"
#include "CpuConvolution.h"
#include "GpuConvolution.h"
#include "Kernels.h"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

// --- Configuration Constants ---
const int NUM_TIMING_RUNS = 30;
const int NUM_STATISTICAL_IMAGES = 5;

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
  else if (mode_ == "--visual")
    run_visual_test();
  else
    std::cerr << "Unknown mode: " << mode_ << "\n";
}

void BenchmarkSuite::parse_args(int argc, char **argv) {
  std::vector<char *> filtered_argv;
  for (int i = 0; i < argc; ++i) {
    if (strcmp(argv[i], "--shared") == 0) {
      use_shared_mem_ = true;
    } else if (strcmp(argv[i], "--tuned") == 0) {
      use_tuned_params_ = true;
    } else {
      filtered_argv.push_back(argv[i]);
    }
  }

  if (use_shared_mem_) {
    std::cout << ">> OPTIMIZATION: Shared Memory Kernel ENABLED <<\n\n";
  }
  if (use_tuned_params_) {
    std::cout << ">> OPTIMIZATION: Tuned Parameters ENABLED <<\n\n";
  }

  if (filtered_argv.size() < 2) {
    std::cerr << "Usage: KernelApp [flags] <mode>\n";
    std::cerr << "Modes: --throughput, --blocksize, --kernelsize --visual\n";
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
    filename += "shared_";
  } else {
    filename += "stock_";
  }
  if (use_tuned_params_) {
    filename += "tuned_benchmark_throughput.csv";
  } else {
    filename += "benchmark_throughput.csv";
  }
  std::ofstream csv(filename);
  csv << "Resolution,Kernel,AvgCPUTime,AvgGPUTime,Speedup,Optimization\n";
  std::string opt_type = use_shared_mem_ ? "Shared" : "Global";
  std::map<std::string, dim3> optimal_block_sizes = {
      {"PrewittX", dim3(16, 8)},
      {"Gauss5x5", dim3(32, 16)},
      {"Gauss7x7", dim3(32, 16)},
      {"LoG5x5", dim3(16, 8)},
      {"Sharpen3x3", dim3(16, 16)}};
  const dim3 default_block_dim(16, 16);

  std::vector<int> sizes = {512, 1024, 1920, 2560, 3840, 4096};

  for (int size : sizes) {
    std::cout << "--- Benchmarking " << size << "x" << size << " Images ---"
              << std::endl;
    std::map<std::string, BenchmarkRunner::Result> accumulated_results;

    for (int i = 0; i < NUM_STATISTICAL_IMAGES; ++i) {
      cv::Mat image =
          getOrGenerateImage(size, size, "_img" + std::to_string(i));
      for (const auto &test : tests_) {
        dim3 block_dim;
        if (use_tuned_params_) {
          block_dim = optimal_block_sizes[test.name];
        } else {
          block_dim = default_block_dim; // Use a default block size
        }
        auto result =
            BenchmarkRunner::run(image, test, block_dim, use_shared_mem_);
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

  std::vector<dim3> block_sizes = {dim3(4, 4),   dim3(8, 8),  dim3(16, 8),
                                   dim3(16, 16), dim3(32, 8), dim3(32, 16),
                                   dim3(32, 32)};

  for (const auto &test : tests_) {
    std::cout << "Benchmarking Kernel: " << test.name << std::endl;

    // Use a map to accumulate results for each block size configuration.
    // The key is the block size as a string (e.g., "16x16").
    std::map<std::string, BenchmarkRunner::Result> accumulated_results;

    for (int i = 0; i < NUM_STATISTICAL_IMAGES; ++i) {
      cv::Mat image = getOrGenerateImage(w, h, "_img" + std::to_string(i));

      for (const auto &block_dim : block_sizes) {
        auto result =
            BenchmarkRunner::run(image, test, block_dim, use_shared_mem_);

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
      std::string opt_type = use_shared_mem_ ? "Shared" : "Global";

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
void BenchmarkSuite::run_visual_test() {
  std::cout << "\n--- Running Visual Kernel Output Test ---\n";

  std::string input_dir = "img/1920x1080";
  std::string output_dir = "output_images";
  std::filesystem::create_directories(output_dir);

  dim3 blockDim(16, 16);

  // --- Create PrewittY kernel specifically for the combined gradient test ---
  auto prewittY_data = Kernels::PrewittY();
  Test prewittY_test = {"PrewittY",
                        cv::Mat(3, 3, CV_32F, prewittY_data.data())};

  for (const auto &entry : std::filesystem::directory_iterator(input_dir)) {
    if (entry.is_regular_file()) {
      std::string input_path = entry.path().string();
      std::string filename = entry.path().stem().string();

      std::cout << "Processing image: " << input_path << std::endl;
      cv::Mat image = cv::imread(input_path, cv::IMREAD_COLOR);
      if (image.empty()) {
        std::cerr << "Failed to read image: " << input_path << "\n";
        continue;
      }

      cv::Mat grayscale_image;
      cv::cvtColor(image, grayscale_image, cv::COLOR_BGR2GRAY);

      for (const auto &test : tests_) {
        std::string out_name =
            output_dir + "/" + filename + "_" + test.name + ".png";
        cv::Mat output_vis;

        // --- SPECIAL CASE FOR PREWITT GRADIENT MAGNITUDE ---
        if (test.name.find("Prewitt") != std::string::npos) {
          std::cout << "  (Visualizing '" << test.name
                    << "' as Combined Gradient Magnitude)" << std::endl;

          GpuConvolution gpu_conv_x(test.kern, use_shared_mem_);
          cv::Mat gx_float = gpu_conv_x.apply(grayscale_image, blockDim);

          GpuConvolution gpu_conv_y(prewittY_test.kern, use_shared_mem_);
          cv::Mat gy_float = gpu_conv_y.apply(grayscale_image, blockDim);

          cv::Mat magnitude;
          cv::magnitude(gx_float, gy_float,
                        magnitude); // Simpler way to do sqrt(gx^2 + gy^2)

          // Normalize the magnitude to the full 0-255 range and then convert.
          // This is more robust than a direct convertScaleAbs.
          cv::normalize(magnitude, output_vis, 0, 255, cv::NORM_MINMAX, CV_8U);

          out_name = output_dir + "/" + filename + "_Prewitt_Magnitude.png";

        } else { // --- UNIFIED, ROBUST STRATEGY FOR ALL OTHER FILTERS ---

          cv::Mat imageToProcess;
          // LoG also works best on grayscale
          if (test.name.find("LoG") != std::string::npos) {
            imageToProcess = grayscale_image;
          } else {
            imageToProcess = image; // Use color for Gaussian, Sharpen, etc.
          }

          GpuConvolution gpu_conv(test.kern, use_shared_mem_);
          cv::Mat output_float = gpu_conv.apply(imageToProcess, blockDim);

          std::cout << "  (Visualizing '" << test.name
                    << "' with robust normalization)" << std::endl;

          // This is the universal "make it visible" function. It will find the
          // min and max values in the float image (e.g., -150 to 200) and
          // stretch that range linearly to fit into 0-255. This works for
          // blurs, sharpening, and LoG.
          cv::normalize(output_float, output_vis, 0, 255, cv::NORM_MINMAX,
                        CV_8U);
        }

        cv::imwrite(out_name, output_vis);
        std::cout << "  Saved: " << out_name << "\n";
      }
    }
  }
  std::cout << "\nVisual test complete. Output saved to: " << output_dir
            << "\n";
}
