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
const int IMAGES_PER_RESOLUTION = 5;

bool g_use_shared_memory = false;

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

cv::Mat getOrGenerateImage(int width, int height,
                           const std::filesystem::path &image_path) {
  // Ensure the parent directory exists
  std::filesystem::create_directories(image_path.parent_path());

  if (std::filesystem::exists(image_path)) {
    std::cout << "Loading existing image: " << image_path << std::endl;
    cv::Mat img = cv::imread(image_path.string(), cv::IMREAD_COLOR);
    if (!img.empty()) {
      return img;
    }
    std::cerr << "Warning: Failed to load existing image, will regenerate."
              << std::endl;
  }

  std::cout << "Generating new random image (" << width << "x" << height
            << ") and saving to: " << image_path << std::endl;
  cv::Mat new_img(height, width, CV_8UC3);
  // Use a fixed seed for the random number generator for 100% reproducibility
  cv::RNG rng(12345);
  rng.fill(new_img, cv::RNG::UNIFORM, 0, 256);
  std::vector<int> tiff_params = {cv::IMWRITE_TIFF_COMPRESSION,
                                  1}; // 1 = No compression
  cv::imwrite(image_path.string(), new_img, tiff_params);
  return new_img;
}

// Forward declarations
void run_throughput_benchmark(const std::vector<Test> &tests);
void run_blocksize_benchmark(const std::vector<Test> &tests);
void run_kernelsize_benchmark(const std::vector<Test> &tests);
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
const ResolutionDir *

find_target_resolution_dir(std::vector<ResolutionDir> &resolution_dirs,
                           int target_w, int target_h) {
  const ResolutionDir *target_dir = nullptr;
  const ResolutionDir *fallback_dir = nullptr;

  // Sort to make finding the highest resolution easy for the fallback
  std::sort(resolution_dirs.begin(), resolution_dirs.end());
  if (!resolution_dirs.empty()) {
    fallback_dir =
        &resolution_dirs.back(); // The last element is the highest resolution
  }

  // Search for the specific target resolution
  for (const auto &dir : resolution_dirs) {
    if (dir.width == target_w && dir.height == target_h) {
      target_dir = &dir; // Found it!
      break;
    }
  }

  if (target_dir) {
    std::cout << "Found target resolution directory: '" << target_dir->name
              << "' (" << target_dir->width << "x" << target_dir->height
              << ")\n\n";
    return target_dir;
  } else {
    std::cout << "Warning: Target resolution " << target_w << "x" << target_h
              << " not found.\n";
    if (fallback_dir) {
      std::cout << "Falling back to highest available resolution: '"
                << fallback_dir->name << "' (" << fallback_dir->width << "x"
                << fallback_dir->height << ")\n\n";
    }
    return fallback_dir;
  }
}

// ===================================================================
//                          MAIN DISPATCHER
// ===================================================================

int main(int argc, char **argv) {
  // --- NEW ---: Simple argument parsing for the --shared flag
  std::vector<char *> filtered_argv;
  for (int i = 0; i < argc; ++i) {
    if (strcmp(argv[i], "--shared") == 0) {
      g_use_shared_memory = true;
    } else {
      filtered_argv.push_back(argv[i]);
    }
  }
  // Update argc and argv to a version without our custom flag
  argc = filtered_argv.size();
  argv = filtered_argv.data();

  if (g_use_shared_memory) {
    std::cout << ">> OPTIMIZATION: Shared Memory Kernel ENABLED <<\n\n";
  }

  if (argc < 3) {
    std::cerr << "Usage: KernelApp [flags] <mode> <root_image_directory>\n";
    std::cerr << "Modes:\n";
    std::cerr << "  --throughput   : Iterates through all subdirs and runs "
                 "throughput tests.\n";
    std::cerr
        << "  --blocksize    : Tests different block sizes on a 1440p image.\n";
    std::cerr << "  --kernelsize   : Tests performance as kernel size "
                 "increases on a 1440p image.\n";
    std::cerr << "Optional Flags:\n";
    std::cerr
        << "  --shared       : Use the optimized shared memory GPU kernel.\n";
    return 1;
  }

  std::string mode = argv[1];
  std::filesystem::path root_image_dir(argv[2]);

  if (!std::filesystem::is_directory(root_image_dir)) {
    std::cerr << "Error: Provided path is not a directory: " << argv[2] << "\n";
    return 1;
  }

  std::filesystem::create_directories("output");

  // Define tests (you can add more here)
  auto prewittX_data = Kernels::PrewittX();
  auto gauss5_data = Kernels::Gaussian5x5();
  auto gauss7_data = Kernels::Gaussian7x7();
  auto log_data = Kernels::LaplacianOfGaussian5x5(); // <-- ADD
  auto sharpen_data = Kernels::Sharpen();            // <-- ADD

  std::vector<Test> tests = {
      {"PrewittX", cv::Mat(3, 3, CV_32F, prewittX_data.data())},
      {"Gauss5x5", cv::Mat(5, 5, CV_32F, gauss5_data.data())},
      {"Gauss7x7", cv::Mat(7, 7, CV_32F, gauss7_data.data())},
      {"LoG5x5", cv::Mat(5, 5, CV_32F, log_data.data())},        // <-- ADD
      {"Sharpen3x3", cv::Mat(3, 3, CV_32F, sharpen_data.data())} // <-- ADD
  };

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

  // Dispatch to benchmark functions
  if (mode == "--throughput") {
    run_throughput_benchmark(tests);
  } else if (mode == "--blocksize") {
    run_blocksize_benchmark(tests);
  } else if (mode == "--kernelsize") {
    run_kernelsize_benchmark(tests);
  } else {
    std::cerr << "Unknown mode: " << mode << "\n";
    return 1;
  }

  return 0;
}

// ===================================================================
//                  BENCHMARK IMPLEMENTATIONS
// ===================================================================

// In main.cpp

// --- Add this constant at the top of your file ---

void run_throughput_benchmark(const std::vector<Test> &tests) {
  std::cout << "\n--- Running Throughput Benchmark ---\n";
  std::cout << "Using " << IMAGES_PER_RESOLUTION
            << " unique images per resolution.\n\n";

  std::ofstream csv("output/benchmark_throughput.csv");
  csv << "Resolution,Kernel,AvgCPUTime,AvgGPUTime,Speedup,Optimization\n";
  std::string optimization_type = g_use_shared_memory ? "Shared" : "Global";

  std::vector<int> sizes = {512, 1024, 1920, 2560, 3840, 4096};

  for (int size : sizes) {
    std::cout << "--- Benchmarking " << size << "x" << size << " Images ---"
              << std::endl;

    // --- NEW: Data structures to hold results across multiple images ---
    // We use a map to store total times for each kernel name
    std::map<std::string, double> total_cpu_times;
    std::map<std::string, double> total_gpu_times;

    // --- NEW: Loop over the number of images ---
    for (int img_idx = 0; img_idx < IMAGES_PER_RESOLUTION; ++img_idx) {
      std::string resolution_str =
          std::to_string(size) + "x" + std::to_string(size);

      // Generate a unique filename for each image
      std::filesystem::path image_path = "generated_images/" + resolution_str +
                                         "_img" + std::to_string(img_idx) +
                                         ".tiff";
      cv::Mat base_image = getOrGenerateImage(size, size, image_path);

      std::cout << "  Processing Image " << img_idx + 1 << "/"
                << IMAGES_PER_RESOLUTION << "..." << std::endl;

      // Run all tests on this single image
      for (const auto &t : tests) {
        // (image processing logic to get imageToProcess is the same)
        cv::Mat imageToProcess;
        if (t.name.find("Prewitt") != std::string::npos &&
            base_image.channels() > 1) {
          cv::cvtColor(base_image, imageToProcess, cv::COLOR_BGR2GRAY);
        } else {
          imageToProcess = base_image;
        }

        // Average the runs for this specific image and kernel
        double avg_cpu_time_for_this_image = 0;
        for (int i = 0; i < NUM_AVERAGING_RUNS; ++i)
          avg_cpu_time_for_this_image += timeCpu(imageToProcess, t.kern);
        avg_cpu_time_for_this_image /= NUM_AVERAGING_RUNS;

        GpuConvolution gpu_conv(t.kern, g_use_shared_memory);
        gpu_conv.apply(imageToProcess, dim3(16, 16), 1); // Warmup
        cudaDeviceSynchronize();

        double avg_gpu_time_for_this_image = 0;
        for (int i = 0; i < NUM_AVERAGING_RUNS; ++i)
          avg_gpu_time_for_this_image +=
              timeGpu(gpu_conv, imageToProcess, dim3(16, 16), 0);
        avg_gpu_time_for_this_image /= NUM_AVERAGING_RUNS;

        // Accumulate the results
        total_cpu_times[t.name] += avg_cpu_time_for_this_image;
        total_gpu_times[t.name] += avg_gpu_time_for_this_image;
      }
    }

    // --- NEW: Final averaging and writing to CSV ---
    // After testing all images for this resolution, calculate the final average
    // and write one row per kernel.
    std::cout << "  Final Results for " << size << "x" << size << ":"
              << std::endl;
    for (const auto &t : tests) {
      double final_avg_cpu_time =
          total_cpu_times[t.name] / IMAGES_PER_RESOLUTION;
      double final_avg_gpu_time =
          total_gpu_times[t.name] / IMAGES_PER_RESOLUTION;
      double speedup = (final_avg_gpu_time > 0)
                           ? final_avg_cpu_time / final_avg_gpu_time
                           : 0;

      std::cout << "    " << t.name << " | Speedup: " << speedup << "x"
                << std::endl;

      csv << size << "x" << size << "," << t.name << "," << final_avg_cpu_time
          << "," << final_avg_gpu_time << "," << speedup << ","
          << optimization_type << "\n";
    }
  }
  csv.close();
}

// In main.cpp

// In main.cpp

void run_kernelsize_benchmark(const std::vector<Test> &tests_to_ignore) {
  std::cout << "\n--- Running Kernel Size Benchmark ---\n";
  std::cout << "Analyzes performance as the N in an NxN kernel increases.\n";
  std::cout << "Using " << IMAGES_PER_RESOLUTION
            << " unique images for averaging.\n\n";

  const int w = 1920, h = 1920;
  std::string resolution_str = std::to_string(w) + "x" + std::to_string(h);

  std::ofstream csv("output/benchmark_kernelsize.csv");
  csv << "Resolution,KernelType,KernelSize,AvgCPUTime,AvgGPUTime,Speedup,"
         "Optimization\n";

  std::vector<int> kernel_sizes = {3, 5, 7, 9, 11, 15, 21};
  dim3 blockDim(16, 16); // A fixed, reasonable block size

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
    std::map<std::string, double> total_cpu_times;
    std::map<std::string, double> total_gpu_times;

    // Outer loop to process multiple unique images
    for (int img_idx = 0; img_idx < IMAGES_PER_RESOLUTION; ++img_idx) {
      std::filesystem::path image_path = "generated_images/" + resolution_str +
                                         "_img" + std::to_string(img_idx) +
                                         ".tiff";
      cv::Mat base_image = getOrGenerateImage(w, h, image_path);

      std::cout << "  Processing Image " << img_idx + 1 << "/"
                << IMAGES_PER_RESOLUTION << " on " << size << "x" << size
                << " kernels..." << std::endl;

      // Inner loop for the two kernel types (Gaussian and Box)
      for (const auto &t : current_tests) {
        // Since these are blur filters, we always use the color image
        cv::Mat imageToProcess = base_image;

        // Average the runs for this specific image and kernel
        double avg_cpu_time_for_this_image = 0;
        for (int i = 0; i < NUM_AVERAGING_RUNS; ++i)
          avg_cpu_time_for_this_image += timeCpu(imageToProcess, t.kern);
        avg_cpu_time_for_this_image /= NUM_AVERAGING_RUNS;

        GpuConvolution gpu_conv(t.kern, g_use_shared_memory);
        gpu_conv.apply(imageToProcess, blockDim, 1); // Warm-up
        cudaDeviceSynchronize();

        double avg_gpu_time_for_this_image = 0;
        for (int i = 0; i < NUM_AVERAGING_RUNS; ++i)
          avg_gpu_time_for_this_image +=
              timeGpu(gpu_conv, imageToProcess, blockDim, 0);
        avg_gpu_time_for_this_image /= NUM_AVERAGING_RUNS;

        // Accumulate results
        total_cpu_times[t.name] += avg_cpu_time_for_this_image;
        total_gpu_times[t.name] += avg_gpu_time_for_this_image;
      }
    }

    // Final averaging and writing to CSV for this kernel size
    std::cout << "  Final Results for " << size << "x" << size << ":"
              << std::endl;
    for (const auto &t : current_tests) {
      double final_avg_cpu_time =
          total_cpu_times[t.name] / IMAGES_PER_RESOLUTION;
      double final_avg_gpu_time =
          total_gpu_times[t.name] / IMAGES_PER_RESOLUTION;
      double speedup = (final_avg_gpu_time > 0)
                           ? final_avg_cpu_time / final_avg_gpu_time
                           : 0;

      std::string optimization_type = g_use_shared_memory ? "Shared" : "Global";
      std::cout << "    " << t.name << " | Speedup: " << speedup << "x\n";

      csv << resolution_str << "," << t.name << "," << size << ","
          << final_avg_cpu_time << "," << final_avg_gpu_time << "," << speedup
          << "," << optimization_type << "\n";
    }
  }
  csv.close();
  std::cout << "\nKernel size benchmark data saved to "
               "'output/benchmark_kernelsize.csv'\n";
}

// In main.cpp

void run_blocksize_benchmark(const std::vector<Test> &tests) {
  std::cout << "\n--- Running Block Size Benchmark ---\n";
  std::cout << "Analyzes performance by varying threads-per-block on a "
               "fixed-size image.\n";
  std::cout << "Using " << IMAGES_PER_RESOLUTION
            << " unique images for averaging.\n\n";

  const int w = 1920, h = 1920;
  std::string resolution_str = std::to_string(w) + "x" + std::to_string(h);

  std::ofstream csv("output/benchmark_blocksize.csv");
  csv << "Resolution,Kernel,BlockSize,ThreadsPerBlock,AvgCPUTime,AvgGPUTime,"
         "Speedup,Optimization\n";

  std::vector<dim3> block_sizes = {dim3(4, 4), dim3(8, 8), dim3(16, 16),
                                   dim3(32, 32)};

  for (const auto &t : tests) {
    std::cout << "Benchmarking Kernel: " << t.name << std::endl;

    // We will calculate the average CPU time once across all images,
    // as it's independent of the GPU block size.
    double total_cpu_time_across_images = 0;

    // This map will store the total GPU time for each block size across all
    // images.
    std::map<std::string, double> total_gpu_times_by_blocksize;

    // Outer loop to process multiple unique images
    for (int img_idx = 0; img_idx < IMAGES_PER_RESOLUTION; ++img_idx) {
      std::filesystem::path image_path = "generated_images/" + resolution_str +
                                         "_img" + std::to_string(img_idx) +
                                         ".tiff";
      cv::Mat base_image = getOrGenerateImage(w, h, image_path);

      std::cout << "  Processing Image " << img_idx + 1 << "/"
                << IMAGES_PER_RESOLUTION << "..." << std::endl;

      cv::Mat imageToProcess;
      if (t.name.find("Prewitt") != std::string::npos &&
          base_image.channels() > 1) {
        cv::cvtColor(base_image, imageToProcess, cv::COLOR_BGR2GRAY);
      } else {
        imageToProcess = base_image;
      }

      // Accumulate CPU time
      double avg_cpu_time_for_this_image = 0;
      for (int i = 0; i < NUM_AVERAGING_RUNS; ++i)
        avg_cpu_time_for_this_image += timeCpu(imageToProcess, t.kern);
      total_cpu_time_across_images +=
          (avg_cpu_time_for_this_image / NUM_AVERAGING_RUNS);

      // Inner loop to test all block sizes on the current image
      for (const auto &block_dim : block_sizes) {
        bool can_use_shared = g_use_shared_memory;
        if (g_use_shared_memory && (block_dim.x != 16 || block_dim.y != 16)) {
          can_use_shared = false;
        }

        GpuConvolution gpu_conv(t.kern, can_use_shared);
        gpu_conv.apply(imageToProcess, block_dim, 1); // Warm-up
        cudaDeviceSynchronize();

        double avg_gpu_time_for_this_block = 0;
        for (int i = 0; i < NUM_AVERAGING_RUNS; ++i) {
          avg_gpu_time_for_this_block +=
              timeGpu(gpu_conv, imageToProcess, block_dim, 0);
        }

        std::string block_str =
            std::to_string(block_dim.x) + "x" + std::to_string(block_dim.y);
        total_gpu_times_by_blocksize[block_str] +=
            (avg_gpu_time_for_this_block / NUM_AVERAGING_RUNS);
      }
    }

    // Final averaging and writing to CSV for this kernel
    double final_avg_cpu_time =
        total_cpu_time_across_images / IMAGES_PER_RESOLUTION;
    std::cout << "  Final Results for " << t.name << ":" << std::endl;

    for (const auto &block_dim : block_sizes) {
      std::string block_str =
          std::to_string(block_dim.x) + "x" + std::to_string(block_dim.y);
      double final_avg_gpu_time =
          total_gpu_times_by_blocksize[block_str] / IMAGES_PER_RESOLUTION;
      double speedup =
          final_avg_gpu_time > 0 ? final_avg_cpu_time / final_avg_gpu_time : 0;

      bool can_use_shared =
          g_use_shared_memory && (block_dim.x == 16 && block_dim.y == 16);
      std::string optimization_type = can_use_shared ? "Shared" : "Global";
      unsigned int threads_per_block = block_dim.x * block_dim.y;

      std::cout << "    Block Size: " << std::setw(7) << std::left << block_str
                << " | " << std::setw(17)
                << ("(" + std::to_string(threads_per_block) + " thr / " +
                    optimization_type + ")")
                << " | Speedup: " << speedup << "x\n";

      csv << resolution_str << "," << t.name << "," << block_str << ","
          << threads_per_block << "," << final_avg_cpu_time << ","
          << final_avg_gpu_time << "," << speedup << "," << optimization_type
          << "\n";
    }
    std::cout << std::endl;
  }
  csv.close();
  std::cout << "\nBlock size benchmark data saved to "
               "'output/benchmark_blocksize.csv'\n";
}
