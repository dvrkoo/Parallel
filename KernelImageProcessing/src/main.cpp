#include "CpuConvolution.h"
#include "GpuConvolution.h"
#include "Kernels.h"
#include <chrono>
#include <iostream>
#include <opencv2/opencv.hpp>

using Clock = std::chrono::high_resolution_clock;

double timeSeq(const cv::Mat &img, const cv::Mat &kern) {
  CpuConvolution conv(kern);
  auto start = Clock::now();
  conv.apply(img);
  return std::chrono::duration<double>(Clock::now() - start).count();
}

double timeGpu(const cv::Mat &img, const cv::Mat &kern) {
  GpuConvolution conv(kern);
  auto start = Clock::now();
  conv.apply(img);
  return std::chrono::duration<double>(Clock::now() - start).count();
}

int main(int argc, char **argv) {
  if (argc < 2) {
    std::cerr << "Usage: KernelApp <image>\n";
    return 1;
  }
  cv::Mat img = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
  if (img.empty()) {
    std::cerr << "Image load fail\n";
    return 1;
  }

  struct Test {
    std::string name;
    cv::Mat kern;
  };
  std::vector<Test> tests = {
      {"PrewittX", cv::Mat(3, 3, CV_32F, Kernels::PrewittX().data())},
      {"Gauss5x5", cv::Mat(5, 5, CV_32F, Kernels::Gaussian5x5().data())},
      {"Gauss7x7", cv::Mat(7, 7, CV_32F, Kernels::Gaussian7x7().data())}};

  std::vector<int> sizes = {256, 1024, 2048};
  for (int s : sizes) {
    cv::Mat rimg;
    cv::resize(img, rimg, cv::Size(s, s));
    std::cout << "Image " << s << "x" << s << "\n";
    for (auto &t : tests) {
      double cs = timeSeq(rimg, t.kern);
      double cg = timeGpu(rimg, t.kern);
      std::cout << "  " << t.name << ": CPU=" << cs << "s GPU=" << cg
                << "s speedup=" << cs / cg << "\n";
    }
  }
  return 0;
}
