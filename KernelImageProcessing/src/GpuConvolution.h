#ifndef GPU_CONVOLUTION_H
#define GPU_CONVOLUTION_H

#include <cuda_runtime.h> // Needed for dim3
#include <opencv2/opencv.hpp>

class GpuConvolution {
public:
  GpuConvolution(const cv::Mat &kernel, bool use_shared_memory = false);
  ~GpuConvolution();

  cv::Mat apply(const cv::Mat &input, const dim3 &blockDim,
                int maxGridDimX = 0);

  cv::Mat apply(const cv::Mat &input) const {
    // We use const_cast because the "master" apply method is non-const,
    // but this operation doesn't logically change the state of the object.
    return const_cast<GpuConvolution *>(this)->apply(input, dim3(16, 16), 0);
  }

  // Apply method for scaling tests, also a wrapper.
  cv::Mat apply(const cv::Mat &input, int maxGridDimX) {
    return apply(input, dim3(16, 16), maxGridDimX);
  }

private:
  cv::Mat kernel_;
  int kRows_;
  int kCols_;

  float *d_k_ = nullptr;
  bool use_shared_mem_ = false;
};

#endif
