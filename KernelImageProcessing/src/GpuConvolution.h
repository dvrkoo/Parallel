// GpuConvolution.h

#ifndef GPU_CONVOLUTION_H
#define GPU_CONVOLUTION_H

#include <cuda_runtime.h> // Needed for dim3
#include <opencv2/opencv.hpp>

class GpuConvolution {
public:
  // --- DECLARATIONS ---
  // These functions have their code in the .cu file.
  GpuConvolution(const cv::Mat &kernel);
  ~GpuConvolution();

  // The "master" apply method that does all the work.
  cv::Mat apply(const cv::Mat &input, const dim3 &blockDim,
                int maxGridDimX = 0);

  // --- INLINE WRAPPER DEFINITIONS ---
  // These are simple helper functions. Defining them inside the class
  // in the header file makes them 'inline' and resolves compilation errors.

  // Original apply method, now a wrapper.
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
};

#endif // GPU_CONVOLUTION_H
