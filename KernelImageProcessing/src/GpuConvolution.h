#ifndef GPU_CONVOLUTION_H
#define GPU_CONVOLUTION_H

#include <opencv2/opencv.hpp>

class GpuConvolution {
public:
  // Constructor
  GpuConvolution(const cv::Mat &kernel);
  ~GpuConvolution();

  // Original apply method, remains for compatibility
  cv::Mat apply(const cv::Mat &input) const;

  // --- NEW ---
  // Overloaded apply method for scaling tests.
  // It accepts a parameter to limit the number of blocks in the X-dimension.
  // A value of 0 means no limit.
  cv::Mat apply(const cv::Mat &input, int maxGridDimX);

private:
  cv::Mat kernel_;
  int kRows_;
  int kCols_;

  // --- NEW ---
  // Member variables for CUDA device pointers to avoid re-allocation
  // on every 'apply' call, which makes timing more accurate.
  float *d_k_ = nullptr;
};

#endif
