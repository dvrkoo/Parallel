#pragma once
#include <opencv2/opencv.hpp>

class GpuConvolution {
public:
  GpuConvolution(const cv::Mat &kernel);
  cv::Mat apply(const cv::Mat &input) const;

private:
  cv::Mat kernel_;
  int kRows_, kCols_;
};
