#pragma once
#include <opencv2/opencv.hpp>

class CpuConvolution {
public:
  CpuConvolution(const cv::Mat &kernel);
  cv::Mat apply(const cv::Mat &input) const;

private:
  cv::Mat kernel_;
  int kRows_, kCols_, kCenterY_, kCenterX_;
};
