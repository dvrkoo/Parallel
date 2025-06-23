#include "CpuConvolution.h"
#include <opencv2/core.hpp>

CpuConvolution::CpuConvolution(const cv::Mat &kernel)
    : kernel_(kernel), kRows_(kernel.rows), kCols_(kernel.cols),
      kCenterY_(kernel.rows / 2), kCenterX_(kernel.cols / 2) {
  CV_Assert(kernel.type() == CV_32F);
}

cv::Mat CpuConvolution::apply(const cv::Mat &input) const {
  CV_Assert(input.channels() == 1);
  cv::Mat output = cv::Mat::zeros(input.size(), input.type());

  for (int y = 0; y < input.rows; ++y) {
    for (int x = 0; x < input.cols; ++x) {
      float sum = 0;
      for (int m = 0; m < kRows_; ++m) {
        int yy = y + (m - kCenterY_);
        if (yy < 0 || yy >= input.rows)
          continue;
        for (int n = 0; n < kCols_; ++n) {
          int xx = x + (n - kCenterX_);
          if (xx < 0 || xx >= input.cols)
            continue;
          sum += input.at<uchar>(yy, xx) * kernel_.at<float>(m, n);
        }
      }
      output.at<uchar>(y, x) = cv::saturate_cast<uchar>(sum);
    }
  }
  return output;
}
