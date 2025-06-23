#include "CpuConvolution.h"
#include <opencv2/core.hpp>

CpuConvolution::CpuConvolution(const cv::Mat &kernel)
    : kernel_(kernel), kRows_(kernel.rows), kCols_(kernel.cols),
      kCenterY_(kernel.rows / 2), kCenterX_(kernel.cols / 2) {
  CV_Assert(kernel.type() == CV_32F);
}

cv::Mat CpuConvolution::apply(const cv::Mat &input) const {
  // Output should be floating point to preserve negative values
  cv::Mat output = cv::Mat::zeros(input.size(), CV_32FC(input.channels()));
  int channels = input.channels();

  if (channels == 1) {
    for (int y = 0; y < input.rows; ++y) {
      for (int x = 0; x < input.cols; ++x) {
        float sum = 0.0f;
        for (int m = 0; m < kRows_; ++m) {
          int yy = y + m - kCenterY_;
          if (yy < 0 || yy >= input.rows)
            continue;
          for (int n = 0; n < kCols_; ++n) {
            int xx = x + n - kCenterX_;
            if (xx < 0 || xx >= input.cols)
              continue;
            sum += input.at<uchar>(yy, xx) * kernel_.at<float>(m, n);
          }
        }
        output.at<float>(y, x) = sum;
      }
    }
  } else if (channels == 3) {
    for (int y = 0; y < input.rows; ++y) {
      for (int x = 0; x < input.cols; ++x) {
        cv::Vec3f sum(0, 0, 0);
        for (int m = 0; m < kRows_; ++m) {
          int yy = y + m - kCenterY_;
          if (yy < 0 || yy >= input.rows)
            continue;
          for (int n = 0; n < kCols_; ++n) {
            int xx = x + n - kCenterX_;
            if (xx < 0 || xx >= input.cols)
              continue;
            cv::Vec3b pixel = input.at<cv::Vec3b>(yy, xx);
            float kval = kernel_.at<float>(m, n);
            sum[0] += pixel[0] * kval;
            sum[1] += pixel[1] * kval;
            sum[2] += pixel[2] * kval;
          }
        }
        // FIX: Do not take absolute value. Store the raw Vec3f sum.
        output.at<cv::Vec3f>(y, x) = sum;
      }
    }
  }

  return output;
}
