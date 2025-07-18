#include "CpuConvolution.h"
#include <opencv2/core.hpp>

CpuConvolution::CpuConvolution(const cv::Mat &kernel)
    : kernel_(kernel), kRows_(kernel.rows), kCols_(kernel.cols),
      kCenterY_(kernel.rows / 2), kCenterX_(kernel.cols / 2) {
  CV_Assert(kernel.type() == CV_32F);
}

cv::Mat CpuConvolution::apply(const cv::Mat &input) const {
  // Output should be floating point to preserve negative/out-of-range values.
  cv::Mat output = cv::Mat::zeros(input.size(), CV_32FC(input.channels()));
  int channels = input.channels();

  // Iterate over each pixel in the output image.
  for (int y = 0; y < input.rows; ++y) {
    for (int x = 0; x < input.cols; ++x) {

      // --- Unified Kernel Loop ---
      // This inner block calculates the convolution for one (x,y) location.
      for (int m = 0; m < kRows_; ++m) {
        int yy = y + m - kCenterY_;
        // Boundary check for rows
        if (yy < 0 || yy >= input.rows) {
          continue;
        }

        for (int n = 0; n < kCols_; ++n) {
          int xx = x + n - kCenterX_;
          // Boundary check for columns
          if (xx < 0 || xx >= input.cols) {
            continue;
          }

          float k_val = kernel_.at<float>(m, n);
          for (int c = 0; c < channels; ++c) {
            const uchar *p_in = input.data + yy * input.step + xx * channels;
            float *p_out =
                (float *)(output.data + y * output.step) + x * channels;
            p_out[c] += p_in[c] * k_val;
          }
        }
      }
    }
  }
  return output;
}
