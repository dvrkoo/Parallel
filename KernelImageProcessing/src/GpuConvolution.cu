#include "Convolution.h"
#include <opencv2/core.hpp>

Convolution::Convolution(const cv::Mat& kernel)
    : kernel_(kernel)
    , kRows_(kernel.rows)
    , kCols_(kernel.cols)
    , kCenterY_(kernel.rows / 2)
    , kCenterX_(kernel.cols / 2)
{
    CV_Assert(kernel.type() == CV_32F);
}

cv::Mat Convolution::apply(const cv::Mat& input) const {
    CV_Assert(input.channels() == 1);
    cv::Mat output = cv::Mat::zeros(input.size(), input.type());
    for (int y = 0; y < input.rows; ++y) {
        for (int x = 0; x < input.cols; ++x) {
            float sum = 0;
            for (int m = 0; m < kRows_; ++m) {
                int yy = y + (m - kCenterY_);
                if (yy < 0 || yy >= input.rows) continue;
                for (int n = 0; n < kCols_; ++n) {
                    int xx = x + (n - kCenterX_);
                    if (xx < 0 || xx >= input.cols) continue;
                    sum += input.at<uchar>(yy, xx) * kernel_.at<float>(m, n);
                }
            }
            output.at<uchar>(y, x) = cv::saturate_cast<uchar>(sum);
        }
    }
    return output;
}

// src/GpuConvolution.cu
#include "GpuConvolution.h"
#include <cuda_runtime.h>
#include <iostream>

#define checkCuda(val) check((val), #val, __FILE__, __LINE__)
static void check(cudaError_t result, const char* func, const char* file, int line) {
    if (result) {
        std::cerr << "CUDA error=" << result << " at " << file << ":" << line << " '" << func << "'\n";
        cudaDeviceReset();
        exit(EXIT_FAILURE);
    }
}

__global__ void conv2dKernel(const unsigned char* in, unsigned char* out,
                             int width, int height, const float* kernel,
                             int kW, int kH) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int cx = kW / 2;
    int cy = kH / 2;
    if (x < width && y < height) {
        float sum = 0;
        for (int m = 0; m < kH; ++m) {
            int yy = y + m - cy;
            if (yy < 0 || yy >= height) continue;
            for (int n = 0; n < kW; ++n) {
                int xx = x + n - cx;
                if (xx < 0 || xx >= width) continue;
                sum += in[yy * width + xx] * kernel[m * kW + n];
            }
        }
        out[y * width + x] = min(max(int(sum), 0), 255);
    }
}

GpuConvolution::GpuConvolution(const cv::Mat& kernel)
    : kernel_(kernel), kRows_(kernel.rows), kCols_(kernel.cols) {
    CV_Assert(kernel.type() == CV_32F);
}

cv::Mat GpuConvolution::apply(const cv::Mat& input) const {
    CV_Assert(input.channels() == 1);
    int w = input.cols;
    int h = input.rows;
    size_t imgB = w * h * sizeof(unsigned char);
    size_t kernB = kRows_ * kCols_ * sizeof(float);
    unsigned char *d_in, *d_out;
    float* d_k;
    checkCuda(cudaMalloc(&d_in, imgB));
    checkCuda(cudaMalloc(&d_out, imgB));
    checkCuda(cudaMalloc(&d_k, kernB));
    checkCuda(cudaMemcpy(d_in, input.data, imgB, cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_k, kernel_.ptr<float>(), kernB, cudaMemcpyHostToDevice));
    dim3 b(16, 16);
    dim3 g((w + 15) / 16, (h + 15) / 16);
    conv2dKernel<<<g, b>>>(d_in, d_out, w, h, d_k, kCols_, kRows_);
    checkCuda(cudaDeviceSynchronize());
    cv::Mat out(h, w, CV_8UC1);
    checkCuda(cudaMemcpy(out.data, d_out, imgB, cudaMemcpyDeviceToHost));
    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_k);
    return out;
}
