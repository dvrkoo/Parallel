#include "GpuConvolution.h"
#include <cuda_runtime.h>
#include <iostream>

#define checkCuda(val) check((val), #val, __FILE__, __LINE__)
static void check(cudaError_t result, const char* func, const char* file, int line) {
    if (result) {
        std::cerr << "CUDA error=" << static_cast<int>(result) << " (" << cudaGetErrorString(result) << ") at " << file << ":" << line << " '" << func << "'\n";
        cudaDeviceReset();
        exit(EXIT_FAILURE);
    }
}

__global__ void conv2dKernel(const unsigned char* in, float* out, // <--- out is now float*
                             int width, int height, int channels,
                             const float* kernel, int kW, int kH) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int cx = kW / 2;
    int cy = kH / 2;

    if (x < width && y < height) {
        for (int c = 0; c < channels; ++c) {
            float sum = 0;
            for (int m = 0; m < kH; ++m) {
                int yy = y + m - cy;
                if (yy < 0 || yy >= height) continue;
                for (int n = 0; n < kW; ++n) {
                    int xx = x + n - cx;
                    if (xx < 0 || xx >= width) continue;
                    int neighbor_idx = (yy * width + xx) * channels + c;
                    sum += in[neighbor_idx] * kernel[m * kW + n];
                }
            }
            // FIX 2: Do NOT take absolute value or clamp. Write the raw float sum.
            int output_idx = (y * width + x) * channels + c;
            out[output_idx] = sum;
        }
    }
}

GpuConvolution::GpuConvolution(const cv::Mat& kernel)
    : kernel_(kernel), kRows_(kernel.rows), kCols_(kernel.cols) {
    CV_Assert(kernel.type() == CV_32F);
}

cv::Mat GpuConvolution::apply(const cv::Mat& input) const {
    int w = input.cols;
    int h = input.rows;
    int channels = input.channels();

    // FIX 3: Output buffer must be floating point.
    size_t inputBytes = w * h * channels * sizeof(unsigned char);
    size_t outputBytes = w * h * channels * sizeof(float); // <--- Note sizeof(float)
    size_t kernBytes = kRows_ * kCols_ * sizeof(float);

    unsigned char* d_in;
    float* d_out; // <--- d_out is now float*
    float* d_k;

    checkCuda(cudaMalloc(&d_in, inputBytes));
    checkCuda(cudaMalloc(&d_out, outputBytes)); // <--- Allocate float buffer
    checkCuda(cudaMalloc(&d_k, kernBytes));

    checkCuda(cudaMemcpy(d_in, input.data, inputBytes, cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_k, kernel_.ptr<float>(), kernBytes, cudaMemcpyHostToDevice));
    
    dim3 b(16, 16);
    dim3 g((w + 15) / 16, (h + 15) / 16);

    conv2dKernel<<<g, b>>>(d_in, d_out, w, h, channels, d_k, kCols_, kRows_);
    checkCuda(cudaGetLastError());
    checkCuda(cudaDeviceSynchronize());

    // FIX 4: Create a floating point cv::Mat to receive the data.
    cv::Mat out(h, w, CV_32FC(channels));
    checkCuda(cudaMemcpy(out.data, d_out, outputBytes, cudaMemcpyDeviceToHost));
    
    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_k);
    
    return out;
}
