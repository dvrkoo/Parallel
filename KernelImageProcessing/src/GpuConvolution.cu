// GpuConvolution.cu

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


__global__ void conv2dKernel(const unsigned char* in, float* out,
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
            int output_idx = (y * width + x) * channels + c;
            out[output_idx] = sum;
        }
    }
}

GpuConvolution::GpuConvolution(const cv::Mat& kernel)
    : kernel_(kernel), kRows_(kernel.rows), kCols_(kernel.cols) {
    CV_Assert(kernel.type() == CV_32F);
    
    // Allocate device memory for the kernel once
    size_t kernBytes = kRows_ * kCols_ * sizeof(float);
    checkCuda(cudaMalloc(&d_k_, kernBytes));
    checkCuda(cudaMemcpy(d_k_, kernel_.ptr<float>(), kernBytes, cudaMemcpyHostToDevice));
}

// Add a destructor to free the device memory for the kernel
GpuConvolution::~GpuConvolution() {
    if (d_k_) {
        cudaFree(d_k_);
    }
}

cv::Mat GpuConvolution::apply(const cv::Mat& input) const {
    // This is a bit of a trick to call the non-const method from a const one.
    // It's safe here because the non-const method doesn't actually modify the class state long-term.
    return const_cast<GpuConvolution*>(this)->apply(input, 0); // 0 means no limit
}

cv::Mat GpuConvolution::apply(const cv::Mat& input, int maxGridDimX) {
    int w = input.cols;
    int h = input.rows;
    int channels = input.channels();

    size_t inputBytes = w * h * channels * sizeof(unsigned char);
    size_t outputBytes = w * h * channels * sizeof(float); 

    unsigned char* d_in;
    float* d_out;

    checkCuda(cudaMalloc(&d_in, inputBytes));
    checkCuda(cudaMalloc(&d_out, outputBytes));

    checkCuda(cudaMemcpy(d_in, input.data, inputBytes, cudaMemcpyHostToDevice));
    
    dim3 threadsPerBlock(16, 16);
    dim3 gridDim((w + 15) / 16, (h + 15) / 16);

    // --- THIS IS THE KEY LOGIC FOR SCALING ---
    // If a limit is specified (and is valid), apply it to the grid's X-dimension.
    if (maxGridDimX > 0 && maxGridDimX < gridDim.x) {
        gridDim.x = maxGridDimX;
    }

    // Launch the kernel with the (potentially limited) grid
    conv2dKernel<<<gridDim, threadsPerBlock>>>(d_in, d_out, w, h, channels, d_k_, kCols_, kRows_);
    
    // Check for kernel launch errors
    checkCuda(cudaGetLastError());
    
    // NOTE: cudaDeviceSynchronize() is removed from here because the timing function
    // in main.cpp will handle synchronization with cudaEventSynchronize.

    cv::Mat out(h, w, CV_32FC(channels));
    checkCuda(cudaMemcpy(out.data, d_out, outputBytes, cudaMemcpyDeviceToHost));
    
    cudaFree(d_in);
    cudaFree(d_out);
    
    return out;
}
