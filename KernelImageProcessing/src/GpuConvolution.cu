// GpuConvolution.cu

#include "GpuConvolution.h"
#include <iostream>

// (checkCuda function remains unchanged)
#define checkCuda(val) check((val), #val, __FILE__, __LINE__)
static void check(cudaError_t result, const char* func, const char* file, int line) {
    if (result) {
        std::cerr << "CUDA error=" << static_cast<int>(result) << " (" << cudaGetErrorString(result) << ") at " << file << ":" << line << " '" << func << "'\n";
        cudaDeviceReset();
        exit(EXIT_FAILURE);
    }
}

// (conv2dKernel remains unchanged)
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




// --- IMPLEMENTATION OF FUNCTIONS DECLARED IN THE HEADER ---

GpuConvolution::GpuConvolution(const cv::Mat& kernel)
    : kernel_(kernel), kRows_(kernel.rows), kCols_(kernel.cols) {
    CV_Assert(kernel.type() == CV_32F);
    
    size_t kernBytes = kRows_ * kCols_ * sizeof(float);
    checkCuda(cudaMalloc(&d_k_, kernBytes));
    checkCuda(cudaMemcpy(d_k_, kernel_.ptr<float>(), kernBytes, cudaMemcpyHostToDevice));
}

GpuConvolution::~GpuConvolution() {
    if (d_k_) {
        cudaFree(d_k_);
    }
}

// --- IMPLEMENTATION OF THE "MASTER" APPLY METHOD ONLY ---
// DO NOT implement the other apply() wrappers here, they are in the header.
cv::Mat GpuConvolution::apply(const cv::Mat& input, const dim3& blockDim, int maxGridDimX) {
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
    
    // The blockDim is now a parameter.
    dim3 gridDim((w + blockDim.x - 1) / blockDim.x, (h + blockDim.y - 1) / blockDim.y);

    if (maxGridDimX > 0 && maxGridDimX < gridDim.x) {
        gridDim.x = maxGridDimX;
    }

    // Launch the kernel with the specified block dimensions
    conv2dKernel<<<gridDim, blockDim>>>(d_in, d_out, w, h, channels, d_k_, kCols_, kRows_);
    
    checkCuda(cudaGetLastError());
    
    cv::Mat out(h, w, CV_32FC(channels));
    checkCuda(cudaMemcpy(out.data, d_out, outputBytes, cudaMemcpyDeviceToHost));
    
    cudaFree(d_in);
    cudaFree(d_out);
    
    return out;
}
