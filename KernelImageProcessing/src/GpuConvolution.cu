#include "GpuConvolution.h"
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


template <typename InType, typename OutType>
__global__ void conv2dSharedMemKernel_Final(const InType* in, OutType* out,
                                            int width, int height,
                                            const float* kernel, int k_side) {
    
    extern __shared__ unsigned char tile_raw[];
    InType* const tile = (InType*)tile_raw;

    const int HALO = k_side / 2;
    // The tile width needs to accommodate the block width plus the halo on both sides.
    const int TILE_WIDTH = blockDim.x + 2 * HALO;
    // The tile height is simply the block height. We load 'k_side' rows for each block row.
    const int TILE_HEIGHT = blockDim.y;


    // --- High-Performance Loading Pattern (Corrected) ---
    // Each thread loads a vertical stripe of 'k_side' pixels.
    int base_load_x = blockIdx.x * blockDim.x + threadIdx.x - HALO;
    int base_load_y = blockIdx.y * blockDim.y - HALO;

    for (int m = 0; m < k_side; ++m) {
        int load_y = base_load_y + threadIdx.y + m;
        int tile_y = threadIdx.y + m;

        if (base_load_x >= 0 && base_load_x < width && load_y >= 0 && load_y < height) {
            // This load is coalesced because adjacent threads access adjacent memory.
            tile[tile_y * TILE_WIDTH + threadIdx.x] = in[load_y * width + base_load_x];
        } else {
            memset(&tile[tile_y * TILE_WIDTH + threadIdx.x], 0, sizeof(InType));
        }
    }
    
    // To complete the halo on the right side, the first few threads in each row
    // need to load extra pixels.
    for (int m = 0; m < k_side; ++m) {
        if (threadIdx.x < 2 * HALO) {
            int load_y = base_load_y + threadIdx.y + m;
            int tile_y = threadIdx.y + m;
            int load_x = base_load_x + blockDim.x + threadIdx.x; // Load from the right
            int tile_x = blockDim.x + threadIdx.x;

            if (load_x >= 0 && load_x < width && load_y >= 0 && load_y < height) {
                tile[tile_y * TILE_WIDTH + tile_x] = in[load_y * width + load_x];
            } else {
                memset(&tile[tile_y * TILE_WIDTH + tile_x], 0, sizeof(InType));
            }
        }
    }

    __syncthreads();

    // --- Computation (uses the loaded tile) ---
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (out_x < width && out_y < height) {
        OutType sum;
        memset(&sum, 0, sizeof(OutType));

        // The thread's local (x,y) position within the shared memory tile's top-left corner
        const int local_x_in_tile = threadIdx.x;
        const int local_y_in_tile = threadIdx.y;

        for (int m = 0; m < k_side; m++) {
            for (int n = 0; n < k_side; n++) {
                // Read from the sub-region relative to the thread's position
                InType pixel = tile[(local_y_in_tile + m) * TILE_WIDTH + (local_x_in_tile + n)];
                float k_val = kernel[m * k_side + n];

                if constexpr (sizeof(InType) == 1) { // Grayscale
                    sum += pixel * k_val;
                } else { // Color
                    sum.x += pixel.x * k_val;
                    sum.y += pixel.y * k_val;
                    sum.z += pixel.z * k_val;
                }
            }
        }
        out[out_y * width + out_x] = sum;
    }
}


GpuConvolution::GpuConvolution(const cv::Mat& kernel, bool use_shared_memory)
    : kernel_(kernel), kRows_(kernel.rows), kCols_(kernel.cols), use_shared_mem_(use_shared_memory) 
{
    // The rest of the constructor body is the same.
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


// In GpuConvolution.cu

cv::Mat GpuConvolution::apply(const cv::Mat& input, const dim3& blockDim, int maxGridDimX) {
    int w = input.cols;
    int h = input.rows;
    int channels = input.channels();

    // (Memory allocation and setup logic is the same...)
    size_t inputBytes = w * h * channels * sizeof(unsigned char);
    size_t outputBytes = w * h * channels * sizeof(float);
    unsigned char* d_in;
    float* d_out;
    checkCuda(cudaMalloc(&d_in, inputBytes));
    checkCuda(cudaMalloc(&d_out, outputBytes));
    checkCuda(cudaMemcpy(d_in, input.data, inputBytes, cudaMemcpyHostToDevice));

    dim3 gridDim((w + blockDim.x - 1) / blockDim.x, (h + blockDim.y - 1) / blockDim.y);
    if (maxGridDimX > 0 && maxGridDimX < gridDim.x) {
        gridDim.x = maxGridDimX;
    }

    if (use_shared_mem_) {
        CV_Assert(blockDim.x == 16 && blockDim.y == 16 && "Shared memory kernel is optimized for 16x16 blocks.");
        
        int halo = kCols_ / 2;
        int tile_width = blockDim.x + 2 * halo;
        int tile_height = blockDim.y + 2 * halo; // <-- This was the subtle bug here
        if (channels == 3) {
            size_t shared_mem_bytes = tile_height * tile_width * sizeof(uchar3);
            conv2dSharedMemKernel_Final<uchar3, float3><<<gridDim, blockDim, shared_mem_bytes>>>(
                (const uchar3*)d_in, (float3*)d_out, w, h, d_k_, kCols_);

        } else if (channels == 1) {
            size_t shared_mem_bytes = tile_width * tile_height * sizeof(unsigned char);
            conv2dSharedMemKernel_Final<unsigned char, float><<<gridDim, blockDim, shared_mem_bytes>>>(
                (const unsigned char*)d_in, (float*)d_out, w, h, d_k_, kCols_);

        } else {
             CV_Assert(false && "Shared memory kernel only supports 1 or 3 channel images.");
        }

    } else { // Use the original naive kernel
        conv2dKernel<<<gridDim, blockDim>>>(d_in, (float*)d_out, w, h, channels, d_k_, kCols_, kRows_);
    }

    checkCuda(cudaGetLastError());
    
    cv::Mat out(h, w, CV_32FC(channels));
    checkCuda(cudaMemcpy(out.data, d_out, outputBytes, cudaMemcpyDeviceToHost));
    
    cudaFree(d_in);
    cudaFree(d_out);
    
    return out;
}
