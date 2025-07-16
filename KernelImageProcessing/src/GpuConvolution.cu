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


template <typename InType, typename OutType>
__global__ void conv2dSharedMemKernel(const InType* in, OutType* out,
                                      int width, int height,
                                      const float* kernel, int k_side) {
    
    const int HALO = k_side / 2;
    const int TILE_DIM = 16; // Block dimension is fixed at 16 for this kernel
    const int SHARED_MEM_DIM = TILE_DIM + 2 * HALO;

    extern __shared__ unsigned char tile_raw[]; // Raw byte buffer
    InType* tile = (InType*)tile_raw; // Cast raw buffer to the correct type

    int g_x = blockIdx.x * TILE_DIM + threadIdx.x;
    int g_y = blockIdx.y * TILE_DIM + threadIdx.y;
    int l_x = threadIdx.x;
    int l_y = threadIdx.y;

    // Load data into shared memory tile
    for (int i = l_y; i < SHARED_MEM_DIM; i += TILE_DIM) {
        for (int j = l_x; j < SHARED_MEM_DIM; j += TILE_DIM) {
            int load_g_x = (blockIdx.x * TILE_DIM) - HALO + j;
            int load_g_y = (blockIdx.y * TILE_DIM) - HALO + i;

            int tile_idx = i * SHARED_MEM_DIM + j;
            if (load_g_x >= 0 && load_g_x < width && load_g_y >= 0 && load_g_y < height) {
                tile[tile_idx] = in[load_g_y * width + load_g_x];
            } else {
                // Zero-out the padding area
                memset(&tile[tile_idx], 0, sizeof(InType));
            }
        }
    }
    
    __syncthreads();

    if (g_x < width && g_y < height) {
        OutType sum;
        // Zero-out the accumulator struct/variable
        memset(&sum, 0, sizeof(OutType));

        for (int m = 0; m < k_side; ++m) {
            for (int n = 0; n < k_side; ++n) {
                InType pixel = tile[(l_y + m) * SHARED_MEM_DIM + (l_x + n)];
                float k_val = kernel[m * k_side + n];
                
                // This part requires a small helper for different types
                // We handle it with another template function or if constexpr
                if constexpr (sizeof(InType) == 1) { // Grayscale case (uchar -> float)
                    sum += pixel * k_val;
                } else { // Color case (uchar3 -> float3)
                    sum.x += pixel.x * k_val;
                    sum.y += pixel.y * k_val;
                    sum.z += pixel.z * k_val;
                }
            }
        }
        out[g_y * width + g_x] = sum;
    }
}


// --- IMPLEMENTATION OF FUNCTIONS DECLARED IN THE HEADER ---

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

    dim3 gridDim((w + blockDim.x - 1) / blockDim.x, (h + blockDim.y - 1) / blockDim.y);
    if (maxGridDimX > 0 && maxGridDimX < gridDim.x) {
        gridDim.x = maxGridDimX;
    }

    // --- KERNEL DISPATCH LOGIC ---
    if (use_shared_mem_) {
        // Shared memory kernel requires 16x16 blocks for this implementation
        CV_Assert(blockDim.x == 16 && blockDim.y == 16 && "Shared memory kernel is optimized for 16x16 blocks.");
        
        int halo = kCols_ / 2;
        int shared_mem_tile_dim = blockDim.x + 2 * halo;
        
        if (channels == 3) {
            size_t shared_mem_bytes = shared_mem_tile_dim * shared_mem_tile_dim * sizeof(uchar3);
            
            // Instantiate and launch the 3-channel version of the template
            conv2dSharedMemKernel<uchar3, float3><<<gridDim, blockDim, shared_mem_bytes>>>(
                (const uchar3*)d_in, (float3*)d_out, w, h, d_k_, kCols_);

        } else if (channels == 1) {
            size_t shared_mem_bytes = shared_mem_tile_dim * shared_mem_tile_dim * sizeof(unsigned char);
            
            // Instantiate and launch the 1-channel version of the template
            conv2dSharedMemKernel<unsigned char, float><<<gridDim, blockDim, shared_mem_bytes>>>(
                (const unsigned char*)d_in, (float*)d_out, w, h, d_k_, kCols_);

        } else {
             CV_Assert(false && "Shared memory kernel only supports 1 or 3 channel images.");
        }

    } else { // Use the original naive kernel
        conv2dKernel<<<gridDim, blockDim>>>(
            d_in, (float*)d_out, w, h, channels, d_k_, kCols_, kRows_);
    }

    checkCuda(cudaGetLastError());
    
    cv::Mat out(h, w, CV_32FC(channels));
    checkCuda(cudaMemcpy(out.data, d_out, outputBytes, cudaMemcpyDeviceToHost));
    
    cudaFree(d_in);
    cudaFree(d_out);
    
    return out;
}
