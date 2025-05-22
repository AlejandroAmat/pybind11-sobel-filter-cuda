#include <cuda.h>
#include "sobel_filter.h"
#include <stdio.h>
#include "cuda_runtime.h"

#define BLOCK_X 32
#define BLOCK_Y 32
#define SHARED_X (BLOCK_X + 2) 
#define SHARED_Y (BLOCK_Y + 2)

template <int32_t C>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
filter(float* __restrict__ input,    
       float* __restrict__ output,   
       const int width, 
       const int height) 
{
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int gx = bx * BLOCK_X + tx;
    const int gy = by * BLOCK_Y + ty;

    __shared__ float tile[SHARED_Y][SHARED_X];
    const int total_shared_elements = SHARED_Y * SHARED_X;
    const int threads_per_block = BLOCK_X * BLOCK_Y;

    if (gx < width && gy < height) {
        for (int c = 0; c < C; c++) {
            const int channel_offset = c * height * width;
            
            // Load shared memory - First pass
            {
                int tid = ty * BLOCK_X + tx;
                if (tid < total_shared_elements) {
                    int shared_y = tid / SHARED_X;
                    int shared_x = tid % SHARED_X;
                    
                    int global_y = by * BLOCK_Y + shared_y - 1;
                    int global_x = bx * BLOCK_X + shared_x - 1;
                    
                    float value = 0.0f;
                    if (global_y >= 0 && global_y < height && 
                        global_x >= 0 && global_x < width) {
                        value = input[channel_offset + global_y * width + global_x];  // Read from INPUT
                    }
                    tile[shared_y][shared_x] = value;
                }
            }
            
            // Load shared memory - Second pass
            {
                int tid = threads_per_block + ty * BLOCK_X + tx;
                if (tid < total_shared_elements) {
                    int shared_y = tid / SHARED_X;
                    int shared_x = tid % SHARED_X;
                    
                    int global_y = by * BLOCK_Y + shared_y - 1;
                    int global_x = bx * BLOCK_X + shared_x - 1;
                    
                    float value = 0.0f;
                    if (global_y >= 0 && global_y < height && 
                        global_x >= 0 && global_x < width) {
                        value = input[channel_offset + global_y * width + global_x];  // Read from INPUT
                    }
                    tile[shared_y][shared_x] = value;
                }
            }
            
            __syncthreads();
            
            float gx_val = tile[ty][tx] - tile[ty][tx+2] +
                          2.0f * (tile[ty+1][tx] - tile[ty+1][tx+2]) +
                          tile[ty+2][tx] - tile[ty+2][tx+2];

            float gy_val = tile[ty][tx] + 2.0f * tile[ty][tx+1] + tile[ty][tx+2] -
                          tile[ty+2][tx] - 2.0f * tile[ty+2][tx+1] - tile[ty+2][tx+2];

            int output_idx = channel_offset + gy * width + gx;
            output[output_idx] = sqrtf(gx_val * gx_val + gy_val * gy_val);  // Write to OUTPUT
        }
    }
}

void apply_filter(const torch::Tensor& image, torch::Tensor& output) {
    float* image_ptr = image.data_ptr<float>();
    float* out_ptr = output.data_ptr<float>();

    const int C = image.size(0);
    const int H = image.size(1);  
    const int W = image.size(2);  

    const dim3 tile_grid((W + BLOCK_X - 1)/BLOCK_X, (H + BLOCK_Y - 1)/BLOCK_Y, 1);
    const dim3 block(BLOCK_X, BLOCK_Y);

    switch(C) {
        case 1:
            filter<1><<<tile_grid, block>>>(image_ptr, out_ptr, W, H);  // Fixed parameter order
            break;
        case 3:
            filter<3><<<tile_grid, block>>>(image_ptr, out_ptr, W, H);  // Fixed parameter order
            break;
        default:
            throw std::runtime_error("Unsupported channel count");
    }
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err));
    }
}