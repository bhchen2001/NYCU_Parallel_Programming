#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <cmath>

# define MAX_THREADS_PER_BLOCK 1024
# define BLOCK_SIZE 8

__device__ int mandelDevice(float c_re, float c_im, int maxIterations) {
    float z_re = c_re, z_im = c_im;
    int i;

#pragma unroll

    for(i = 0; i < maxIterations; ++i) {
        if(z_re * z_re + z_im * z_im > 4.f) {
            break;
        }
        float new_re = z_re * z_re - z_im * z_im;
        float new_im = 2.f * z_re * z_im;
        z_re = c_re + new_re;
        z_im = c_im + new_im;
    }

    return i;
}

__global__ void mandelKernel(float lowerX, float lowerY, float stepX, float stepY, int* d_img, int resX, int maxIterations) {
    // To avoid error caused by the floating number, use the following pseudo code
    //
    int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
    int tid_y = blockIdx.y * blockDim.y + threadIdx.y;  
    float c_re = lowerX + tid_x * stepX;
    float c_im = lowerY + tid_y * stepY;

    d_img[tid_y * resX + tid_x] = mandelDevice(c_re, c_im, maxIterations);

}

// Host front-end function that allocates the memory and launches the GPU kernel
void hostFE (float upperX, float upperY, float lowerX, float lowerY, int* img, int resX, int resY, int maxIterations)
{
    float stepX = (upperX - lowerX) / resX;
    float stepY = (upperY - lowerY) / resY;

    /* allocate memory on host and device */
    int *d_img;
    size_t size = resX * resY * sizeof(int);
    cudaMalloc((void**)&d_img, size);

    /* setup the execution config */
    /* beware of the limitation of block size */
    dim3 threads_per_block(BLOCK_SIZE, BLOCK_SIZE);
    if (threads_per_block.x * threads_per_block.y > MAX_THREADS_PER_BLOCK) {
        printf("The thread number exceeds the maximum thread number per block.\n");
        exit(1);
    }
    dim3 number_of_blocks(resX / threads_per_block.x, resY / threads_per_block.y);
    mandelKernel<<<number_of_blocks, threads_per_block>>>(lowerX, lowerY, stepX, stepY, d_img, resX, maxIterations);

    cudaError_t cudaerr = cudaDeviceSynchronize();
    if (cudaerr != cudaSuccess)
        printf("kernel launch failed with error \"%s\".\n",
               cudaGetErrorString(cudaerr));

    /* read output from device */
    cudaMemcpy(img, d_img, size, cudaMemcpyDeviceToHost);

    /* free device memory */
    cudaFree(d_img);
}
