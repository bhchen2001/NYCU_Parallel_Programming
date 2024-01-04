#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
extern "C"{
#include "hostFE.h"
}

#define BLOCK_SIZE 32

__global__ void convolution(
    int filterWidth, float *filter, int imageHeight, int imageWidth,
    float *inputImage, float *outputImage){

    // __shared__ float shared_filter[100];
    // for (int i = 0; i < filterWidth * filterWidth; i++) {
    //     shared_filter[i] = filter[i];
    // }
    // __syncthreads();

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    __shared__ float shared_filter[100];
    if (x < filterWidth*filterWidth) {
        shared_filter[x] = filter[x];
    }
    __syncthreads();

    if (x >= imageWidth || y >= imageHeight) return;

    int halfWidth = filterWidth / 2;
    float sum = 0.0f;

    int x_img_start = x - halfWidth < 0 ? 0 : x - halfWidth;
    int x_img_end = x + filterWidth - halfWidth - 1 >= imageWidth ? imageWidth - 1 : x + filterWidth - halfWidth - 1;
    int y_img_start = y - halfWidth < 0 ? 0 : y - halfWidth;
    int y_img_end = y + filterWidth - halfWidth - 1 >= imageHeight ? imageHeight - 1 : y + filterWidth - halfWidth - 1;
    int x_filter_start = x - halfWidth < 0 ? halfWidth - x : 0;
    int x_filter_end = x + filterWidth - halfWidth - 1 >= imageWidth ? filterWidth - 1 - (x + filterWidth - halfWidth - 1 - imageWidth) : filterWidth - 1;
    int y_filter_start = y - halfWidth < 0 ? halfWidth - y : 0;
    int y_filter_end = y + filterWidth - halfWidth - 1 >= imageHeight ? filterWidth - 1 - (y + filterWidth - halfWidth - 1 - imageHeight) : filterWidth - 1;

    for(int j = y_img_start; j <= y_img_end; j++){
        int filter_index = (j - y_img_start + y_filter_start) * filterWidth;
        int image_index = j * imageWidth;
        for(int i = x_img_start; i <= x_img_end; i++){
            if(!filter[i - x_img_start + x_filter_start + filter_index]){
                continue;
            }
            sum += filter[i - x_img_start + x_filter_start + filter_index] * inputImage[image_index + i];
        }
    }

    outputImage[y * imageWidth + x] = sum;
}

extern "C"
void hostFE(int filterWidth, float *filter, int imageHeight, int imageWidth,
            float *inputImage, float *outputImage){
    
    int img_size = imageHeight * imageWidth;
    int filter_size = filterWidth * filterWidth;

    float *d_filter, *d_inputImage, *d_outputImage;
    cudaMalloc((void **)&d_filter, filter_size * sizeof(float));
    cudaMalloc((void **)&d_inputImage, img_size * sizeof(float));

    cudaHostRegister((void *)outputImage, img_size * sizeof(float), cudaHostRegisterDefault);
    cudaHostGetDevicePointer((void **)&d_outputImage, (void *)outputImage, 0);

    cudaMemcpy(d_filter, filter, filter_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_inputImage, inputImage, img_size * sizeof(float), cudaMemcpyHostToDevice);

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(imageWidth / dimBlock.x, imageHeight / dimBlock.y);

    convolution<<<dimGrid, dimBlock>>>(filterWidth, d_filter, imageHeight, imageWidth, d_inputImage, d_outputImage);

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("Error: %s\n", cudaGetErrorString(err));
        exit(1);
    }

    // cudaFree(d_filter);
    // cudaFree(d_inputImage);
    // cudaHostUnregister(outputImage);
}