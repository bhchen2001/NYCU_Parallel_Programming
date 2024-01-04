#include <stdio.h>
#include <stdlib.h>
#include "hostFE.h"
#include "helper.h"

#define WS_SIZE 4

void hostFE(int filterWidth, float *filter, int imageHeight, int imageWidth,
            float *inputImage, float *outputImage, cl_device_id *device,
            cl_context *context, cl_program *program)
{
    cl_int status;
    int filter_size = filterWidth * filterWidth;
    int img_size = imageHeight * imageWidth;
    int half_filter_width = filterWidth / 2;

    /* using device and context to create a command queue */
    cl_command_queue command_queue = clCreateCommandQueue(*context, *device, 0, &status);

    /* create buffers on device */
    cl_mem input_buffer = clCreateBuffer(*context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, img_size * sizeof(float), inputImage, &status);
    cl_mem filter_buffer = clCreateBuffer(*context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, filter_size * sizeof(float), filter, &status);
    cl_mem output_buffer = clCreateBuffer(*context, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR, img_size * sizeof(float), outputImage, &status);
    if(status != CL_SUCCESS)
    {
        printf("Error: Failed to allocate device memory!\n");
        exit(1);
    }

    /* create kernel */
    cl_kernel kernel = clCreateKernel(*program, "convolution", &status);
    if(kernel == NULL){
        printf("Failed to create kernel.\n");
        exit(1);
    }

    /* set kernel arguments */
    if(clSetKernelArg(kernel, 0, sizeof(cl_int), (void *)&filterWidth) != CL_SUCCESS){
        printf("Failed to set kernel arguments: filterWidth.\n");
        exit(1);
    }
    if(clSetKernelArg(kernel, 1, sizeof(cl_int), (void *)&half_filter_width) != CL_SUCCESS){
        printf("Failed to set kernel arguments: filter_buffer.\n");
        exit(1);
    }
    if(clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&filter_buffer) != CL_SUCCESS){
        printf("Failed to set kernel arguments: filter_buffer.\n");
        exit(1);
    }
    if(clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *)&input_buffer) != CL_SUCCESS){
        printf("Failed to set kernel arguments: input_buffer.\n");
        exit(1);
    }
    if(clSetKernelArg(kernel, 4, sizeof(cl_mem), (void *)&output_buffer) != CL_SUCCESS){
        printf("Failed to set kernel arguments: output_buffer.\n");
        exit(1);
    }

    /* set workgroup size */
    size_t local_work_size[2] = {8, 8};
    size_t global_work_size[2] = {imageWidth, imageHeight};

    /* execute kernel */
    if(clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, global_work_size, local_work_size, 0, NULL, NULL) != CL_SUCCESS){
        printf("Failed to execute kernel.\n");
        exit(1);
    }

    /* read output data */
    if(clEnqueueReadBuffer(command_queue, output_buffer, CL_TRUE, 0, img_size * sizeof(float), (void *)outputImage, 0, NULL, NULL) != CL_SUCCESS){
        printf("Failed to read output data.\n");
        exit(1);
    }

    /* release resources */
    // status = clReleaseKernel(kernel);
    // status = clReleaseMemObject(input_buffer);
    // status = clReleaseMemObject(filter_buffer);
    // status = clReleaseMemObject(output_buffer);
    // status = clReleaseCommandQueue(command_queue);
    // status = clReleaseContext(*context);
    // status = clReleaseProgram(*program);

    return;
}