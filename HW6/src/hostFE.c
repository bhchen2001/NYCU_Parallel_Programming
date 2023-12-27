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

    /* get platfrom ids */
    // cl_uint num_platforms;
    // cl_platform_id platform;
    // if(clGetPlatformIDs(1, &platform, &num_platforms) != CL_SUCCESS)
    // {
    //     printf("Error: Failed to find an OpenCL platform!\n");
    //     exit(1);
    // }

    /* get the num of GPU */
    // cl_uint num_devices;
    // if(clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, NULL, &num_devices) != CL_SUCCESS)
    // {
    //     printf("Error: Failed to find an OpenCL device!\n");
    //     exit(1);
    // }

    // printf("num_platforms: %d\n", num_platforms);
    // printf("num_devices: %d\n", num_devices);
    
    /* choose GPU device */
    // if(clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, num_devices, device, NULL) != CL_SUCCESS)
    // {
    //     printf("Error: Failed to create a device group!\n");
    //     exit(1);
    // }

    /* declare context */
    // cl_context_properties properties[] = {CL_CONTEXT_PLATFORM, (cl_context_properties)platform, 0};
    // *context = clCreateContextFromType(properties, CL_DEVICE_TYPE_GPU, NULL, NULL, &status);
    // if(*context == 0)
    // {
    //     printf("Error: Failed to create a compute context!\n");
    //     exit(1);
    // }
    /* choose device from context */
    // status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, device, NULL);
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
    /* transfer input data to device */
    // status = clEnqueueWriteBuffer(command_queue, input_buffer, CL_TRUE, 0, img_size * sizeof(float), (void *)inputImage, 0, NULL, NULL);
    // status = clEnqueueWriteBuffer(command_queue, filter_buffer, CL_TRUE, 0, filter_size * sizeof(float), (void *)filter, 0, NULL, NULL);

    /* read kernel source file */
    FILE *file;
    size_t source_size;
    char *source_str;
    file = fopen("kernel.cl", "r");
    if(file == NULL){
        printf("Failed to load kernel.\n");
        exit(1);
    }

    /* get the length of source */
    // fseek(file, 0, SEEK_END);
    // source_size = ftell(file);
    // fseek(file, 0, SEEK_SET);

    // printf("source_size: %d\n", source_size);

    /* read the source file */
    // source_str = (char *)malloc(source_size + 1); // +1 for '\0'
    // if(fread(source_str, source_size, 1, file) == 0){
    //     printf("Failed to read kernel.\n");
    //     exit(1);
    // }
    // fclose(file);
    // source_str[source_size] = '\0';

    /* create program object */
    // *program = clCreateProgramWithSource(*context, 1, (const char **)&source_str, (const size_t *)&source_size, &status);
    // if(*program == NULL){
    //     printf("Failed to create program.\n");
    //     exit(1);
    // }

    /* build program */
    // status = clBuildProgram(*program, 0, NULL, NULL, NULL, NULL);
    // if(status != CL_SUCCESS){
    //     printf("Failed to build program.\n");
    //     exit(1);
    // }

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
    if(clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&filter_buffer) != CL_SUCCESS){
        printf("Failed to set kernel arguments: filter_buffer.\n");
        exit(1);
    }
    if(clSetKernelArg(kernel, 2, sizeof(cl_int), (void *)&imageHeight) != CL_SUCCESS){
        printf("Failed to set kernel arguments: imageHeight.\n");
        exit(1);
    }
    if(clSetKernelArg(kernel, 3, sizeof(cl_int), (void *)&imageWidth) != CL_SUCCESS){
        printf("Failed to set kernel arguments: imageWidth.\n");
        exit(1);
    }
    if(clSetKernelArg(kernel, 4, sizeof(cl_mem), (void *)&input_buffer) != CL_SUCCESS){
        printf("Failed to set kernel arguments: input_buffer.\n");
        exit(1);
    }
    if(clSetKernelArg(kernel, 5, sizeof(cl_mem), (void *)&output_buffer) != CL_SUCCESS){
        printf("Failed to set kernel arguments: output_buffer.\n");
        exit(1);
    }

    /* set workgroup size */
    size_t local_work_size[2] = {40, 25};
    size_t global_work_size[2] = {imageWidth, imageHeight};

    /* execute kernel */
    if(clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, global_work_size, local_work_size, 0, NULL, NULL) != CL_SUCCESS){
        printf("Failed to execute kernel.\n");
        exit(1);
    }

    /* read output data */
    if(clEnqueueReadBuffer(command_queue, output_buffer, CL_TRUE, 0, img_size * sizeof(float), (void *)outputImage, 0, NULL, NULL != CL_SUCCESS)){
        printf("Failed to read output data.\n");
        exit(1);
    }

    /* release resources */
    // status = clReleaseKernel(kernel);
    // status = clReleaseProgram(*program);
    // status = clReleaseMemObject(input_buffer);
    // status = clReleaseMemObject(filter_buffer);
    // status = clReleaseMemObject(output_buffer);
    // status = clReleaseCommandQueue(command_queue);
    // status = clReleaseContext(*context);

    return;
}