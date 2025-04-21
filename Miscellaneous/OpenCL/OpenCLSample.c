#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>

#define N 1024

const char* kernelSource = 
"__kernel void vec_add(__global float* A, __global float* B, __global float* C) {\n"
"    int id = get_global_id(0);\n"
"    C[id] = A[id] + B[id];\n"
"}\n";

int main() {
    float *A = (float*)malloc(sizeof(float) * N);
    float *B = (float*)malloc(sizeof(float) * N);
    float *C = (float*)malloc(sizeof(float) * N);

    // STEP 1 & 2: Discover and initialize the platform and device
    cl_platform_id platform_id;
    cl_device_id device_id;
    cl_uint ret_num_platforms, ret_num_devices;

    clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
    clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, &ret_num_devices);

    // STEP 3: Create an OpenCL context
    cl_context context = clCreateContext(NULL, 1, &device_id, NULL, NULL, NULL);

    // STEP 4: Create a command queue
    cl_command_queue command_queue = clCreateCommandQueue(context, device_id, 0, NULL);

    // Initialize data
    for (int i = 0; i < N; i++) {
        A[i] = i;
        B[i] = i * 2;
    }

    // STEP 5: Create memory buffers
    cl_mem bufferA = clCreateBuffer(context, CL_MEM_READ_ONLY, N * sizeof(float), NULL, NULL);
    cl_mem bufferB = clCreateBuffer(context, CL_MEM_READ_ONLY, N * sizeof(float), NULL, NULL);
    cl_mem bufferC = clCreateBuffer(context, CL_MEM_WRITE_ONLY, N * sizeof(float), NULL, NULL);

    // STEP 6: Write data to device buffers
    clEnqueueWriteBuffer(command_queue, bufferA, CL_TRUE, 0, N * sizeof(float), A, 0, NULL, NULL);
    clEnqueueWriteBuffer(command_queue, bufferB, CL_TRUE, 0, N * sizeof(float), B, 0, NULL, NULL);

    // STEP 7: Create and compile the program
    cl_program program = clCreateProgramWithSource(context, 1, &kernelSource, NULL, NULL);
    clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);

    // STEP 8: Create the kernel
    cl_kernel kernel = clCreateKernel(program, "vec_add", NULL);

    // STEP 9: Set kernel arguments
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufferA);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufferB);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufferC);

    // STEP 10: Configure work-item structure
    size_t global_item_size = N;
    size_t local_item_size = 64;

    // STEP 11: Enqueue the kernel
    clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global_item_size, &local_item_size, 0, NULL, NULL);

    // STEP 12: Read the result back to host
    clEnqueueReadBuffer(command_queue, bufferC, CL_TRUE, 0, N * sizeof(float), C, 0, NULL, NULL);

    // Print a few output elements
    printf("Result (first 10 elements):\n");
    for (int i = 0; i < 10; i++) {
        printf("C[%d] = %.1f\n", i, C[i]);
    }

    // STEP 13: Release OpenCL resources
    clFlush(command_queue);
    clFinish(command_queue);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseMemObject(bufferA);
    clReleaseMemObject(bufferB);
    clReleaseMemObject(bufferC);
    clReleaseCommandQueue(command_queue);
    clReleaseContext(context);

    // Free host memory
    free(A);
    free(B);
    free(C);

    return 0;
}
