#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <random>
#include <stdio.h>
#include <iostream>
#include <time.h>

using namespace std;

// Errores

#define GPUErrorAssertion(ans){gpuAssert((ans),__FILE__, __LINE__);}

inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
    if (code != cudaSuccess) {}
    {
        fprintf(stderr, "GPUassert: %s %s %d\n\r", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}
__host__ void check_CUDA_error(const char* e) {
    cudaError_t error;
    cudaDeviceSynchronize();
    error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        printf("\nERROR %d: %s (%s)", error, cudaGetErrorString(error), e);
    }
}


__global__ void convolution(int* a, int* k, int* c, int n, int kSize) {
    int row = threadIdx.y;
    int col = threadIdx.x;

    int suma = 0;
    if (row > 0 && row < n - 1 && col>0 && col < n - 1) {
        for (int i = 0; i < kSize; i++) {
            for (int j = 0; j < kSize; j++) {
                suma += (a[(row ) * n + i + (col ) + j] * k[i * kSize + j]);            
            }
        }
        c[row * n + col] = suma;
    }
}

int main() {

    const int n = 8;
    const int kLength = 3;
    int size = n * n * sizeof(int);
    int sizeK = kLength * kLength * sizeof(int);
    int* host_a, * host_c, * host_kernel;
    int* dev_a, * dev_c, * dev_kernel;
    host_a = (int*)malloc(size);
    host_c = (int*)malloc(size);
    host_kernel = (int*)malloc(sizeK);
    cudaMalloc(&dev_a, size);
    check_CUDA_error("Error");
    cudaMalloc(&dev_c, size);
    check_CUDA_error("Error");
    cudaMalloc(&dev_kernel, sizeK);
    check_CUDA_error("Error");
    for (int i = 0; i < n * n; i++) {
        int r = (rand() % (3));
        host_a[i] = r;
        host_c[i] = r;
    }

    host_kernel[0] = 0;
    host_kernel[1] = 1;
    host_kernel[2] = 0;
    host_kernel[3] = 0;
    host_kernel[4] = 0;
    host_kernel[5] = 0;
    host_kernel[6] = 0;
    host_kernel[7] = 0;
    host_kernel[8] = 0;

   
    cout << "old:\n";
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%d ", host_a[i * n + j]);
        }
        printf("\n");
    }
   

    cudaMemcpy(dev_a, host_a, size, cudaMemcpyHostToDevice);
    check_CUDA_error("Error");
    cudaMemcpy(dev_c, host_c, size, cudaMemcpyHostToDevice);
    check_CUDA_error("Error");
    cudaMemcpy(dev_kernel, host_kernel, sizeK, cudaMemcpyHostToDevice);
    check_CUDA_error("Error");
   
    dim3 grid(1, 1, 1);
    dim3 block(8, 8, 1);
    convolution << <grid, block >> > (dev_a, dev_kernel, dev_c, n, kLength);
    check_CUDA_error("Error");
    cudaMemcpy(host_c, dev_c, size, cudaMemcpyDeviceToHost);
    check_CUDA_error("Error");

    cudaDeviceSynchronize();
    check_CUDA_error("Error");
    cudaDeviceReset();
    check_CUDA_error("Error");

    cout << "new:\n";
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            cout << host_c[i * n + j] << " ";
        }
        cout << "\n";
    }
    free(host_a);
    free(host_c);
    free(host_kernel);
 


    //return 0;
}
