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

#define TILE_DIM 16

__global__ void transpos_no_SM(int* source, int* dest, int size)
{

    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    if (i < size && j < size)
    {
        int des_idx = i * size + j;
        int src_idx = j * size + i;
        dest[des_idx] = source[src_idx];
    }
}

__global__ void transpos_Shared(int* source, int* dest, int size)
{
    __shared__ int tile[TILE_DIM][TILE_DIM + 1];

    // input threads idx
    int i_in = threadIdx.x + blockIdx.x * blockDim.x;
    int j_in = threadIdx.y + blockIdx.y * blockDim.y;
   
    //input index
    int src_idx = j_in * size + i_in;

    // 1D index calculation shared memory
    int _1D_index = threadIdx.y * blockDim.x + threadIdx.x;

    // col major row and col index calculation
    int i_row = _1D_index / blockDim.y;
    int i_col = _1D_index % blockDim.y;

    //coordinate for transpose matrix
    int i_out = blockIdx.y * blockDim.y + threadIdx.x;
    int j_out = blockIdx.x * blockDim.y + threadIdx.y;

    //output index
    int dst_idx = j_out * size + i_out;

    if (i_in < size && j_in < size)
    {
        //load from in array in row major and store to shared
        tile[threadIdx.y][threadIdx.x] = source[src_idx];

        // wait untill all the threads load the values
        __syncthreads();

        dest[dst_idx] = tile[threadIdx.x][threadIdx.y];
    }
}

__global__ void multMatrix(int* sourceA, int* sourceB,int* dest,int size)
{

    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    if (i < size && j < size)
    {
        int des_idx = i * size + j;
        dest[des_idx] = sourceA[des_idx] * sourceA[des_idx];
    }
}


__global__ void convolution(int* a, int* k, int* c, int n, int kSize) {
    int row = threadIdx.y;
    int col = threadIdx.x;

    int suma = 0;
    if (row > 0 && row < n - 1 && col>0 && col < n - 1) {
        for (int i = 0; i < kSize; i++) {
            for (int j = 0; j < kSize; j++) {
                suma += (a[(row)*n + i + (col)+j] * k[i * kSize + j]);
            }
        }
        c[row * n + col] = suma;
    }
}

int main() {

    const int n = 6;
    int size = n * n * sizeof(int);
    int* host_a, * host_c, * host_mulMa;
    int* dev_a, * dev_c ,* dev_mulMa;
   

    host_a = (int*)malloc(size);
    host_c = (int*)malloc(size);
    host_mulMa = (int*)malloc(size);
   
    cudaMalloc(&dev_a, size);
    check_CUDA_error("Error");
    cudaMalloc(&dev_c, size);
    check_CUDA_error("Error");
    cudaMalloc(&dev_mulMa, size);
    check_CUDA_error("Error");

    for (int i = 0; i < n * n; i++) {
        int r = (rand() % (3));
        host_a[i] = i;
        host_c[i] = 0;
    }

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

    dim3 block(n/2, n/2, 1);
    dim3 grid(n/(n/2), n / (n / 2), 1);

    //convolution << <grid, block >> > (dev_a, dev_kernel, dev_c, n, kLength);
    transpos_no_SM << <grid, block >> > (dev_a, dev_c, (n));
    //transpos_Shared << <grid, block >> > (dev_a,dev_c,(n));
    check_CUDA_error("Error");
    cudaMemcpy(host_c, dev_c, size, cudaMemcpyDeviceToHost);
    check_CUDA_error("Error");


    multMatrix << <grid, block >> > (dev_a,dev_c,dev_mulMa,(n));
    check_CUDA_error("Error");
    cudaMemcpy(host_mulMa, dev_mulMa, size, cudaMemcpyDeviceToHost);
    check_CUDA_error("Error");

    cudaDeviceSynchronize();
    check_CUDA_error("Error");
    cudaDeviceReset();
    check_CUDA_error("Error");

    cout << "\n";

    cout << "new:\n";
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            cout << host_c[i * n + j] << " ";
        }
        cout << "\n";
    }

    cout << "\n";

    cout << "Mult Matrix:\n";
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            cout << host_mulMa[i * n + j] << " ";
        }
        cout << "\n";
    }
    free(host_a);
    free(host_c);
    free(host_mulMa);


    //return 0;
}