#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>


using namespace std;

// Errores

__host__ void check_CUDA_error(const char* e) {
    cudaError_t error;
    cudaDeviceSynchronize();
    error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        printf("\nERROR %d: %s (%s)", error, cudaGetErrorString(error), e);
    }
}


__global__ void idx_calc_gid3D(int* a)
{
    int totalThreads = blockDim.x * blockDim.y * blockDim.z;

    int tid = threadIdx.x
        + threadIdx.y * blockDim.x
        + threadIdx.z * blockDim.x * blockDim.y;

    int bid = blockIdx.x
        + blockIdx.y * gridDim.x
        + blockIdx.z * gridDim.x * gridDim.y;

    int gid = tid + bid * totalThreads;

    printf("Gid: %d, Valores: %d\n", gid, a[gid]);


}

__global__ void sum_array_gpu(int* a, int* b, int* c, int size)
{
    int totalThreads = blockDim.x * blockDim.y * blockDim.z;

    int tid = threadIdx.x
        + threadIdx.y * blockDim.x
        + threadIdx.z * blockDim.x * blockDim.y;

    int bid = blockIdx.x
        + blockIdx.y * gridDim.x
        + blockIdx.z * gridDim.x * gridDim.y;

    int gid = tid + bid * totalThreads;

    if (gid < size)
    {
        c[gid] = a[gid] + b[gid];
      printf("Gid: %d, Valores: %d\n", gid, c[gid]);
    }



}
void sum_array_cpu(int* a, int* b, int* c, int size) {

    for (int x = 0; x < size; x++) {

        c[x] = a[x] + b[x];
    }

}


// h - host - cpu 
// g - global - gpu

// _ - host
// d_ - global
//

int main() {

    const int n = 100000;
    int size = sizeof(int) * n;


    int* a = (int*)malloc(size);
    int* b = (int*)malloc(size);
    int* c = (int*)malloc(size);
    int* c_gpu_result = (int*)malloc(size);


    int* d_a;
    int* d_b;
    int* d_c;

    for (int x = 0; x < n; x++) {
        a[x] = x;
        b[x] = x;
    }


    // malloc Cuda
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);
    
    //Cuda Memcopy host to device: d_c <- c

    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, c, size, cudaMemcpyHostToDevice);

    dim3 gridGid3D(2, 2, 2); //128 / 3D
    dim3 blockGid3D(2, 2, 2); //10K / 128 / 3D
    clock_t gpu_s, gpu_e;


    printf("GID 3D\n");

    idx_calc_gid3D << <gridGid3D, blockGid3D >> > (d_a);
    check_CUDA_error("Error en lanzamiento del kernel");
    cudaDeviceSynchronize();

    dim3 gridGid3DSuma(5, 4, 4); //128 / 3D -> 10k hilos *32
    dim3 blockGid3DSuma(32, 2, 2); //10K / 128 / 3D -> 128

    printf("\nSUMA 3D\n");
    gpu_s = clock();
    sum_array_gpu << <gridGid3DSuma, blockGid3DSuma >> > (d_a, d_b, d_c,n);
    check_CUDA_error("Error en lanzamiento del kernel");
    cudaDeviceSynchronize();
    gpu_e = clock();

    double cps_fpu = (double)((double)(gpu_e - gpu_s) / CLOCKS_PER_SEC);

    printf("Execution Time: %4.6f",cps_fpu);

    //Cuda Memcopy device to host: c <- d_c
    cudaMemcpy(c_gpu_result, d_c, size, cudaMemcpyDeviceToHost);

    sum_array_cpu(a, b, c, n);

    for (int x = 0; x < n; x++) {
        if (c_gpu_result[x] != c[x])
        {
            cout << "\nERROR\n\n";
            return(0);
        }
    }
    cout << "\nSi no murio entonces el GPU y CPU es lo mismo\n";

    //Cuda free
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    free(a);
    free(b);
    free(c);
    free(c_gpu_result);
    cudaDeviceReset();

    return 0;
}