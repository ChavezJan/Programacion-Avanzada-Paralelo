#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>


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




__global__ void sum_array_gpu(int* a, int* b, int* c, int* d,int size)
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
        d[gid] = a[gid] + b[gid] + c[gid];
        printf("Gid: %d, Valores: %d\n", gid, d[gid]);
    }



}
void sum_array_cpu(int* a, int* b, int* c, int* d,int size) {

    for (int x = 0; x < size; x++) {

        d[x] = a[x] + b[x] + c[x];
    }

}
int sum_array_cpu_NEW( int a, int b, int c) {
   
    int d = 0;

    d = a + b + c;

    return d;

}

void query_device() {
    int d_Count = 0;
    cudaGetDeviceCount(&d_Count);

    if (d_Count == 0)
    {
        printf("No CUDA support device found!\n\r");
    }

    for (int devNo = 0; devNo < d_Count; devNo++)
    {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, devNo);
        printf("Device Number:   %d\n", devNo);
        printf("Device name:     %s\n", prop.name);
        printf("No. of MultiProcessors:  %d\n", prop.multiProcessorCount);
        printf("Compute Capabiliuty:  %d, %d\n", prop.major, prop.minor);
        printf("Memory Clock Rate (KHz): %d\n", prop.memoryClockRate);
        printf("Memory Bus Width (bits): %d\n", prop.memoryBusWidth);
        printf("Peak Memory Bandwith (GB/s): %8.2\n", 2.0 * prop.memoryClockRate + (prop.memoryBusWidth / 8) / 1.0e6);
        printf("Total amount of Global Memory: %dKB\n", prop.totalGlobalMem / 1024);
        printf("Total amount of const Memory: %dKB\n", prop.totalConstMem / 1024);
        printf("Total of Shared Memory per block: %dKB\n", prop.sharedMemPerBlock / 1024);
        printf("Total of Shared Memory per MP: %dKB\n", prop.sharedMemPerMultiprocessor / 1024);
        printf("Warp Size: %d\n", prop.warpSize);
        printf("Max. threads per block: %d\n", prop.maxThreadsPerBlock);
        printf("Max. threads per MP: %d\n", prop.maxThreadsPerMultiProcessor);
        printf("Maximum number of warps per multiprocessor: %d\n", prop.maxThreadsPerMultiProcessor / 32);
        printf("Maximum Grid size: (%d, %d, %d)\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
        printf("Maximum block dimension: (%d, %d, %d)\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);

    }
}

/*
* Conocimientos basicos del paralelismo
* concurrencia
* procesos
* dependencia
* ruta critica
* que son los hilos
*
*/
// h - host - cpu
// g - global - gpu

// _ - host
// d_ - global
//

int main() {

    const int n = 1000;
    int size = sizeof(int) * n;


    int* a = (int*)malloc(size);
    int* b = (int*)malloc(size);
    int* c = (int*)malloc(size);
    int* d = (int*)malloc(size);
    int* d_gpu_result = (int*)malloc(size);


    int* d_a;
    int* d_b;
    int* d_c;
    int* d_d;

    for (int x = 0; x < n; x++) {
        a[x] = x;
        b[x] = x;
        c[x] = x;
        d[x] = 0;
    }


    // malloc Cuda
    GPUErrorAssertion(cudaMalloc(&d_a, size));
    GPUErrorAssertion(cudaMalloc(&d_b, size));
    GPUErrorAssertion(cudaMalloc(&d_c, size));
    GPUErrorAssertion(cudaMalloc(&d_d, size));

    //Cuda Memcopy host to device: d_c <- c

    GPUErrorAssertion(cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice));
    GPUErrorAssertion(cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice));
    GPUErrorAssertion(cudaMemcpy(d_c, c, size, cudaMemcpyHostToDevice));
    GPUErrorAssertion(cudaMemcpy(d_d, d, size, cudaMemcpyHostToDevice));

    clock_t gpu_s, gpu_e;

    dim3 gridGid3DSuma(5, 4, 4); //128 / 3D -> 10k hilos *32
    dim3 blockGid3DSuma(32, 2, 2); //10K / 128 / 3D -> 128

    printf("\nSUMA 3D\n");
    gpu_s = clock();
    GPUErrorAssertion(sum_array_gpu << <gridGid3DSuma, blockGid3DSuma >> > (d_a, d_b, d_c, d_d, n));
    check_CUDA_error("Error en lanzamiento del kernel");
    GPUErrorAssertion(cudaDeviceSynchronize());
    gpu_e = clock();

    double cps_fpu = (double)((double)(gpu_e - gpu_s) / CLOCKS_PER_SEC);

    printf("Execution Time: %4.6f", cps_fpu);

    //Cuda Memcopy device to host: c <- d_c
    GPUErrorAssertion(cudaMemcpy(d_gpu_result, d_d, size, cudaMemcpyDeviceToHost));

    //sum_array_cpu(a, b, c, d, n);

    for (int x = 0; x < n; x++) {
        d[x] = sum_array_cpu_NEW(a[x],b[x],c[x]);
        //printf("Valor en x: %d -> %d\n", x, d[x]);

    }
   
    for (int x = 0; x < n; x++) {
        printf("cpu: %d -- %d :gpu\n", d[x], d_gpu_result[x]);
        if (d_gpu_result[x] != d[x])
        {
            cout << "\nERROR\n\n";
            return(0);
        }
    }
    cout << "\nSi no murio entonces el GPU y CPU es lo mismo\n";

    //Cuda free
    GPUErrorAssertion(cudaFree(d_a));
    GPUErrorAssertion(cudaFree(d_b));
    GPUErrorAssertion(cudaFree(d_c));
    GPUErrorAssertion(cudaFree(d_d));

    free(a);
    free(b);
    free(c);
    free(d);
    free(d_gpu_result);
    GPUErrorAssertion(cudaDeviceReset());

    return 0;
}