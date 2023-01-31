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

__global__ void idx_calc_tid(int* a)
{

    int tid = threadIdx.x;

     printf("threadx - x: %d, data: %d\n", tid, a[tid]);
     /* printf("blockidx - x: %d, y: %d, z: %d\n", blockIdx.x, blockIdx.y, blockIdx.z);
     printf("gridDim - x: %d, y: %d, z: %d\n", gridDim.x, gridDim.y, gridDim.z);*/


}
__global__ void idx_calc_gid(int* a)
{

    int tid = threadIdx.x;
    //gid = tid + offset;
    int offset = blockIdx.x * blockDim.x;
    int gid = tid + offset;


    printf("blockIdx - x: %d, threadx - x: %d, gid: %d, data: %d\n",blockIdx.x,tid, gid, a[gid]);
    /* printf("blockidx - x: %d, y: %d, z: %d\n", blockIdx.x, blockIdx.y, blockIdx.z);
    printf("gridDim - x: %d, y: %d, z: %d\n", gridDim.x, gridDim.y, gridDim.z);*/


}
__global__ void idx_calc_gid2D(int* a)
{

    int tid = threadIdx.x;
    //gid = tid + offset;
    int offsetRow = gridDim.x * blockDim.x * blockIdx.y;
    int offsetBlock = blockIdx.x * blockDim.x ;
    int gid = tid + offsetBlock + offsetRow;


    printf("blockIdx - x: %d, threadx - x: %d, gid: %d, data: %d\n", blockIdx.x, tid, gid, a[gid]);
    /* printf("blockidx - x: %d, y: %d, z: %d\n", blockIdx.x, blockIdx.y, blockIdx.z);
    printf("gridDim - x: %d, y: %d, z: %d\n", gridDim.x, gridDim.y, gridDim.z);*/


}
__global__ void idx_calc_gid2D2(int* a)
{

    int tid = threadIdx.x;
    int n_thBlock = blockDim.x * blockDim.y;
    int n_thRow = n_thBlock * gridDim.x;
    //gid = tid + offset;
    int offsetRow = n_thRow * blockIdx.y;
    int offsetBlock = blockIdx.x * n_thBlock;
    int gid = tid + offsetBlock + offsetRow;


    printf("blockIdx - x: %d, threadx - x: %d, gid: %d, data: %d\n", blockIdx.x, tid, gid, a[gid]);
    /* printf("blockidx - x: %d, y: %d, z: %d\n", blockIdx.x, blockIdx.y, blockIdx.z);
    printf("gridDim - x: %d, y: %d, z: %d\n", gridDim.x, gridDim.y, gridDim.z);*/


}


// h - host - cpu 
// g - global - gpu

// _ - host
// d_ - global
//

int main() {

    const int n = 16;

    int a[n] = { 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16 };

    int size = sizeof(int) * n;

    int* d_a;
  

    // malloc Cuda
    cudaMalloc(&d_a, size);
  
    //Cuda Memcopy host to device: d_c <- c

    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    

    dim3 gridTid(1);
    dim3 blockTid(n,1,1);
    
    printf("TID\n");

    idx_calc_tid << <gridTid, blockTid >> > (d_a);
    check_CUDA_error("Error en lanzamiento del kernel");
    cudaDeviceSynchronize();
    
    
    dim3 gridGid(4,1,1);
    dim3 blockGid(4,1,1);


    printf("GID\n");

    idx_calc_gid << <gridGid, blockGid >> > (d_a);
    check_CUDA_error("Error en lanzamiento del kernel");
    cudaDeviceSynchronize();

    dim3 gridGid2D(2, 2, 1);
    dim3 blockGid2D(4, 1, 1);


    printf("GID 2D\n");

    idx_calc_gid2D << <gridGid2D, blockGid2D >> > (d_a);
    check_CUDA_error("Error en lanzamiento del kernel");
    cudaDeviceSynchronize();

    dim3 gridGid2D2(2, 2, 1);
    dim3 blockGid2D2(2, 2, 1);


    printf("GID 2D2\n");

    idx_calc_gid2D2 << <gridGid2D, blockGid2D >> > (d_a);
    check_CUDA_error("Error en lanzamiento del kernel");
    cudaDeviceSynchronize();

    //Cuda Memcopy device to host: c <- d_c
    //cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    //Cuda free
    cudaFree(d_a);
    cudaDeviceReset();

    return 0;
}