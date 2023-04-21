%%cu
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>
#include <time.h>

#define BLOCK_SIZE 1024

__global__ void search(int* a, int n, int* pos, int find) {
    __shared__ int shared_a[BLOCK_SIZE];

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int start = blockIdx.x * blockDim.x;
    int end = start + blockDim.x;

    if (end > n) {
        end = n;
    }

    for (int i = threadIdx.x; i < BLOCK_SIZE; i += blockDim.x) {
        if (i + start < n) {
            shared_a[i] = a[i + start];
        }
    }

    __syncthreads();

    for (int i = start; i < end; i++) {
        if (shared_a[i - start] == find) {
            int old_pos = atomicExch(pos, i);
            if (old_pos == -1) {
                return;
            }
        }
    }
}

int main() {
    int size = 32;
    int find = 24;
    int* h_a, * res, * pos;
    int* d_a, * d_pos;

    h_a = (int*)malloc(size * sizeof(int));
    pos = (int*)malloc(sizeof(int));
    pos[0] = -1;
    res = (int*)malloc(size * sizeof(int));

    cudaMalloc(&d_a, size * sizeof(int));
    cudaMalloc(&d_pos, sizeof(int));

    srand(time(NULL));

        for (int i = 0; i < size; i++) {
        int r1 = (size - i);
        h_a[i] = r1;
        printf("%d ", h_a[i]);
    }

    printf("\n");

    cudaMemcpy(d_a, h_a, size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pos, pos, sizeof(int), cudaMemcpyHostToDevice);

    dim3 grid(size >= 1024 ? size / 1024 : 1);
    dim3 block(1024);

    printf("Valor a encontrar: ");


    search << <grid, block >> > (d_a, size, d_pos, find);

    cudaDeviceSynchronize();

    cudaMemcpy(pos, d_pos, sizeof(int), cudaMemcpyDeviceToHost);

    if (pos[0] == -1) {
        printf("No se encontro el valor\n");
    }
    else {
        printf("Encontrado en el indice %d \n", pos[0]);
    }

    free(h_a);
    free(pos);
    free(res);
    cudaFree(d_a);
    cudaFree(d_pos);

    return 0;
}
