#include<stdio.h>
#include<stdlib.h>
#include<unistd.h>
#include"k.h"
#include<math.h>

// CUDA Headers
#include </usr/local/cuda/include/cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// Helper definition
#define VAR(v, i) __typeof(i) v=(i)
#define FOR(i, j, k) for (int i = (j); i <= (k); ++i)
#define REP(i, n) for(int i = 0;i <(n); ++i)

// CONSTS
#define INF     1061109567 // 3F 3F 3F 3F
#define CHARINF 63       // 3F    
#define CHARBIT 8
#define NONE    -1

#define CMCPYHTD cudaMemcpyHostToDevice
#define CMCPYDTH cudaMemcpyDeviceToHost

// CONSTS for compute capability 2.0
#define BLOCK_WIDTH 16
#define WARP         32

/** Cuda handle error, if err is not success print error and line in code
*
* @param status CUDA Error types
*/
#define HANDLE_ERROR(err) \
{ \
    if (err != cudaSuccess) \
    { \
        fprintf(stderr, "%s failed  at line %d \nError message: %s \n", \
            __FILE__, __LINE__ ,cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
}

extern  "C" K gpu_floydwarshall(K matrix);

/**Kernel for wake gpu
*
* @param reps dummy variable only to perform some action
*/
__global__ void wake_gpu_kernel(int reps) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= reps) return;
}

/**Kernel for parallel Floyd Warshall algorithm on gpu
* 
* @param u number vertex of which is performed relaxation paths [v1, v2]
* @param n number of vertices in the graph G:=(V,E), n := |V(G)|
* @param d matrix of shortest paths d(G)
*/
__global__ void fw_kernel(const unsigned int u, const unsigned int n, int * const d)
{
    int v1 = blockDim.y * blockIdx.y + threadIdx.y;
    int v2 = blockDim.x * blockIdx.x + threadIdx.x;
    
    if (v1 < n && v2 < n) 
    {
        int newPath = d[v1 * n + u] + d[u * n + v2];
        int oldPath = d[v1 * n + v2];
        if (oldPath > newPath)
        {
            d[v1 * n + v2] = newPath;
        }
    }
}

K gpu_floydwarshall(K matrix)
{
    unsigned int V = sqrt(matrix->n);
    unsigned int n = V;
    // Alloc host data for G - graph, d - matrix of shortest paths
    unsigned int size = V * V;
    int *d = (int *) malloc (sizeof(int) * size);
    int *dev_d = 0;
    cudaError_t cudaStatus;
    cudaStream_t cpyStream;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    HANDLE_ERROR(cudaStatus);

    // Initialize the grid and block dimensions here
    dim3 dimGrid((n - 1) / BLOCK_WIDTH + 1, (n - 1) / BLOCK_WIDTH + 1, 1);
    dim3 dimBlock(BLOCK_WIDTH, BLOCK_WIDTH, 1);

    // Create new stream to copy data
    cudaStatus = cudaStreamCreate(&cpyStream);
    HANDLE_ERROR(cudaStatus);

    // Allocate GPU buffers for matrix of shortest paths d)
    cudaStatus =  cudaMalloc((void**)&dev_d, n * n * sizeof(int));
    HANDLE_ERROR(cudaStatus);
 
    // Wake up gpu
    wake_gpu_kernel<<<1, dimBlock>>>(32);

    // Copy input from host memory to GPU buffers.
    int *host_memoryd = (int*)&(kI(matrix)[0]);
    cudaStatus = cudaMemcpyAsync(dev_d, host_memoryd, n * n * sizeof(int), CMCPYHTD, cpyStream);

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    cudaStatus = cudaDeviceSynchronize();
    HANDLE_ERROR(cudaStatus);

    cudaFuncSetCacheConfig(fw_kernel, cudaFuncCachePreferL1 );
    FOR(u, 0, n - 1)
    {
        fw_kernel<<<dimGrid, dimBlock>>>(u, n, dev_d);
    }

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    HANDLE_ERROR(cudaStatus);

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    HANDLE_ERROR(cudaStatus);

    cudaStatus = cudaMemcpy(host_memoryd, dev_d, n * n * sizeof(int), CMCPYDTH);
    HANDLE_ERROR(cudaStatus);

    cudaStatus = cudaFree(dev_d);
    HANDLE_ERROR(cudaStatus);

    // Delete allocated memory 
    free(d);
    R r1(matrix);
}
