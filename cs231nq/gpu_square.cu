// Include the cuda header and the k.h interface.
#include </usr/local/cuda/include/cuda.h>
#include"stdio.h"
#include"k.h"

// Export the function we will load into kdb+
extern  "C" K gpu_square(K x);

// Define the "Kernel" that executes on the CUDA device in parallel
__global__ void square_array(double *a, int N) {
 int idx = blockIdx.x * blockDim.x + threadIdx.x;
 if (idx<N)
    a[idx] = a[idx] * a[idx];
}

// A function to use from kdb+ to square a vector of reals by
// - allocating space on the graphics card
// - copying the data over from the K object
// - doing the work
// - copy back and overwrite the K object data
K gpu_square(K x) {
  // Pointers to host & device arrays
 double *host_memory = (float*) &(kF(x)[0]), *device_memory;

 // Allocate memory on the device for the data and copy it to the GPU
 size_t size = xn * sizeof(double);
 cudaMalloc((void **)&device_memory, size);
 cudaMemcpy(device_memory, host_memory, size, cudaMemcpyHostToDevice);

 // Do the computaton on the card
 int block_size = 4;
 int n_blocks = xn/block_size + (xn%block_size == 0 ? 0:1);
 square_array <<< n_blocks, block_size >>> (device_memory, xn);

 // Copy back the data, overwriting the input, 
 // free the memory we allocated on the graphics card
 cudaMemcpy(host_memory, device_memory, size, cudaMemcpyDeviceToHost);
 cudaFree(device_memory);
 R r1(x);
}

