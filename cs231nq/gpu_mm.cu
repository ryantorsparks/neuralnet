#include </usr/local/cuda/include/cuda.h>
#include"stdio.h"
#include"k.h"
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cublas_v2.h>
#include <curand.h>

// compile with:
//   nvcc --compiler-options '-fPIC -DKXVER=3 -O2' -o $QHOME/l64/gpu_mm.so --shared -lcurand -lcublas gpu_mm.cu

// Export the function we will load into kdb+
extern  "C" K gpu_mm(K A, K B, K C);

// Fill the array A(nr_rows_A, nr_cols_A) with random numbers on GPU
void GPU_fill_rand(float *A, int nr_rows_A, int nr_cols_A) {
    // Create a pseudo-random number generator
    curandGenerator_t prng;
    curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);

    // Set the seed for the random number generator using the system clock
    curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long) clock());

    // Fill the array with random numbers on the device
    curandGenerateUniform(prng, A, nr_rows_A * nr_cols_A);
}

// Multiply the arrays A and B on GPU and save the result in C
// C(m,n) = A(m,k) * B(k,n)
void gpu_blas_mmul(const float *A, const float *B, float *C, const int m, const int k, const int n) {
    int lda=m,ldb=k,ldc=m;
    const float alf = 1;
    const float bet = 0;
    const float *alpha = &alf;
    const float *beta = &bet;

    // Create a handle for CUBLAS
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Do the actual multiplication
    // CUBLAS_OP_T means input is row major, CUBLAS_OP_N means input is column major
//    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);

    // Destroy the handle
    cublasDestroy(handle);
}


//Print matrix A(nr_rows_A, nr_cols_A) storage in column-major format
void print_matrix(const float *A, int nr_rows_A, int nr_cols_A) {

    for(int i = 0; i < nr_rows_A; ++i){
        for(int j = 0; j < nr_cols_A; ++j){
            std::cout << A[j * nr_rows_A + i] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

K gpu_mm(K A, K B, K C) {
    // Allocate 3 arrays on CPU
    int nr_rows_A, nr_cols_A, nr_rows_B, nr_cols_B, nr_rows_C, nr_cols_C;

    // get shape of input A (assume A/B/C are all square, same shape
    long n=A->n;
    long r=sqrt(n);

    // for simplicity we are going to use square arrays
    nr_rows_A = nr_cols_A = nr_rows_B = nr_cols_B = nr_rows_C = nr_cols_C = r;

    // allocate memory, host arrays
    float *h_A = (float *)malloc(nr_rows_A * nr_cols_A * sizeof(float));
    float *h_B = (float *)malloc(nr_rows_B * nr_cols_B * sizeof(float));
    float *h_C = (float *)malloc(nr_rows_C * nr_cols_C * sizeof(float));

    // Allocate 3 arrays on GPU, device arrays
    float *d_A, *d_B, *d_C;
    float *host_memoryA = (float*) &(kE(A)[0]);
    float *host_memoryB = (float*) &(kE(B)[0]);
    float *host_memoryC = (float*) &(kE(C)[0]);
    size_t size = n * sizeof(float);
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc(&d_C,nr_rows_C * nr_cols_C * sizeof(float));

    // If you already have useful values in A and B you can copy them in GPU:
    cudaMemcpy(d_A, host_memoryA, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, host_memoryB, size, cudaMemcpyHostToDevice);

    // Multiply A and B on GPU
    gpu_blas_mmul(d_A, d_B, d_C, nr_rows_A, nr_cols_A, nr_cols_B);

    // Copy the result on host memory
    cudaMemcpy(host_memoryC,d_C,nr_rows_C * nr_cols_C * sizeof(float),cudaMemcpyDeviceToHost);

    //Free GPU memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);    

    // Free CPU memory
    free(h_A);
    free(h_B);
    free(h_C);

    R r1(C);
}
