#include </usr/local/cuda/include/cuda.h>
#include"stdio.h"
#include"k.h"
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cublas_v2.h>
#include <curand.h>
#include <time.h>
#include <sys/time.h>
#define uS_PER_SEC 1000000
#define uS_PER_mS 1000
#define N  1000
#define M 1000

// compile with:
// >> nvcc --compiler-options '-fPIC -DKXVER=3 -O2' -o $QHOME/l64/gpu_mm_unrazeflip_timer.so --shared -lcurand -lcublas gpu_mm_unrazeflip_timer.cu
// load into q with:
// q).gpu.mm:`gpu_mm_unrazeflip 2:(`gpu_mm;7)
// q).gpu.unrazeflip:`gpu_mm_unrazeflip 2:(`unrazeflip;3)
// q).gpu.mmu:{[x;y] .gpu.unrazeflip[;rows_x;cols_y] .gpu.mm[raze x;rows_x;count x 0;raze y;count y;cols_y;((rows_x:count x)*cols_y:count y 0)#0f]}
// .gpu.mm[a;b]~mmu[a;b]

// Export the function we will load into kdb+
extern  "C" K gpu_mm(K A, K rA, K cA, K B, K rB, K cB, K C);
extern  "C" K unrazeflip(K x, K rows_x, K cols_x);

// Multiply the arrays A and B on GPU and save the result in C
// C(m,n) = A(m,k) * B(k,n)
// m= nr_rows_A
// k= nr_cols_A
// n= nr_cols_B
void gpu_blas_mmul(const double *A, const double *B, double *C, const int m, const int k, const int n) {
    int lda=k,ldb=n,ldc=m;
    timeval t1, t2;
    const double alf = 1;
    const double bet = 0;
    const double *alpha = &alf;
    const double *beta = &bet;

    // Create a handle for CUBLAS
    cublasHandle_t handle;
    cublasCreate(&handle);
    gettimeofday(&t1, NULL);
    // Do the actual multiplication
    // CUBLAS_OP_T means input is row major, CUBLAS_OP_N means input is column major
    cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    gettimeofday(&t2, NULL);
    float et2 = (((t2.tv_sec*uS_PER_SEC)+t2.tv_usec) - ((t1.tv_sec*uS_PER_SEC)+t1.tv_usec))/(float)uS_PER_mS;
    printf("time to perform cublas matrix multiply: %fms\n", et2);

    // Destroy the handle
    cublasDestroy(handle);
}

K gpu_mm(K A, K rA, K cA, K B, K rB, K cB,  K C) {
    // Allocate 3 arrays on CPU
    int nr_rows_A = rA->n;
    int nr_cols_A = cA->n;
    int nr_rows_B = rB->n;
    int nr_cols_B = cB->n;
    int nr_rows_C = nr_rows_A;
    int nr_cols_C = nr_cols_B;
    timeval t1, t2, t3, t4, t5, t6;

    // allocate memory, host arrays
        gettimeofday(&t1, NULL);
    double *h_A = (double *)malloc(nr_rows_A * nr_cols_A * sizeof(double));
    double *h_B = (double *)malloc(nr_rows_B * nr_cols_B * sizeof(double));
    double *h_C = (double *)malloc(nr_rows_C * nr_cols_C * sizeof(double));

    // Allocate 3 arrays on GPU, device arrays
    double *d_A, *d_B, *d_C;
    double *host_memoryA = (double*) &(kF(A)[0]);
    double *host_memoryB = (double*) &(kF(B)[0]);
    double *host_memoryC = (double*) &(kF(C)[0]);
    size_t sizeA = nr_rows_A * nr_cols_A * sizeof(double);
    size_t sizeB = nr_rows_B * nr_cols_B * sizeof(double);
    cudaMalloc((void **)&d_A, sizeA);
    cudaMalloc((void **)&d_B, sizeB);
    cudaMalloc(&d_C,nr_rows_C * nr_cols_C * sizeof(double));
        gettimeofday(&t2, NULL);
        float et2 = (((t2.tv_sec*uS_PER_SEC)+t2.tv_usec) - ((t1.tv_sec*uS_PER_SEC)+t1.tv_usec))/(float)uS_PER_mS;
        printf("time to allocate host and device array mems: %fms\n", et2);


    // copy A and B to GPU:
        gettimeofday(&t3, NULL);
    cudaMemcpy(d_A, host_memoryA, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, host_memoryB, sizeB, cudaMemcpyHostToDevice);
        gettimeofday(&t4, NULL);
        float et4 = (((t4.tv_sec*uS_PER_SEC)+t4.tv_usec) - ((t3.tv_sec*uS_PER_SEC)+t3.tv_usec))/(float)uS_PER_mS;
        printf("time to copy inputs to GPU: %fms\n", et4);

    // Multiply A and B on GPU
    gpu_blas_mmul(d_A, d_B, d_C, nr_rows_A, nr_cols_A, nr_cols_B);

    // Copy the result back to host memory
        gettimeofday(&t5, NULL);
    cudaMemcpy(host_memoryC,d_C,nr_rows_C * nr_cols_C * sizeof(double),cudaMemcpyDeviceToHost);
        gettimeofday(&t6, NULL);
        float et6 = (((t6.tv_sec*uS_PER_SEC)+t6.tv_usec) - ((t5.tv_sec*uS_PER_SEC)+t5.tv_usec))/(float)uS_PER_mS;
        printf("time to copy result from GPU back to host: %fms\n", et6);

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


// hack function, cublas returns a "razed matrix transpose", i.e. .gpu.mm[x;y]~raze flip mmu[x;y]
// so this basically undoes this
K unrazeflip(K x, K rows_x, K cols_x){
    long r=rows_x->n;
    long c=cols_x->n;
    K res, row;
    res = ktn(0,0);
    long j=0;
    long i=0;
    for(j=0;j<r;++j){
       row = ktn(KF,c);
       for(i=0;i<c;++i){
          kF(row)[i]=kF(x)[(i*r)+j];
       }
       jk(&res,row);
    }
    return res;
}
