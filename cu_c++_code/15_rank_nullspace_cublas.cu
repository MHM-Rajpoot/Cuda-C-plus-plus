#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CUDA_CHECK(x) if((x)!=cudaSuccess){ \
    std::cerr<<"CUDA error at "<<__LINE__<<": "<<cudaGetErrorString(x)<<std::endl; return -1;}
#define CUBLAS_CHECK(x) if((x)!=CUBLAS_STATUS_SUCCESS){ \
    std::cerr<<"cuBLAS error at "<<__LINE__<<std::endl; return -1;}

int main() {
    // Matrix dimensions
    int m = 2, n = 3;

    // Column-major matrix A
    std::vector<float> h_A = {
        1,4,   // col 0
        2,5,   // col 1
        3,6    // col 2
    };

    // Device memory
    float *d_A, *d_AtA;
    CUDA_CHECK(cudaMalloc(&d_A, m*n*sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), m*n*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_AtA, n*n*sizeof(float)));

    // cuBLAS handle
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    float alpha = 1.0f, beta = 0.0f;

    // Compute AtA = A^T * A
    // A: m x n, AtA: n x n
    CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                             n, n, m, &alpha,
                             d_A, m,
                             d_A, m,
                             &beta, d_AtA, n));

    // Copy AtA to host
    std::vector<float> h_AtA(n*n);
    CUDA_CHECK(cudaMemcpy(h_AtA.data(), d_AtA, n*n*sizeof(float), cudaMemcpyDeviceToHost));

    std::cout << "A^T * A = \n";
    for(int i=0;i<n;i++){
        for(int j=0;j<n;j++) std::cout << h_AtA[i + j*n] << " "; // column-major
        std::cout << "\n";
    }

    // Approximate rank by counting non-zero diagonals
    float eps = 1e-6;
    int rank = 0;
    for(int i=0;i<n;i++) if(fabs(h_AtA[i + i*n]) > eps) rank++;
    std::cout << "Approximate Rank of A = " << rank << std::endl;

    // Cleanup
    cudaFree(d_A); cudaFree(d_AtA);
    cublasDestroy(handle);

    return 0;
}
