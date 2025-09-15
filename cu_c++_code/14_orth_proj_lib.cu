#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <cublas_v2.h>

#define CUDA_CHECK(x) if((x)!=cudaSuccess){ \
    std::cerr<<"CUDA error at "<<__LINE__<<": "<<cudaGetErrorString(x)<<std::endl; return -1;}
#define CUSOLVER_CHECK(x) if((x)!=CUSOLVER_STATUS_SUCCESS){ \
    std::cerr<<"cuSOLVER error at "<<__LINE__<<std::endl; return -1;}
#define CUBLAS_CHECK(x) if((x)!=CUBLAS_STATUS_SUCCESS){ \
    std::cerr<<"cuBLAS error at "<<__LINE__<<std::endl; return -1;}

int main() {
    int m = 3; // dimension
    int n = 2; // number of vectors spanning subspace

    // Columns are vectors spanning subspace
    std::vector<float> h_A = {1,1,0,  // v1
                              1,0,1}; // v2
    // Row-major to column-major
    std::vector<float> h_A_col(m*n);
    for(int j=0;j<n;j++)
        for(int i=0;i<m;i++)
            h_A_col[i + j*m] = h_A[j*m + i];

    // Vector to project
    std::vector<float> h_b = {3,2,1};

    float *d_A, *d_tau, *d_b, *d_proj;
    CUDA_CHECK(cudaMalloc(&d_A, m*n*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_tau, n*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b, m*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_proj, m*sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_A, h_A_col.data(), m*n*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), m*sizeof(float), cudaMemcpyHostToDevice));

    cusolverDnHandle_t solverH;
    CUSOLVER_CHECK(cusolverDnCreate(&solverH));

    int lwork = 0;
    CUSOLVER_CHECK(cusolverDnSgeqrf_bufferSize(solverH, m, n, d_A, m, &lwork));

    float *d_work;
    int *devInfo;
    CUDA_CHECK(cudaMalloc(&d_work, lwork*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&devInfo, sizeof(int)));

    // Compute QR decomposition: d_A -> Q (orthonormal vectors in first n columns)
    CUSOLVER_CHECK(cusolverDnSgeqrf(solverH, m, n, d_A, m, d_tau, d_work, lwork, devInfo));

    // Generate explicit Q
    CUSOLVER_CHECK(cusolverDnSorgqr(solverH, m, n, n, d_A, m, d_tau, d_work, lwork, devInfo));

    // Now d_A contains Q (m x n), orthonormal columns
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    float alpha=1.0f, beta=0.0f;
    float *d_temp;
    CUDA_CHECK(cudaMalloc(&d_temp, n*sizeof(float)));

    // temp = Q^T * b
    CUBLAS_CHECK(cublasSgemv(handle, CUBLAS_OP_T, m, n, &alpha, d_A, m, d_b, 1, &beta, d_temp, 1));
    // proj = Q * temp
    CUBLAS_CHECK(cublasSgemv(handle, CUBLAS_OP_N, m, n, &alpha, d_A, m, d_temp, 1, &beta, d_proj, 1));

    // Copy projection back
    std::vector<float> h_proj(m);
    CUDA_CHECK(cudaMemcpy(h_proj.data(), d_proj, m*sizeof(float), cudaMemcpyDeviceToHost));

    std::cout << "Projection of b onto subspace spanned by Q:\n";
    for(int i=0;i<m;i++) std::cout << h_proj[i] << " ";
    std::cout << std::endl;

    // Cleanup
    cudaFree(d_A); cudaFree(d_tau); cudaFree(d_b); cudaFree(d_proj); cudaFree(d_work); cudaFree(d_temp); cudaFree(devInfo);
    cusolverDnDestroy(solverH);
    cublasDestroy(handle);

    return 0;
}
