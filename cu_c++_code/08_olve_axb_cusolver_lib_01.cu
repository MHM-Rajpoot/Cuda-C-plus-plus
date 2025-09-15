#include <iostream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#include <cusolverDn.h>

#define CUDA_CHECK(x) if((x)!=cudaSuccess){ \
    std::cerr<<"CUDA error at "<<__LINE__<<": "<<cudaGetErrorString(x)<<std::endl; return -1;}
#define CUSOLVER_CHECK(x) if((x)!=CUSOLVER_STATUS_SUCCESS){ \
    std::cerr<<"cuSOLVER error at "<<__LINE__<<std::endl; return -1;}

int main() {
    int n = 3; // matrix size

    // Example matrix A (row-major)
    std::vector<float> h_A = {4, 1, 0,
                              1, 3, 1,
                              0, 1, 2};
    // Right-hand side b
    std::vector<float> h_b = {1, 2, 3};

    // Device memory
    float *d_A = nullptr, *d_b = nullptr;
    int *d_ipiv = nullptr, *d_info = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, n*n*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b, n*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ipiv, n*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_info, sizeof(int)));

    // Copy A and b to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), n*n*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), n*sizeof(float), cudaMemcpyHostToDevice));

    // cuSOLVER handle
    cusolverDnHandle_t cusolverH;
    CUSOLVER_CHECK(cusolverDnCreate(&cusolverH));

    // Workspace query
    int work_size = 0;
    CUSOLVER_CHECK(cusolverDnSgetrf_bufferSize(cusolverH, n, n, d_A, n, &work_size));

    float *d_work = nullptr;
    CUDA_CHECK(cudaMalloc(&d_work, work_size * sizeof(float)));

    // --- LU factorization ---
    CUSOLVER_CHECK(cusolverDnSgetrf(cusolverH, n, n, d_A, n, d_work, d_ipiv, d_info));

    int h_info = 0;
    CUDA_CHECK(cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
    if (h_info != 0) {
        std::cerr << "LU factorization failed, info = " << h_info << std::endl;
        return -1;
    }

    // --- Solve Ax = b ---
    CUSOLVER_CHECK(cusolverDnSgetrs(cusolverH, CUBLAS_OP_N, n, 1, d_A, n, d_ipiv, d_b, n, d_info));
    CUDA_CHECK(cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
    if (h_info != 0) {
        std::cerr << "Solve failed, info = " << h_info << std::endl;
        return -1;
    }

    // Copy solution back to host
    std::vector<float> h_x(n);
    CUDA_CHECK(cudaMemcpy(h_x.data(), d_b, n*sizeof(float), cudaMemcpyDeviceToHost));

    std::cout << "Solution x = [ ";
    for (auto v : h_x) std::cout << v << " ";
    std::cout << "]\n";

    // --- Compute residual norm ||b - Ax|| in double precision ---
    std::vector<double> h_xd(n), h_Ad(n*n), h_bd(n);
    for (int i=0;i<n;i++) h_xd[i] = static_cast<double>(h_x[i]);
    for (int i=0;i<n*n;i++) h_Ad[i] = static_cast<double>(h_A[i]);
    for (int i=0;i<n;i++) h_bd[i] = static_cast<double>(h_b[i]);

    std::vector<double> res(n,0.0);
    for (int i=0;i<n;i++) {
        double sum=0.0;
        for (int j=0;j<n;j++) sum += h_Ad[i*n + j]*h_xd[j];
        res[i] = h_bd[i] - sum;
    }
    double rnorm=0.0;
    for (int i=0;i<n;i++) rnorm += res[i]*res[i];
    rnorm = std::sqrt(rnorm);

    std::cout << "Residual norm ||b - Ax|| = " << rnorm << std::endl;

    // Cleanup
    cudaFree(d_A); cudaFree(d_b); cudaFree(d_ipiv); cudaFree(d_info); cudaFree(d_work);
    cusolverDnDestroy(cusolverH);

    return 0;
}
