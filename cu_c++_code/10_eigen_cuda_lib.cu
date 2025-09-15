#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <cmath>

#define CUDA_CHECK(x) if((x)!=cudaSuccess){ \
    std::cerr<<"CUDA error at "<<__LINE__<<": "<<cudaGetErrorString(x)<<std::endl; return -1;}
#define CUSOLVER_CHECK(x) if((x)!=CUSOLVER_STATUS_SUCCESS){ \
    std::cerr<<"cuSOLVER error at "<<__LINE__<<std::endl; return -1;}

int main() {
    // Symmetric matrix 3x3
    std::vector<float> h_A = {4, 1, 1,
                              1, 3, 0,
                              1, 0, 2}; // row-major
    int n = 3;

    // Device memory
    float *d_A;
    CUDA_CHECK(cudaMalloc(&d_A, n*n*sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), n*n*sizeof(float), cudaMemcpyHostToDevice));

    // cuSOLVER handle
    cusolverDnHandle_t solverH;
    CUSOLVER_CHECK(cusolverDnCreate(&solverH));

    // Workspace
    int lwork = 0;
    CUSOLVER_CHECK(cusolverDnSsyevd_bufferSize(solverH, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, n, d_A, n, nullptr, &lwork));

    float *d_W, *d_work;
    int *devInfo;
    CUDA_CHECK(cudaMalloc(&d_W, n*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_work, lwork*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&devInfo, sizeof(int)));

    // Compute eigenvalues and eigenvectors
    CUSOLVER_CHECK(cusolverDnSsyevd(solverH, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, n, d_A, n, d_W, d_work, lwork, devInfo));

    int h_info;
    CUDA_CHECK(cudaMemcpy(&h_info, devInfo, sizeof(int), cudaMemcpyDeviceToHost));
    if (h_info != 0) {
        std::cerr << "Eigen decomposition failed, info = " << h_info << std::endl;
        return -1;
    }

    // Copy results to host
    std::vector<float> h_W(n);
    std::vector<float> h_V(n*n);
    CUDA_CHECK(cudaMemcpy(h_W.data(), d_W, n*sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_V.data(), d_A, n*n*sizeof(float), cudaMemcpyDeviceToHost));

    std::cout << "Eigenvalues:\n";
    for(auto w:h_W) std::cout << w << " ";
    std::cout << "\nEigenvectors (columns):\n";
    for(int i=0;i<n;i++){
        for(int j=0;j<n;j++) std::cout << h_V[j + i*n] << " ";
        std::cout << "\n";
    }

    // Cleanup
    cudaFree(d_A); cudaFree(d_W); cudaFree(d_work); cudaFree(devInfo);
    cusolverDnDestroy(solverH);

    return 0;
}
