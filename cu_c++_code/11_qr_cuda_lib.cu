#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cusolverDn.h>

#define CUDA_CHECK(x) if((x)!=cudaSuccess){ \
    std::cerr<<"CUDA error at "<<__LINE__<<": "<<cudaGetErrorString(x)<<std::endl; return -1;}
#define CUSOLVER_CHECK(x) if((x)!=CUSOLVER_STATUS_SUCCESS){ \
    std::cerr<<"cuSOLVER error at "<<__LINE__<<std::endl; return -1;}

int main() {
    int m=3, n=2; // matrix size
    std::vector<float> h_A = {12, -51,
                               6, 167,
                              -4, 24}; // row-major

    float *d_A; CUDA_CHECK(cudaMalloc(&d_A, m*n*sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), m*n*sizeof(float), cudaMemcpyHostToDevice));

    cusolverDnHandle_t solverH;
    CUSOLVER_CHECK(cusolverDnCreate(&solverH));

    int lwork=0;
    CUSOLVER_CHECK(cusolverDnSgeqrf_bufferSize(solverH, m, n, d_A, m, &lwork));

    float *d_tau, *d_work; int *devInfo;
    CUDA_CHECK(cudaMalloc(&d_tau, n*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_work, lwork*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&devInfo, sizeof(int)));

    // Compute QR factorization
    CUSOLVER_CHECK(cusolverDnSgeqrf(solverH, m, n, d_A, m, d_tau, d_work, lwork, devInfo));

    // Copy back results
    std::vector<float> h_R(m*n);
    std::vector<float> h_tau(n);
    CUDA_CHECK(cudaMemcpy(h_R.data(), d_A, m*n*sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_tau.data(), d_tau, n*sizeof(float), cudaMemcpyDeviceToHost));

    std::cout << "R (upper-triangular part of A):\n";
    for(int i=0;i<m;i++){
        for(int j=0;j<n;j++) std::cout<<h_R[i + j*m]<<" ";
        std::cout<<std::endl;
    }

    cudaFree(d_A); cudaFree(d_tau); cudaFree(d_work); cudaFree(devInfo);
    cusolverDnDestroy(solverH);
    return 0;
}
