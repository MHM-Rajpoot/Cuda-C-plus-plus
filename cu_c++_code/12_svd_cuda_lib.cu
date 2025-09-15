#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cusolverDn.h>

#define CUDA_CHECK(x) if((x)!=cudaSuccess){ \
    std::cerr<<"CUDA error at "<<__LINE__<<": "<<cudaGetErrorString(x)<<std::endl; return -1;}
#define CUSOLVER_CHECK(x) if((x)!=CUSOLVER_STATUS_SUCCESS){ \
    std::cerr<<"cuSOLVER error at "<<__LINE__<<std::endl; return -1;}

int main() {
    int m=3, n=2;
    std::vector<float> h_A = {1, 0,
                              0, 1,
                              1, 1}; // row-major m x n

    float *d_A;
    CUDA_CHECK(cudaMalloc(&d_A, m*n*sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), m*n*sizeof(float), cudaMemcpyHostToDevice));

    cusolverDnHandle_t solverH;
    CUSOLVER_CHECK(cusolverDnCreate(&solverH));

    int lwork=0;
    CUSOLVER_CHECK(cusolverDnSgesvd_bufferSize(solverH, m, n, &lwork));

    float *d_S, *d_U, *d_VT, *d_work; int *devInfo;
    CUDA_CHECK(cudaMalloc(&d_S, n*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_U, m*m*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_VT, n*n*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_work, lwork*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&devInfo, sizeof(int)));

    signed char jobu = 'A', jobvt = 'A';
    CUSOLVER_CHECK(cusolverDnSgesvd(solverH, jobu, jobvt, m, n, d_A, m, d_S, d_U, m, d_VT, n, d_work, lwork, nullptr, devInfo));

    std::vector<float> h_S(n);
    CUDA_CHECK(cudaMemcpy(h_S.data(), d_S, n*sizeof(float), cudaMemcpyDeviceToHost));

    std::cout << "Singular values:\n";
    for(auto s:h_S) std::cout << s << " ";
    std::cout << std::endl;

    cudaFree(d_A); cudaFree(d_S); cudaFree(d_U); cudaFree(d_VT); cudaFree(d_work); cudaFree(devInfo);
    cusolverDnDestroy(solverH);

    return 0;
}
