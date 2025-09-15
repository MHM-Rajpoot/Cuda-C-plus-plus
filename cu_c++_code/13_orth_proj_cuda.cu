#include <iostream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CUDA_CHECK(x) if((x)!=cudaSuccess){ \
    std::cerr<<"CUDA error at "<<__LINE__<<": "<<cudaGetErrorString(x)<<std::endl; return -1;}
#define CUBLAS_CHECK(x) if((x)!=CUBLAS_STATUS_SUCCESS){ \
    std::cerr<<"cuBLAS error at "<<__LINE__<<std::endl; return -1;}

// Gram-Schmidt orthonormalization on host
void gram_schmidt(std::vector<float>& v1, std::vector<float>& v2) {
    // Normalize v1
    float norm1 = std::sqrt(v1[0]*v1[0]+v1[1]*v1[1]+v1[2]*v1[2]);
    for(int i=0;i<3;i++) v1[i] /= norm1;

    // v2 = v2 - proj_v1(v2)
    float dot = v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2];
    for(int i=0;i<3;i++) v2[i] -= dot*v1[i];

    // Normalize v2
    float norm2 = std::sqrt(v2[0]*v2[0]+v2[1]*v2[1]+v2[2]*v2[2]);
    for(int i=0;i<3;i++) v2[i] /= norm2;
}

int main() {
    // 3D vectors spanning subspace
    std::vector<float> v1 = {1,1,0};
    std::vector<float> v2 = {1,0,1};
    gram_schmidt(v1,v2);

    // Vector to project
    std::vector<float> b = {3,2,1};

    // Device memory
    float *d_Q, *d_b, *d_proj;
    CUDA_CHECK(cudaMalloc(&d_Q, 3*2*sizeof(float))); // 2 orthonormal vectors
    CUDA_CHECK(cudaMalloc(&d_b, 3*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_proj, 3*sizeof(float)));

    // Copy Q and b
    std::vector<float> h_Q = {v1[0], v1[1], v1[2],
                              v2[0], v2[1], v2[2]}; // column-major
    CUDA_CHECK(cudaMemcpy(d_Q, h_Q.data(), 3*2*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, b.data(), 3*sizeof(float), cudaMemcpyHostToDevice));

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    float alpha=1.0f, beta=0.0f;
    // proj = Q * (Q^T * b)
    float *d_temp;
    CUDA_CHECK(cudaMalloc(&d_temp, 2*sizeof(float))); // 2x1 vector
    // d_temp = Q^T * b
    CUBLAS_CHECK(cublasSgemv(handle, CUBLAS_OP_T, 3, 2, &alpha, d_Q, 3, d_b, 1, &beta, d_temp, 1));
    // d_proj = Q * d_temp
    CUBLAS_CHECK(cublasSgemv(handle, CUBLAS_OP_N, 3, 2, &alpha, d_Q, 3, d_temp, 1, &beta, d_proj, 1));

    // Copy projection back
    std::vector<float> h_proj(3);
    CUDA_CHECK(cudaMemcpy(h_proj.data(), d_proj, 3*sizeof(float), cudaMemcpyDeviceToHost));

    std::cout << "Projection of b onto subspace spanned by v1 and v2:\n";
    std::cout << "(" << h_proj[0] << ", " << h_proj[1] << ", " << h_proj[2] << ")\n";

    cudaFree(d_Q); cudaFree(d_b); cudaFree(d_proj); cudaFree(d_temp);
    cublasDestroy(handle);

    return 0;
}
