#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cmath>

#define CUDA_CHECK(x) if((x)!=cudaSuccess){ \
    std::cerr<<"CUDA error at "<<__LINE__<<": "<<cudaGetErrorString(x)<<std::endl; exit(-1);}
#define CUBLAS_CHECK(x) if((x)!=CUBLAS_STATUS_SUCCESS){ \
    std::cerr<<"cuBLAS error at "<<__LINE__<<std::endl; exit(-1);}

// Compute Euclidean norm of a vector on GPU
float gpu_norm(cublasHandle_t handle, float* d_x, int len) {
    float result;
    CUBLAS_CHECK(cublasSnrm2(handle, len, d_x, 1, &result));
    return result;
}

// Subtract projection: x = x - (v^T x) * v
void gpu_subtract_projection(cublasHandle_t handle, float* d_x, float* d_v, int len) {
    float dot;
    CUBLAS_CHECK(cublasSdot(handle, len, d_v, 1, d_x, 1, &dot));
    float alpha = -dot;
    CUBLAS_CHECK(cublasSaxpy(handle, len, &alpha, d_v, 1, d_x, 1));
}

int main() {
    int m = 2, n = 3;
    std::vector<float> h_A = {
        1,4,  // col 0
        2,5,  // col 1
        3,6   // col 2
    };

    float *d_A;
    CUDA_CHECK(cudaMalloc(&d_A, m*n*sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), m*n*sizeof(float), cudaMemcpyHostToDevice));

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    // Null space vector
    float *d_x;
    CUDA_CHECK(cudaMalloc(&d_x, n*sizeof(float)));

    // Initial guess (non-zero vector)
    std::vector<float> h_x = {1.0f, -2.0f, 1.0f};
    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), n*sizeof(float), cudaMemcpyHostToDevice));

    // Orthogonalize w.r.t columns of A (Gram-Schmidt)
    for(int j=0; j<m; j++) {
        float *d_col = d_A + j; // column-major stride
        gpu_subtract_projection(handle, d_x, d_col, n);
    }

    // Normalize
    float norm = gpu_norm(handle, d_x, n);
    float alpha = 1.0f/norm;
    CUBLAS_CHECK(cublasSscal(handle, n, &alpha, d_x, 1));

    // Copy back to host
    CUDA_CHECK(cudaMemcpy(h_x.data(), d_x, n*sizeof(float), cudaMemcpyDeviceToHost));

    std::cout << "Null space basis vector (Ax=0): [ ";
    for(auto v : h_x) std::cout << v << " ";
    std::cout << "]\n";

    // Cleanup
    cudaFree(d_A); cudaFree(d_x);
    cublasDestroy(handle);

    return 0;
}
