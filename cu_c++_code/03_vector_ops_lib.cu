#include <iostream>
#include <vector>
#include <cmath>
#include <cublas_v2.h>
#include <cuda_runtime.h>

// Helper: check CUDA errors
#define CUDA_CHECK(x) if((x) != cudaSuccess){ \
    std::cerr << "CUDA error at " << __LINE__ << std::endl; return -1; }

#define CUBLAS_CHECK(x) if((x) != CUBLAS_STATUS_SUCCESS){ \
    std::cerr << "cuBLAS error at " << __LINE__ << std::endl; return -1; }

int main() {
    // Example vectors (3D for cross product)
    int n = 3;
    std::vector<float> h_v1 = {1.0f, 2.0f, 3.0f};
    std::vector<float> h_v2 = {4.0f, 5.0f, 6.0f};

    float *d_v1, *d_v2;
    CUDA_CHECK(cudaMalloc(&d_v1, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_v2, n * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_v1, h_v1.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_v2, h_v2.data(), n * sizeof(float), cudaMemcpyHostToDevice));

    // cuBLAS handle
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    // --- 1. Scalar Multiplication (cublasSscal) ---
    float alpha = 2.0f;
    CUBLAS_CHECK(cublasSscal(handle, n, &alpha, d_v1, 1));
    CUDA_CHECK(cudaMemcpy(h_v1.data(), d_v1, n * sizeof(float), cudaMemcpyDeviceToHost));
    std::cout << "2 * v1 = [ ";
    for (auto x : h_v1) std::cout << x << " ";
    std::cout << "]" << std::endl;

    // Reset v1 for other ops
    h_v1 = {1.0f, 2.0f, 3.0f};
    CUDA_CHECK(cudaMemcpy(d_v1, h_v1.data(), n * sizeof(float), cudaMemcpyHostToDevice));

    // --- 2. Vector Addition/Subtraction (cublasSaxpy) ---
    // v3 = v1 + v2
    std::vector<float> h_result(n);
    float *d_result;
    CUDA_CHECK(cudaMalloc(&d_result, n * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_result, d_v2, n * sizeof(float), cudaMemcpyDeviceToDevice)); // copy v2 → result
    alpha = 1.0f;
    CUBLAS_CHECK(cublasSaxpy(handle, n, &alpha, d_v1, 1, d_result, 1));
    CUDA_CHECK(cudaMemcpy(h_result.data(), d_result, n * sizeof(float), cudaMemcpyDeviceToHost));
    std::cout << "v1 + v2 = [ ";
    for (auto x : h_result) std::cout << x << " ";
    std::cout << "]" << std::endl;

    // v3 = v1 - v2
    CUDA_CHECK(cudaMemcpy(d_result, d_v1, n * sizeof(float), cudaMemcpyDeviceToDevice)); // copy v1 → result
    alpha = -1.0f;
    CUBLAS_CHECK(cublasSaxpy(handle, n, &alpha, d_v2, 1, d_result, 1));
    CUDA_CHECK(cudaMemcpy(h_result.data(), d_result, n * sizeof(float), cudaMemcpyDeviceToHost));
    std::cout << "v1 - v2 = [ ";
    for (auto x : h_result) std::cout << x << " ";
    std::cout << "]" << std::endl;

    // --- 3. Dot Product (cublasSdot) ---
    float dot_result;
    CUBLAS_CHECK(cublasSdot(handle, n, d_v1, 1, d_v2, 1, &dot_result));
    std::cout << "Dot(v1, v2) = " << dot_result << std::endl;

    // --- 4. Cross Product (manual, not in cuBLAS) ---
    // cuBLAS does not have cross product; implement with simple kernel
    float h_cross[3] = {
        h_v1[1]*h_v2[2] - h_v1[2]*h_v2[1],
        h_v1[2]*h_v2[0] - h_v1[0]*h_v2[2],
        h_v1[0]*h_v2[1] - h_v1[1]*h_v2[0]
    };
    std::cout << "v1 x v2 = [ " << h_cross[0] << " " << h_cross[1] << " " << h_cross[2] << " ]" << std::endl;

    // --- 5. Norms ---
    float norm1, norm2;
    // L1 norm (cublasSasum)
    CUBLAS_CHECK(cublasSasum(handle, n, d_v1, 1, &norm1));
    std::cout << "L1 norm of v1 = " << norm1 << std::endl;

    // L2 norm (cublasSnrm2)
    CUBLAS_CHECK(cublasSnrm2(handle, n, d_v1, 1, &norm2));
    std::cout << "L2 norm of v1 = " << norm2 << std::endl;

    // Cleanup
    cublasDestroy(handle);
    cudaFree(d_v1);
    cudaFree(d_v2);
    cudaFree(d_result);

    return 0;
}
