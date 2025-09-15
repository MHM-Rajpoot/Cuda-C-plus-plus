#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cmath>

#define CUDA_CHECK(x) if((x)!=cudaSuccess){ \
    std::cerr<<"CUDA error at "<<__LINE__<<": "<<cudaGetErrorString(x)<<std::endl; exit(-1);}
#define CUBLAS_CHECK(x) if((x)!=CUBLAS_STATUS_SUCCESS){ \
    std::cerr<<"cuBLAS error at "<<__LINE__<<std::endl; exit(-1);}

// CUDA kernel to compute absolute values
__global__ void abs_kernel(const float* x, float* y, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n) y[idx] = fabsf(x[idx]);
}

// CUDA kernel for reduction sum (simple version)
__global__ void reduce_sum(float* data, float* result, int n) {
    __shared__ float sdata[256];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    sdata[tid] = (idx < n) ? data[idx] : 0.0f;
    __syncthreads();

    // Reduce within block
    for(int s=blockDim.x/2; s>0; s>>=1) {
        if(tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    // Write result of this block
    if(tid == 0) atomicAdd(result, sdata[0]);
}

int main() {
    int n = 3;
    std::vector<float> h_x = {1.0f, -2.0f, 3.0f};
    std::vector<float> h_y = {4.0f, 0.5f, -1.0f};

    float *d_x, *d_y;
    CUDA_CHECK(cudaMalloc(&d_x, n*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y, n*sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), n*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y, h_y.data(), n*sizeof(float), cudaMemcpyHostToDevice));

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    // --- L2 Norm (Euclidean) ---
    float l2_x, l2_y;
    CUBLAS_CHECK(cublasSnrm2(handle, n, d_x, 1, &l2_x));
    CUBLAS_CHECK(cublasSnrm2(handle, n, d_y, 1, &l2_y));
    std::cout << "L2 Norm of x = " << l2_x << "\n";
    std::cout << "L2 Norm of y = " << l2_y << "\n";

    // --- Cosine Similarity ---
    float dot;
    CUBLAS_CHECK(cublasSdot(handle, n, d_x, 1, d_y, 1, &dot));
    float cosine_sim = dot / (l2_x * l2_y);
    std::cout << "Cosine Similarity between x and y = " << cosine_sim << "\n";

    // --- L1 Norm (Manhattan) on GPU ---
    float *d_abs_x, *d_abs_y, *d_sum_x, *d_sum_y;
    float h_sum_x=0.0f, h_sum_y=0.0f;
    CUDA_CHECK(cudaMalloc(&d_abs_x, n*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_abs_y, n*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sum_x, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sum_y, sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_sum_x, &h_sum_x, sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sum_y, &h_sum_y, sizeof(float), cudaMemcpyHostToDevice));

    abs_kernel<<<1, n>>>(d_x, d_abs_x, n);
    abs_kernel<<<1, n>>>(d_y, d_abs_y, n);

    reduce_sum<<<1, 256>>>(d_abs_x, d_sum_x, n);
    reduce_sum<<<1, 256>>>(d_abs_y, d_sum_y, n);

    CUDA_CHECK(cudaMemcpy(&h_sum_x, d_sum_x, sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_sum_y, d_sum_y, sizeof(float), cudaMemcpyDeviceToHost));

    std::cout << "L1 Norm of x = " << h_sum_x << "\n";
    std::cout << "L1 Norm of y = " << h_sum_y << "\n";

    // Cleanup
    cudaFree(d_x); cudaFree(d_y);
    cudaFree(d_abs_x); cudaFree(d_abs_y);
    cudaFree(d_sum_x); cudaFree(d_sum_y);
    cublasDestroy(handle);

    return 0;
}
