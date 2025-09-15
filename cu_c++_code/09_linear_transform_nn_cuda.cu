#include <iostream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CUDA_CHECK(x) if((x)!=cudaSuccess){ \
    std::cerr<<"CUDA error at "<<__LINE__<<": "<<cudaGetErrorString(x)<<std::endl; return -1;}
#define CUBLAS_CHECK(x) if((x)!=CUBLAS_STATUS_SUCCESS){ \
    std::cerr<<"cuBLAS error at "<<__LINE__<<std::endl; return -1;}

// ReLU activation on host
void relu(std::vector<float> &v){
    for(auto &x:v) if(x<0) x=0;
}

int main() {
    // 3D points: 3 x 3 (each column = a point)
    std::vector<float> h_points = {1, 0, 1,
                                   0, 1, 1,
                                   0, 0, 1}; // 3 points
    int num_points = 3;

    // 3x3 rotation around Z by 90 deg + scaling
    float theta = 90.0f;
    float rad = theta * M_PI / 180.0f;
    float scale = 2.0f;
    std::vector<float> h_R = {
        scale * cos(rad), -scale * sin(rad), 0,
        scale * sin(rad),  scale * cos(rad), 0,
        0,                0,                scale
    };

    // Device memory
    float *d_points, *d_R, *d_result;
    CUDA_CHECK(cudaMalloc(&d_points, 3*num_points*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_R, 3*3*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_result, 3*num_points*sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_points, h_points.data(), 3*num_points*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_R, h_R.data(), 3*3*sizeof(float), cudaMemcpyHostToDevice));

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    float alpha = 1.0f, beta = 0.0f;
    // Linear transformation: result = R * points
    CUBLAS_CHECK(cublasSgemm(handle,
                             CUBLAS_OP_N, CUBLAS_OP_N,
                             3, num_points, 3,
                             &alpha,
                             d_R, 3,
                             d_points, 3,
                             &beta,
                             d_result, 3));

    // Copy result back
    std::vector<float> h_result(3*num_points);
    CUDA_CHECK(cudaMemcpy(h_result.data(), d_result, 3*num_points*sizeof(float), cudaMemcpyDeviceToHost));

    std::cout << "Original 3D points:\n";
    for(int i=0;i<num_points;i++) std::cout << "(" << h_points[3*i] << "," << h_points[3*i+1] << "," << h_points[3*i+2] << ")\n";

    std::cout << "Transformed 3D points (rotation + scaling):\n";
    for(int i=0;i<num_points;i++) std::cout << "(" << h_result[3*i] << "," << h_result[3*i+1] << "," << h_result[3*i+2] << ")\n";

    // ---------------- Simple NN Forward ----------------
    // Input: 3 features x 3 samples
    // Layer1: 3x3 weight + bias
    std::vector<float> W1 = {0.5, -0.2, 0.1,
                             0.3, 0.8, -0.5,
                             -0.6, 0.1, 0.4};
    std::vector<float> b1 = {0.1, -0.1, 0.2};

    float *d_W1, *d_b1;
    CUDA_CHECK(cudaMalloc(&d_W1, 3*3*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b1, 3*sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_W1, W1.data(), 3*3*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b1, b1.data(), 3*sizeof(float), cudaMemcpyHostToDevice));

    // Layer1: Y = W1*X + b
    CUBLAS_CHECK(cublasSgemm(handle,
                             CUBLAS_OP_N, CUBLAS_OP_N,
                             3, num_points, 3,
                             &alpha,
                             d_W1, 3,
                             d_result, 3,
                             &beta,
                             d_points, 3)); // reuse d_points for output

    // Copy result back for activation
    CUDA_CHECK(cudaMemcpy(h_result.data(), d_points, 3*num_points*sizeof(float), cudaMemcpyDeviceToHost));

    // Add bias and ReLU
    for(int i=0;i<num_points;i++)
        for(int j=0;j<3;j++)
            h_result[j + 3*i] += b1[j];
    relu(h_result);

    std::cout << "NN Layer1 output (after ReLU):\n";
    for(int i=0;i<num_points;i++) std::cout << "(" << h_result[3*i] << "," << h_result[3*i+1] << "," << h_result[3*i+2] << ")\n";

    // Cleanup
    cublasDestroy(handle);
    cudaFree(d_points); cudaFree(d_R); cudaFree(d_result);
    cudaFree(d_W1); cudaFree(d_b1);

    return 0;
}
