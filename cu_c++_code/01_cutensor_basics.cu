#include <cutensor.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

int main() {
    // Initialize cuTENSOR
    cutensorHandle_t handle;
    cutensorStatus_t status = cutensorCreate(&handle);

    if (status == CUTENSOR_STATUS_SUCCESS) {
        std::cout << "✅ cuTENSOR initialized successfully!" << std::endl;
    } else {
        std::cout << "❌ cuTENSOR init failed: " << status << std::endl;
        return -1;
    }

    // --- 1. Scalar (just a single number) ---
    float scalar = 3.14f;
    std::cout << "Scalar example: " << scalar << std::endl;

    // --- 2. Vector (1D array) ---
    int n = 4;
    std::vector<float> h_vector = {1.0, 2.0, 3.0, 4.0};
    float* d_vector;
    cudaMalloc(&d_vector, n * sizeof(float));
    cudaMemcpy(d_vector, h_vector.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    std::cout << "Vector example: [1, 2, 3, 4]" << std::endl;

    // --- 3. Matrix (2D array) ---
    int rows = 2, cols = 3;
    std::vector<float> h_matrix = {
        1, 2, 3,
        4, 5, 6
    };
    float* d_matrix;
    cudaMalloc(&d_matrix, rows * cols * sizeof(float));
    cudaMemcpy(d_matrix, h_matrix.data(), rows * cols * sizeof(float), cudaMemcpyHostToDevice);
    std::cout << "Matrix example (2x3): [[1,2,3],[4,5,6]]" << std::endl;

    // --- 4. Tensor (3D array, e.g., RGB image 2x2x3) ---
    int dimX = 2, dimY = 2, dimC = 3;
    std::vector<float> h_tensor = {
        // pixel (0,0)
        255, 0, 0,   // Red
        // pixel (0,1)
        0, 255, 0,   // Green
        // pixel (1,0)
        0, 0, 255,   // Blue
        // pixel (1,1)
        255, 255, 0  // Yellow
    };
    float* d_tensor;
    cudaMalloc(&d_tensor, dimX * dimY * dimC * sizeof(float));
    cudaMemcpy(d_tensor, h_tensor.data(), dimX * dimY * dimC * sizeof(float), cudaMemcpyHostToDevice);
    std::cout << "Tensor example (2x2x3 RGB image)" << std::endl;

    // Cleanup
    cudaFree(d_vector);
    cudaFree(d_matrix);
    cudaFree(d_tensor);
    cutensorDestroy(handle);

    return 0;
}
