#include <iostream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CUDA_CHECK(x) if((x)!=cudaSuccess){ \
    std::cerr<<"CUDA error at "<<__LINE__<<": "<<cudaGetErrorString(x)<<std::endl; return -1;}
#define CUBLAS_CHECK(x) if((x)!=CUBLAS_STATUS_SUCCESS){ \
    std::cerr<<"cuBLAS error at "<<__LINE__<<std::endl; return -1;}

int main() {
    // Example SPD matrix A (3x3) in COLUMN-MAJOR order:
    // A = [ 4 1 0
    //       1 3 1
    //       0 1 2 ]
    // Column-major storage (columns concatenated):
    std::vector<float> h_A = {
        4.0f, 1.0f, 0.0f,   // col 0
        1.0f, 3.0f, 1.0f,   // col 1
        0.0f, 1.0f, 2.0f    // col 2
    };
    // RHS b
    std::vector<float> h_b = {1.0f, 2.0f, 3.0f};

    int n = 3; // matrix size

    // Allocate device memory
    float *d_A = nullptr, *d_x = nullptr, *d_b = nullptr;
    float *d_r = nullptr, *d_p = nullptr, *d_Ap = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, n*n*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_x, n*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b, n*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_r, n*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_p, n*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Ap, n*sizeof(float)));

    // Copy A and b to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), n*n*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), n*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_x, 0, n*sizeof(float))); // initial x = 0

    // cuBLAS handle
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    // Scalars for cuBLAS
    const float one = 1.0f;
    const float zero = 0.0f;
    //const float neg_one = -1.0f;

    // r0 = b - A*x0  (x0 = 0 -> r0 = b)
    CUBLAS_CHECK(cublasScopy(handle, n, d_b, 1, d_r, 1)); // r = b
    // p0 = r0
    CUBLAS_CHECK(cublasScopy(handle, n, d_r, 1, d_p, 1));

    // rsold = r' * r
    float rsold;
    CUBLAS_CHECK(cublasSdot(handle, n, d_r, 1, d_r, 1, &rsold));

    const int max_iter = 1000;
    const float tol = 1e-6f;
    int k;
    for (k = 0; k < max_iter; ++k) {
        // Ap = A * p  (use cublasSgemv: y = alpha*A*x + beta*y)
        // A is column-major, no transpose
        CUBLAS_CHECK(cublasSgemv(handle,
                                CUBLAS_OP_N,
                                n,      // rows
                                n,      // cols
                                &one,
                                d_A,    // A (n x n)
                                n,      // lda
                                d_p,    // x
                                1,
                                &zero,
                                d_Ap,
                                1));   // Ap = A * p

        // alpha = rsold / (p' * Ap)
        float pAp;
        CUBLAS_CHECK(cublasSdot(handle, n, d_p, 1, d_Ap, 1, &pAp));
        if (std::fabs(pAp) < 1e-12f) {
            std::cerr << "Break: p'Ap ~ 0 (numerical issue)\n";
            break;
        }
        float alpha = rsold / pAp;

        // x = x + alpha * p
        CUBLAS_CHECK(cublasSaxpy(handle, n, &alpha, d_p, 1, d_x, 1));

        // r = r - alpha * Ap  (use axpy with -alpha)
        float neg_alpha = -alpha;
        CUBLAS_CHECK(cublasSaxpy(handle, n, &neg_alpha, d_Ap, 1, d_r, 1));

        // rsnew = r' * r
        float rsnew;
        CUBLAS_CHECK(cublasSdot(handle, n, d_r, 1, d_r, 1, &rsnew));

        // Check convergence: sqrt(rsnew) = ||r||
        float rnorm = std::sqrt(rsnew);
        if (rnorm < tol) {
            std::cout << "Converged at iter " << k+1 << ", ||r|| = " << rnorm << "\n";
            break;
        }

        // p = r + (rsnew/rsold) * p
        float beta = rsnew / rsold;
        // p = beta * p
        CUBLAS_CHECK(cublasSscal(handle, n, &beta, d_p, 1));
        // p = p + r
        CUBLAS_CHECK(cublasSaxpy(handle, n, &one, d_r, 1, d_p, 1));

        rsold = rsnew;
    }

    // Copy solution x back to host
    std::vector<float> h_x(n);
    CUDA_CHECK(cudaMemcpy(h_x.data(), d_x, n*sizeof(float), cudaMemcpyDeviceToHost));

    std::cout << "Solution x (GPU CG): [ ";
    for (auto v : h_x) std::cout << v << " ";
    std::cout << "]\n";

    // Compute final residual norm on host for verification: r = b - A*x
    std::vector<float> Ax(n, 0.0f);
    // Compute Ax on host using h_A (column-major)
    for (int col = 0; col < n; ++col) {
        for (int row = 0; row < n; ++row) {
            Ax[row] += h_A[col*n + row] * h_x[col];
        }
    }
    std::vector<float> h_res(n);
    float res_norm = 0.0f;
    for (int i = 0; i < n; ++i) {
        h_res[i] = h_b[i] - Ax[i];
        res_norm += h_res[i] * h_res[i];
    }
    res_norm = std::sqrt(res_norm);
    std::cout << "Residual norm ||b - Ax|| = " << res_norm << std::endl;

    // Cleanup
    CUBLAS_CHECK(cublasDestroy(handle));
    cudaFree(d_A); cudaFree(d_x); cudaFree(d_b);
    cudaFree(d_r); cudaFree(d_p); cudaFree(d_Ap);

    return 0;
}
