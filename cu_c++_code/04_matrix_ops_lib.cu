#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

#define CUDA_CHECK(x) if((x)!=cudaSuccess){ \
    std::cerr<<"CUDA error at "<<__LINE__<<std::endl; return -1;}
#define CUBLAS_CHECK(x) if((x)!=CUBLAS_STATUS_SUCCESS){ \
    std::cerr<<"cuBLAS error at "<<__LINE__<<std::endl; return -1;}
#define CUSOLVER_CHECK(x) if((x)!=CUSOLVER_STATUS_SUCCESS){ \
    std::cerr<<"cuSOLVER error at "<<__LINE__<<std::endl; return -1;}

int main() {
    int m=2, n=2;
    std::vector<float> h_A = {1,2,3,4}; // row-major 2x2
    std::vector<float> h_B = {5,6,7,8};
    std::vector<float> h_C(m*n);

    float *d_A,*d_B,*d_C;
    CUDA_CHECK(cudaMalloc(&d_A, m*n*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, m*n*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, m*n*sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_A,h_A.data(),m*n*sizeof(float),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B,h_B.data(),m*n*sizeof(float),cudaMemcpyHostToDevice));

    cublasHandle_t cublasH;
    cusolverDnHandle_t cusolverH;
    CUBLAS_CHECK(cublasCreate(&cublasH));
    CUSOLVER_CHECK(cusolverDnCreate(&cusolverH));

    float alpha=1.0f, beta=0.0f;

    // --- 1. Matrix Addition: C = A + B
    CUBLAS_CHECK(cublasSgeam(cublasH, CUBLAS_OP_N, CUBLAS_OP_N,
                             m, n, &alpha, d_A, m, &alpha, d_B, m,
                             d_C, m));
    CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, m*n*sizeof(float), cudaMemcpyDeviceToHost));
    std::cout<<"A + B = [ "; for(auto x:h_C) std::cout<<x<<" "; std::cout<<"]"<<std::endl;

    // --- 2. Matrix Subtraction: C = A - B
    float neg=-1.0f;
    CUBLAS_CHECK(cublasSgeam(cublasH, CUBLAS_OP_N, CUBLAS_OP_N,
                             m, n, &alpha, d_A, m, &neg, d_B, m,
                             d_C, m));
    CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, m*n*sizeof(float), cudaMemcpyDeviceToHost));
    std::cout<<"A - B = [ "; for(auto x:h_C) std::cout<<x<<" "; std::cout<<"]"<<std::endl;

    // --- 3. Scalar Multiplication: A = 2*A
    alpha=2.0f;
    CUBLAS_CHECK(cublasSscal(cublasH, m*n, &alpha, d_A, 1));
    CUDA_CHECK(cudaMemcpy(h_C.data(), d_A, m*n*sizeof(float), cudaMemcpyDeviceToHost));
    std::cout<<"2 * A = [ "; for(auto x:h_C) std::cout<<x<<" "; std::cout<<"]"<<std::endl;

    // Reset A
    h_A = {1,2,3,4};
    CUDA_CHECK(cudaMemcpy(d_A,h_A.data(),m*n*sizeof(float),cudaMemcpyHostToDevice));

    // --- 4. Matrix Multiplication: C = A * B
    alpha=1.0f; beta=0.0f;
    CUBLAS_CHECK(cublasSgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N,
                             m, n, m,
                             &alpha, d_A, m, d_B, m, &beta, d_C, m));
    CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, m*n*sizeof(float), cudaMemcpyDeviceToHost));
    std::cout<<"A * B = [ "; for(auto x:h_C) std::cout<<x<<" "; std::cout<<"]"<<std::endl;

    // --- 5. Transpose: C = A^T
    alpha=1.0f; beta=0.0f;
    CUBLAS_CHECK(cublasSgeam(cublasH, CUBLAS_OP_T, CUBLAS_OP_N,
                             m, n, &alpha, d_A, m, &beta, d_A, m, d_C, m));
    CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, m*n*sizeof(float), cudaMemcpyDeviceToHost));
    std::cout<<"A^T = [ "; for(auto x:h_C) std::cout<<x<<" "; std::cout<<"]"<<std::endl;

    // --- 6. Identity Matrix (2x2) ---
    std::vector<float> h_I = {1,0,0,1};
    std::cout<<"I = [ "; for(auto x:h_I) std::cout<<x<<" "; std::cout<<"]"<<std::endl;

    // --- 7. Inverse and Determinant with cuSOLVER ---
    int work_size=0, *devInfo;
    CUDA_CHECK(cudaMalloc(&devInfo, sizeof(int)));
    float *d_inverse;
    CUDA_CHECK(cudaMalloc(&d_inverse, m*n*sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_inverse,h_A.data(),m*n*sizeof(float),cudaMemcpyHostToDevice));

    // Workspace size
    CUSOLVER_CHECK(cusolverDnSgetrf_bufferSize(cusolverH,m,n,d_inverse,m,&work_size));
    float *d_work; CUDA_CHECK(cudaMalloc(&d_work,work_size*sizeof(float)));

    int *d_ipiv; CUDA_CHECK(cudaMalloc(&d_ipiv,m*sizeof(int)));

    // --- LU factorization (A = L*U) ---
    CUSOLVER_CHECK(cusolverDnSgetrf(cusolverH,m,n,d_inverse,m,d_work,d_ipiv,devInfo));
    int h_info; CUDA_CHECK(cudaMemcpy(&h_info,devInfo,sizeof(int),cudaMemcpyDeviceToHost));

    // --- Determinant from U (product of diagonal elements) ---
    std::vector<float> h_LU(m*n);
    CUDA_CHECK(cudaMemcpy(h_LU.data(),d_inverse,m*n*sizeof(float),cudaMemcpyDeviceToHost));
    float det = h_LU[0]*h_LU[3] - h_LU[1]*h_LU[2]; // valid for 2x2
    std::cout<<"det(A) = "<<det<<std::endl;

    // --- Inverse via solving A * X = I ---
    float *d_I;
    CUDA_CHECK(cudaMalloc(&d_I, m*n*sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_I,h_I.data(),m*n*sizeof(float),cudaMemcpyHostToDevice));

    // Solve for inverse (columns of identity)
    CUSOLVER_CHECK(cusolverDnSgetrs(cusolverH,CUBLAS_OP_N,m,n,d_inverse,m,d_ipiv,d_I,m,devInfo));
    CUDA_CHECK(cudaMemcpy(h_C.data(),d_I,m*n*sizeof(float),cudaMemcpyDeviceToHost));

    std::cout<<"A^-1 = [ "; for(auto x:h_C) std::cout<<x<<" "; std::cout<<"]"<<std::endl;

    // Cleanup inverse workspace
    cudaFree(d_I); cudaFree(d_ipiv); cudaFree(d_work); cudaFree(devInfo); cudaFree(d_inverse);

    // Cleanup
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    cudaFree(d_inverse); cudaFree(d_ipiv); cudaFree(d_work); cudaFree(devInfo);
    cublasDestroy(cublasH); cusolverDnDestroy(cusolverH);

    return 0;
}
