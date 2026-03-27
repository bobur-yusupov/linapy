#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>

__global__ void initMatrix(half *mat, float val, size_t size) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < size) {
    mat[idx] = __float2half(val);
  }
}

int main() {
  int N = 28000;

  size_t elements = (size_t)N * N;
  size_t bytes = elements * sizeof(half);

  half *d_A, *d_B, *d_C;
  cudaMalloc(&d_A, bytes);
  cudaMalloc(&d_B, bytes);
  cudaMalloc(&d_C, bytes);

  int threads = 256;
  int blocks = (elements + threads - 1) / threads;
  initMatrix<<<blocks, threads>>>(d_A, 1.0f, elements);
  initMatrix<<<blocks, threads>>>(d_B, 1.0f, elements);
  cudaDeviceSynchronize();

  cublasHandle_t handle;
  cublasCreate(&handle);

  cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);

  float alpha = 1.0f;
  float beta = 0.0f;

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
              N, N, N,
              &alpha,
              d_B, CUDA_R_16F, N,
              d_A, CUDA_R_16F, N,
              &beta,
              d_C, CUDA_R_16F, N,
              CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
  
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);

  double total_operations = 2.0 * (double)N * (double)N * (double)N;
  double seconds = milliseconds / 1000.0;
  double tflops = (total_operations / seconds) / 1e12;

  std::cout << "--- TENSOR CORE EXECUTION ---" << std::endl;
  std::cout << "Matrix Size: " << N << "x" << N << std::endl;
  std::cout << "VRAM Used per Matrix: " << bytes / (1024.0*1024.0*1024.0) << " GB" << std::endl;
  std::cout << "Execution Time: " << milliseconds << " ms" << std::endl;
  std::cout << "Throughput: " << tflops << " TFLOPS" << std::endl;

  cublasDestroy(handle);
  cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);

  return 0;
}
