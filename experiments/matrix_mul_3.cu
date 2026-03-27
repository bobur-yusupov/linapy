#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>

int main() {
  int N = 28000;

  size_t bytes = (size_t)N * N * sizeof(float);

  float *h_A = (float *)malloc(bytes);
  float *h_B = (float *)malloc(bytes);
  float *h_C = (float *)malloc(bytes);

  for (size_t i = 0; i < (size_t)N  * N; i++) {
    h_A[i] = 1.0f;
    h_B[i] = 1.0f;
  }

  float *d_A, *d_B, *d_C;
  cudaMalloc(&d_A, bytes);
  cudaMalloc(&d_B, bytes);
  cudaMalloc(&d_C, bytes);

  cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

  cublasHandle_t handle;
  cublasCreate(&handle);

  float alpha = 1.0f;
  float beta = 0.0f;

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);

  cublasSgemm(
    handle, CUBLAS_OP_N, CUBLAS_OP_N,
    N, N, N,
    &alpha, 
    d_B, N, 
    d_A, N, 
    &beta, 
    d_C, N
  );

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);

  double total_operations = 2.0 * (double)N * (double)N * (double)N;
  double seconds = milliseconds / 1000.0;
  double tflops = (total_operations / seconds) / 1e12;

  std::cout << "--- cuBLAS Execution ---" << std::endl;
  std::cout << "Matrix Size: " << N << "x" << N << std::endl;
  std::cout << "Execution Time: " << milliseconds << " ms" << std::endl;
  std::cout << "Throughput: " << tflops << " TFLOPS" << std::endl;

  cublasDestroy(handle);
  cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
  free(h_A); free(h_B); free(h_C);

  return 0;
}
