#include <iostream>
#include <cuda_runtime.h>

// The Metal: Our Tiled Kernel
__global__ void matrixMulTiled(float *A, float *B, float *C, int N) {
  const int TILE_SIZE = 16;
  __shared__ float tileA[TILE_SIZE][TILE_SIZE];
  __shared__ float tileB[TILE_SIZE][TILE_SIZE];

  int bx = blockIdx.x; int by = blockIdx.y;
  int tx = threadIdx.x; int ty = threadIdx.y;

  int row = by * TILE_SIZE + ty;
  int col = bx * TILE_SIZE + tx;

  float sum = 0.0f;

  for (int i = 0; i < N / TILE_SIZE; ++i) {
    tileA[ty][tx] = A[row * N + (i * TILE_SIZE + tx)];
    tileB[ty][tx] = B[(i * TILE_SIZE + ty) * N + col];
    __syncthreads();

    for (int k = 0; k < TILE_SIZE; ++k) {
      sum += tileA[ty][k] * tileB[k][tx];
    }
    __syncthreads();
  }
  C[row * N + col] = sum;
}

// 2. The Host (Executes on the CPU)
int main() {
  int N = 28000; // Matrix Size: 4096 x 4096
  size_t bytes = N * N * sizeof(float);

  float *h_A = (float *)malloc(bytes);
  float *h_B = (float *)malloc(bytes);
  float *h_C = (float *)malloc(bytes);

  // Fill matrices with the number 1.0 for testing
  for (int i = 0; i < N * N; i++) {
    h_A[i] = 1.0f;
    h_B[i] = 1.0f;
  }

  // Allocate memory on the Device (GPU RAM)
  float *d_A, *d_B, *d_C;
  cudaMalloc(&d_A, bytes);
  cudaMalloc(&d_B, bytes);
  cudaMalloc(&d_C, bytes);

  // Push data across the PCIe (Peripheral Component Interconnect Express) bus to the VRAM
  cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);
  
  // Define the grid of threads mapping to the physical hardware
  dim3 threadsPerBlock(16, 16);
  dim3 blocksPerGrid(N / 16, N / 16);

  // -- INSTRUMENTATION START
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // Record the exact millisecond the GPU starts the math
  cudaEventRecord(start);
  
  // Launch the Kernel!
  matrixMulTiled<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

  // Record the exact millisecond the GPU finishes
  cudaEventRecord(stop);

  // Force the CPU to wait until the GPU physically hits the stop event
  cudaEventSynchronize(stop);

  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  // -- INSTRUMENTATION END

  // Pull the result back across the PCIe bus to the CPU RAM
  cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);
  // 1. Calculate raw operations: 2 * N^3
  double total_operations = 2.0 * (double)N * (double)N * (double)N;
    
  // 2. Calculate TeraFLOPS
  double seconds = milliseconds / 1000.0;
  double tflops = (total_operations / seconds) / 1e12;

  std::cout << "Matrix Size: " << N << "x" << N << std::endl;
  std::cout << "Execution Time: " << milliseconds << " ms" << std::endl;
  std::cout << "Throughput: " << tflops << " TFLOPS" << std::endl;

  cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
  free(h_A); free(h_B); free(h_C);
  cudaEventDestroy(start); cudaEventDestroy(stop);

  return 0;
}
