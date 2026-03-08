#include <metal_stdlib>
using namespace metal;

// Our compute kernel for Matrix Multiplication (GEMM)
// Calculates C = A * B
kernel void gemm_kernel(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& M [[buffer(3)]], // Rows of A/C
    constant uint& K [[buffer(4)]], // Cols of A / Rows of B
    constant uint& N [[buffer(5)]], // Cols of B/C
    uint2 gid [[thread_position_in_grid]]
) {
    // Determine the row and column this thread is responsible for
    uint row = gid.y;
    uint col = gid.x;
    
    // Check bounds to avoid out-of-bounds memory access
    if (row >= M || col >= N) {
        return;
    }
    
    // Perform the dot product for this row of A and column of B
    float sum = 0.0;
    for (uint i = 0; i < K; ++i) {
        // A is M x K, B is K x N
        // Index conceptually: A[row][i] * B[i][col]
        sum += A[row * K + i] * B[i * N + col];
    }
    
    // Write result to C
    C[row * N + col] = sum;
}
