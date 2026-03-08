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

// Kernel to add a 1D bias vector to a 2D matrix (broadcasting bias across rows)
kernel void bias_add_kernel(
    device float* matrix [[buffer(0)]],
    device const float* bias [[buffer(1)]],
    constant uint& M [[buffer(2)]], // Rows
    constant uint& N [[buffer(3)]], // Cols
    uint2 gid [[thread_position_in_grid]]
) {
    uint row = gid.y;
    uint col = gid.x;
    
    if (row >= M || col >= N) {
        return;
    }
    
    // Each row adds the same bias vector
    matrix[row * N + col] += bias[col];
}

// Element-wise ReLU Activation kernel: f(x) = max(0, x)
kernel void relu_kernel(
    device float* data [[buffer(0)]],
    constant uint& length [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= length) {
        return;
    }
    
    float val = data[gid];
    data[gid] = val > 0.0 ? val : 0.0;
}

// Kernel to transpose a matrix: Output[col][row] = Input[row][col]
kernel void transpose_kernel(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& rows [[buffer(2)]],
    constant uint& cols [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= cols || gid.y >= rows) {
        return;
    }
    
    // input is rows x cols, output is cols x rows
    output[gid.x * rows + gid.y] = input[gid.y * cols + gid.x];
}

// Kernel to calculate the derivative of MSE loss: gradient = (output - target)
kernel void mse_derivative_kernel(
    device const float* output [[buffer(0)]],
    device const float* target [[buffer(1)]],
    device float* derivative [[buffer(2)]],
    constant uint& length [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= length) {
        return;
    }
    
    derivative[gid] = output[gid] - target[gid];
}

// Kernel to update parameters using SGD: param = param - lr * gradient
kernel void sgd_update_kernel(
    device float* param [[buffer(0)]],
    device const float* gradient [[buffer(1)]],
    constant float& lr [[buffer(2)]],
    constant uint& length [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= length) {
        return;
    }
    
    param[gid] -= lr * gradient[gid];
}

// Kernel to calculate the derivative of ReLU: grad_out = grad_in * (original_output > 0 ? 1 : 0)
kernel void relu_derivative_kernel(
    device float* grad_in_out [[buffer(0)]],
    device const float* original_output [[buffer(1)]],
    constant uint& length [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= length) {
        return;
    }
    
    // If the forward pass output was 0 (clamped), the gradient is blocked (0)
    if (original_output[gid] <= 0.0) {
        grad_in_out[gid] = 0.0;
    }
    // Otherwise, gradient passes through (multiplied by 1)
}

// Kernel to sum rows of a matrix (used for bias gradient accumulation)
// Output = sum(Input, dim=0)
kernel void sum_rows_kernel(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& rows [[buffer(2)]],
    constant uint& cols [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= cols) {
        return;
    }
    
    float sum = 0.0;
    for (uint r = 0; r < rows; r++) {
        sum += input[r * cols + gid];
    }
    output[gid] = sum;
}
