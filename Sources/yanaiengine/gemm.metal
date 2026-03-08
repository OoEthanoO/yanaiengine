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

// ============================================================
// Transformer Kernels
// ============================================================

// Row-wise Softmax with numerical stability.
// One thread per row — avoids threadgroup barrier issues with partial threadgroups.
// For each row: find max, subtract, exp, sum, divide.
kernel void softmax_kernel(
    device float* data [[buffer(0)]],
    constant uint& rows [[buffer(1)]],
    constant uint& cols [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    uint row = gid;
    if (row >= rows) return;
    
    // Step 1: Find max in this row
    float row_max = -INFINITY;
    for (uint c = 0; c < cols; c++) {
        row_max = max(row_max, data[row * cols + c]);
    }
    
    // Step 2: Subtract max and compute exp, accumulate sum
    float row_sum = 0.0;
    for (uint c = 0; c < cols; c++) {
        float val = exp(data[row * cols + c] - row_max);
        data[row * cols + c] = val;
        row_sum += val;
    }
    
    // Step 3: Normalize
    for (uint c = 0; c < cols; c++) {
        data[row * cols + c] /= row_sum;
    }
}

// Element-wise scale: data[i] = data[i] * scale_factor
kernel void scale_kernel(
    device float* data [[buffer(0)]],
    constant float& scale [[buffer(1)]],
    constant uint& length [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= length) return;
    data[gid] *= scale;
}

// Causal Mask: sets upper triangle of a square matrix to -INFINITY
// For position (row, col): if col > row, set to -INFINITY
kernel void causal_mask_kernel(
    device float* data [[buffer(0)]],
    constant uint& dim [[buffer(1)]],   // square matrix dimension (seqLen)
    uint2 gid [[thread_position_in_grid]]
) {
    uint row = gid.y;
    uint col = gid.x;
    
    if (row >= dim || col >= dim) return;
    
    if (col > row) {
        data[row * dim + col] = -INFINITY;
    }
}

// GELU Activation: the activation function used by GPT and Llama.
// GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
kernel void gelu_kernel(
    device float* data [[buffer(0)]],
    constant uint& length [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= length) return;
    
    float x = data[gid];
    float cdf = 0.5 * (1.0 + tanh(0.7978845608 * (x + 0.044715 * x * x * x)));
    data[gid] = x * cdf;
}

// Layer Normalization: normalizes each row (token) across the embedding dimension.
// y = gamma * (x - mean) / sqrt(variance + eps) + beta
// One thread per row — avoids threadgroup barrier issues.
kernel void layernorm_kernel(
    device float* data [[buffer(0)]],
    device const float* gamma [[buffer(1)]],
    device const float* beta [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    constant uint& cols [[buffer(4)]],
    constant float& eps [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    uint row = gid;
    if (row >= rows) return;
    
    // Compute mean
    float sum = 0.0;
    for (uint c = 0; c < cols; c++) {
        sum += data[row * cols + c];
    }
    float mean = sum / float(cols);
    
    // Compute variance
    float var_sum = 0.0;
    for (uint c = 0; c < cols; c++) {
        float diff = data[row * cols + c] - mean;
        var_sum += diff * diff;
    }
    float inv_std = 1.0 / sqrt(var_sum / float(cols) + eps);
    
    // Normalize and apply affine transform
    for (uint c = 0; c < cols; c++) {
        float normalized = (data[row * cols + c] - mean) * inv_std;
        data[row * cols + c] = gamma[c] * normalized + beta[c];
    }
}

// Element-wise addition: output[i] = a[i] + b[i]
// Used for residual connections.
kernel void elementwise_add_kernel(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& length [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= length) return;
    output[gid] = a[gid] + b[gid];
}

// Rotary Positional Embedding (RoPE): rotates Q/K pairs by position-dependent angles.
// One thread per (position, dimension_pair).
// data layout: [seqLen x dHead], each thread handles one pair (2i, 2i+1).
kernel void rope_kernel(
    device float* data [[buffer(0)]],
    constant uint& seqLen [[buffer(1)]],
    constant uint& dHead [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]]  // x = pair_index, y = position
) {
    uint pos = gid.y;
    uint pair = gid.x;
    uint numPairs = dHead / 2;
    
    if (pos >= seqLen || pair >= numPairs) return;
    
    // θ_i = pos / 10000^(2i / dHead)
    float theta = float(pos) / pow(10000.0, float(2 * pair) / float(dHead));
    float cos_theta = cos(theta);
    float sin_theta = sin(theta);
    
    uint idx0 = pos * dHead + 2 * pair;
    uint idx1 = idx0 + 1;
    
    float x0 = data[idx0];
    float x1 = data[idx1];
    
    data[idx0] = x0 * cos_theta - x1 * sin_theta;
    data[idx1] = x0 * sin_theta + x1 * cos_theta;
}

// Embedding Lookup: copies rows from weight matrix based on token IDs.
// output[i] = weights[tokenIds[i]]  (i.e., row tokenIds[i] of weight matrix)
// One thread per (token_index, dimension).
kernel void embedding_lookup_kernel(
    device const uint* tokenIds [[buffer(0)]],
    device const float* weights [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& seqLen [[buffer(3)]],
    constant uint& dModel [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]  // x = dim, y = token_index
) {
    uint tokenIdx = gid.y;
    uint dim = gid.x;
    
    if (tokenIdx >= seqLen || dim >= dModel) return;
    
    uint tokenId = tokenIds[tokenIdx];
    output[tokenIdx * dModel + dim] = weights[tokenId * dModel + dim];
}

// INT8 Weight-Only Quantized GEMM: C = A * dequantize(B)
// A: Float input activations [M x K]
// B: INT8 quantized weights [K x N] (stored as char/int8)
// scales: Float scale factors [K] (one per row of B)
// C: Float output [M x N]
// Dequantization: float_weight = int8_weight * scale[k]
kernel void q8_gemm_kernel(
    device const float* A [[buffer(0)]],
    device const char* B [[buffer(1)]],
    device const float* scales [[buffer(2)]],
    device float* C [[buffer(3)]],
    constant uint& M [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    constant uint& N [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint row = gid.y;
    uint col = gid.x;
    
    if (row >= M || col >= N) return;
    
    float sum = 0.0;
    for (uint k = 0; k < K; k++) {
        float a_val = A[row * K + k];
        // Dequantize: read 1 byte, multiply by scale to recover approximate float
        float b_val = float(B[k * N + col]) * scales[k];
        sum += a_val * b_val;
    }
    C[row * N + col] = sum;
}

// RMSNorm: x * rsqrt(mean(x²) + eps) * gamma
// Llama 3 replacement for LayerNorm (no mean centering).
// One thread per row. Data is modified in-place.
kernel void rmsnorm_kernel(
    device float* data [[buffer(0)]],
    device const float* gamma [[buffer(1)]],
    constant uint& rows [[buffer(2)]],
    constant uint& cols [[buffer(3)]],
    constant float& eps [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= rows) return;
    
    uint offset = gid * cols;
    
    // Compute mean of squares
    float sumSq = 0.0;
    for (uint c = 0; c < cols; c++) {
        float val = data[offset + c];
        sumSq += val * val;
    }
    float rms = rsqrt(sumSq / float(cols) + eps);
    
    // Normalize and scale by gamma
    for (uint c = 0; c < cols; c++) {
        data[offset + c] = data[offset + c] * rms * gamma[c];
    }
}

// SiLU (Sigmoid Linear Unit): x * sigmoid(x)
// Used inside SwiGLU activation for Llama 3 FFN.
kernel void silu_kernel(
    device float* data [[buffer(0)]],
    constant uint& length [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= length) return;
    float x = data[gid];
    data[gid] = x / (1.0 + exp(-x));
}
