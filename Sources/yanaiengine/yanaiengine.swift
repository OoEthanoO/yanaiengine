import Foundation
import Metal

@main
struct yanaiengine {
    static func main() {
        print("==============================================")
        print("  YanAIEngine: Goal #10 — INT8 Quantization")
        print("  4x Weight Compression with On-the-Fly Dequant")
        print("==============================================\n")
        
        guard let engine = MetalEngine() else { fatalError("Metal not available") }
        
        // Load all kernels
        let kernels = [
            "gemm_kernel", "bias_add_kernel", "relu_kernel",
            "transpose_kernel", "mse_derivative_kernel", "sgd_update_kernel",
            "relu_derivative_kernel", "sum_rows_kernel",
            "softmax_kernel", "scale_kernel", "causal_mask_kernel",
            "gelu_kernel", "layernorm_kernel", "elementwise_add_kernel",
            "rope_kernel", "embedding_lookup_kernel",
            // New for Goal #10:
            "q8_gemm_kernel"
        ]
        for kernel in kernels {
            engine.loadLibrary(resourceName: "gemm", kernelName: kernel)
        }
        
        // ---- Configuration ----
        let batchSize = 4
        let inputDim = 64
        let outputDim = 32
        
        print("Test Configuration:")
        print("  Batch Size:   \(batchSize)")
        print("  Input Dim:    \(inputDim)")
        print("  Output Dim:   \(outputDim)")
        print("  Weights:      \(inputDim)×\(outputDim) = \(inputDim * outputDim) parameters\n")
        
        // ---- Create FP32 Linear Layer ----
        let fp32Layer = LinearLayer(engine: engine, inputDim: inputDim, outputDim: outputDim, batchSize: batchSize, useReLU: false)
        let fp32WeightBytes = inputDim * outputDim * MemoryLayout<Float>.stride
        
        // ---- Create INT8 Quantized Layer (from the FP32 layer) ----
        let int8Layer = QuantizedLinearLayer(engine: engine, from: fp32Layer)
        
        print("=== MEMORY COMPARISON ===")
        print("  FP32 weights: \(fp32WeightBytes) bytes (\(fp32WeightBytes / 1024) KB)")
        print("  INT8 weights: \(int8Layer.int8WeightBytes) bytes (\(int8Layer.int8WeightBytes / 1024) KB)")
        let ratio = Float(fp32WeightBytes) / Float(int8Layer.int8WeightBytes)
        print("  Compression:  \(String(format: "%.1f", ratio))x\n")
        
        // ---- Create Test Input ----
        let input = Tensor(device: engine.device, rows: batchSize, cols: inputDim)
        let ptr = input.pointer()
        for i in 0..<(batchSize * inputDim) {
            ptr[i] = Float.random(in: -1.0...1.0)
        }
        
        // ---- Run FP32 Forward Pass ----
        print("Running FP32 forward pass...")
        fp32Layer.forward(input: input)
        
        // ---- Run INT8 Forward Pass ----
        print("Running INT8 quantized forward pass...\n")
        int8Layer.forward(input: input)
        
        // ---- Compare Outputs ----
        let fp32Out = fp32Layer.output.pointer()
        let int8Out = int8Layer.output.pointer()
        let totalElements = batchSize * outputDim
        
        var maxError: Float = 0
        var totalError: Float = 0
        
        for i in 0..<totalElements {
            let err = abs(fp32Out[i] - int8Out[i])
            totalError += err
            if err > maxError { maxError = err }
        }
        
        let meanError = totalError / Float(totalElements)
        
        // Compute relative error (compared to FP32 output magnitude)
        var fp32Magnitude: Float = 0
        for i in 0..<totalElements {
            fp32Magnitude += abs(fp32Out[i])
        }
        let meanMagnitude = fp32Magnitude / Float(totalElements)
        let relativeError = meanError / max(meanMagnitude, 1e-8) * 100
        
        print("=== FP32 vs INT8 OUTPUT COMPARISON ===")
        print("  FP32 output sample: [\(String(format: "%.4f", fp32Out[0])), \(String(format: "%.4f", fp32Out[1])), \(String(format: "%.4f", fp32Out[2])), ...]")
        print("  INT8 output sample: [\(String(format: "%.4f", int8Out[0])), \(String(format: "%.4f", int8Out[1])), \(String(format: "%.4f", int8Out[2])), ...]")
        print("  Mean Absolute Error: \(String(format: "%.6f", meanError))")
        print("  Max Absolute Error:  \(String(format: "%.6f", maxError))")
        print("  Relative Error:      \(String(format: "%.2f", relativeError))%\n")
        
        // ---- Verify with a larger "model-scale" layer ----
        print("=== SCALE TEST (simulating real model layer) ===")
        let bigInput = 1024
        let bigOutput = 4096
        let bigBatch = 1
        let bigLayer = LinearLayer(engine: engine, inputDim: bigInput, outputDim: bigOutput, batchSize: bigBatch, useReLU: false)
        let bigQuantized = QuantizedLinearLayer(engine: engine, from: bigLayer)
        let bigFp32Bytes = bigInput * bigOutput * MemoryLayout<Float>.stride
        print("  Layer: \(bigInput)×\(bigOutput) = \(bigInput * bigOutput) parameters")
        print("  FP32: \(bigFp32Bytes / 1024) KB")
        print("  INT8: \(bigQuantized.int8WeightBytes / 1024) KB")
        print("  Savings: \(String(format: "%.1f", Float(bigFp32Bytes) / Float(bigQuantized.int8WeightBytes)))x compression\n")
        
        // ---- Verification ----
        let compressionPass = ratio > 3.5  // Should be close to 4x
        let accuracyPass = relativeError < 5.0  // Less than 5% relative error
        let outputFinite = (0..<totalElements).allSatisfy { !int8Out[$0].isNaN && !int8Out[$0].isInfinite }
        
        print("==============================================")
        print("  Verification Results")
        print("==============================================")
        print("  4x compression achieved:  \(compressionPass ? "✅ PASS" : "❌ FAIL")  [\(String(format: "%.1f", ratio))x]")
        print("  Accuracy preserved (<5%): \(accuracyPass ? "✅ PASS" : "❌ FAIL")  [\(String(format: "%.2f", relativeError))%]")
        print("  All outputs finite:       \(outputFinite ? "✅ PASS" : "❌ FAIL")")
        print("==============================================")
        
        if compressionPass && accuracyPass && outputFinite {
            print("\n🚀 Goal #10 COMPLETE: INT8 quantized inference on Apple Silicon!")
            print("   Weights compressed 4x: FP32 → INT8 with per-row scaling")
            print("   Dequantization happens on-the-fly inside GPU registers")
            print("   This is the exact technique behind llama.cpp and GGML.")
        }
    }
}
