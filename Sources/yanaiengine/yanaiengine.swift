import Foundation
import Metal

@main
struct yanaiengine {
    static func main() {
        print("==============================================")
        print("  YanAIEngine: Goal #7 — Transformer Block")
        print("==============================================\n")
        
        guard let engine = MetalEngine() else { fatalError("Metal not available") }
        
        // Load all kernels
        let kernels = [
            "gemm_kernel", "bias_add_kernel", "relu_kernel",
            "transpose_kernel", "mse_derivative_kernel", "sgd_update_kernel",
            "relu_derivative_kernel", "sum_rows_kernel",
            "softmax_kernel", "scale_kernel", "causal_mask_kernel",
            // New for Goal #7:
            "gelu_kernel", "layernorm_kernel", "elementwise_add_kernel"
        ]
        for kernel in kernels {
            engine.loadLibrary(resourceName: "gemm", kernelName: kernel)
        }
        
        // Configuration
        let seqLen = 4     // 4 tokens
        let dModel = 8     // 8-dim embeddings
        let numHeads = 2   // 2 attention heads (dHead = 4)
        
        print("Configuration:")
        print("  Sequence Length:  \(seqLen)")
        print("  Model Dimension:  \(dModel)")
        print("  Attention Heads:  \(numHeads) (dHead = \(dModel / numHeads))")
        print("  FFN Expansion:    4x (\(dModel) → \(4 * dModel) → \(dModel))")
        print("  Causal Mask:      ON")
        print("  Activation:       GELU\n")
        
        // Create input: [seqLen x dModel]
        let input = Tensor(device: engine.device, rows: seqLen, cols: dModel)
        let ptr = input.pointer()
        for i in 0..<(seqLen * dModel) {
            ptr[i] = Float(i) * 0.1
        }
        
        print("Input Embeddings (\(seqLen) tokens × \(dModel) dims):")
        input.printMatrix()
        
        // Create and run TransformerBlock
        let block = TransformerBlock(engine: engine, seqLen: seqLen, dModel: dModel, numHeads: numHeads)
        
        print("Running Full Transformer Block on GPU...")
        print("  Pipeline: LayerNorm → MHA(2 heads) → Residual → LayerNorm → FFN(GELU) → Residual\n")
        block.forward(input: input)
        
        // ---- Verification ----
        print("--- Transformer Block Output (\(seqLen) tokens × \(dModel) dims) ---")
        block.output.printMatrix()
        
        // Check 1: Output shape matches input shape (residual preserves dimensions)
        let shapeMatch = (block.output.rows == seqLen && block.output.cols == dModel)
        
        // Check 2: Output is not identical to input (transformation happened)
        var isTransformed = false
        let outPtr = block.output.pointer()
        for i in 0..<(seqLen * dModel) {
            if abs(outPtr[i] - ptr[i]) > 1e-6 {
                isTransformed = true
                break
            }
        }
        
        // Check 3: Output contains valid numbers (no NaN/Inf)
        var allFinite = true
        for i in 0..<(seqLen * dModel) {
            if outPtr[i].isNaN || outPtr[i].isInfinite {
                allFinite = false
                break
            }
        }
        
        print("==============================================")
        print("  Verification Results")
        print("==============================================")
        print("  Shape preserved (residual):  \(shapeMatch ? "✅ PASS" : "❌ FAIL")  [\(block.output.rows)×\(block.output.cols)]")
        print("  Transformation applied:      \(isTransformed ? "✅ PASS" : "❌ FAIL")")
        print("  All values finite:           \(allFinite ? "✅ PASS" : "❌ FAIL")")
        print("==============================================")
        
        if shapeMatch && isTransformed && allFinite {
            print("\n🚀 Goal #7 COMPLETE: Full Transformer Block running on Apple Silicon!")
            print("   Architecture: Pre-Norm with Multi-Head Attention + FFN(GELU)")
            print("   This is the exact micro-architecture of GPT, Llama, and Claude.")
        }
    }
}
