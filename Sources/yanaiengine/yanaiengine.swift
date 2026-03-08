import Foundation
import Metal

@main
struct yanaiengine {
    static func main() {
        let args = CommandLine.arguments
        
        // Support legacy distributed mode
        if args.contains("--master") || args.contains("--worker") {
            print("Distributed mode requires two terminals. See README.md.")
            return
        }
        
        print("==============================================")
        print("  YanAIEngine: Goal #6 — Self-Attention")
        print("==============================================\n")
        
        guard let engine = MetalEngine() else { fatalError("Metal not available") }
        
        // Load all kernels (existing + new Transformer kernels)
        let kernels = [
            "gemm_kernel", "bias_add_kernel", "relu_kernel",
            "transpose_kernel", "mse_derivative_kernel", "sgd_update_kernel",
            "relu_derivative_kernel", "sum_rows_kernel",
            // New for Goal #6:
            "softmax_kernel", "scale_kernel", "causal_mask_kernel"
        ]
        for kernel in kernels {
            engine.loadLibrary(resourceName: "gemm", kernelName: kernel)
        }
        
        // ---- Setup ----
        let seqLen = 4   // 4 tokens in the sequence
        let dModel = 8   // 8-dimensional embedding per token
        
        print("Configuration:")
        print("  Sequence Length: \(seqLen)")
        print("  Model Dimension: \(dModel)")
        print("  Causal Mask:     ON (autoregressive)\n")
        
        // Create a mock input: [seqLen x dModel] — simulating 4 token embeddings
        let input = Tensor(device: engine.device, rows: seqLen, cols: dModel)
        let ptr = input.pointer()
        for i in 0..<(seqLen * dModel) {
            ptr[i] = Float(i) * 0.1  // Simple ascending pattern
        }
        
        print("Input Embeddings (4 tokens x 8 dims):")
        input.printMatrix()
        
        // ---- Create Self-Attention Layer ----
        let attention = SelfAttention(engine: engine, seqLen: seqLen, dModel: dModel, useCausalMask: true)
        
        // ---- Forward Pass ----
        print("Running Scaled Dot-Product Attention on GPU...")
        attention.forward(input: input)
        
        // ---- Verify Results ----
        print("\n--- Attention Weights (after softmax + causal mask) ---")
        print("Each row should sum to 1.0. Upper triangle should be 0.0.\n")
        
        let scorePtr = attention.scores.pointer()
        for row in 0..<seqLen {
            var rowStr = "Token \(row): ["
            var rowSum: Float = 0
            for col in 0..<seqLen {
                let val = scorePtr[row * seqLen + col]
                rowSum += val
                rowStr += String(format: "%.4f", val)
                if col < seqLen - 1 { rowStr += ", " }
            }
            rowStr += "]  sum=\(String(format: "%.4f", rowSum))"
            print(rowStr)
        }
        
        print("\n--- Attention Output (4 tokens x 8 dims) ---")
        attention.output.printMatrix()
        
        // ---- Validation ----
        var allRowsSumToOne = true
        var upperTriangleZero = true
        
        for row in 0..<seqLen {
            var rowSum: Float = 0
            for col in 0..<seqLen {
                let val = scorePtr[row * seqLen + col]
                rowSum += val
                if col > row && val > 1e-6 {
                    upperTriangleZero = false
                }
            }
            if abs(rowSum - 1.0) > 1e-4 {
                allRowsSumToOne = false
            }
        }
        
        print("\n==============================================")
        print("  Verification Results")
        print("==============================================")
        print("  Softmax rows sum to 1.0:  \(allRowsSumToOne ? "✅ PASS" : "❌ FAIL")")
        print("  Causal mask (upper = 0):  \(upperTriangleZero ? "✅ PASS" : "❌ FAIL")")
        print("==============================================")
        
        if allRowsSumToOne && upperTriangleZero {
            print("\n🚀 Goal #6 COMPLETE: Self-Attention is running natively on Apple Silicon!")
            print("   Your engine can now execute the core of GPT / Llama / Transformer.")
        }
    }
}
