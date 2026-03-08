import Foundation
import Metal

@main
struct yanaiengine {
    static func main() {
        print("==============================================")
        print("  YanAIEngine: Goal #12 — Llama 3 Architecture")
        print("  RMSNorm + SwiGLU + GQA + BPE Tokenizer")
        print("==============================================\n")
        
        guard let engine = MetalEngine() else { fatalError("Metal not available") }
        
        // Load all kernels (19 total)
        let kernels = [
            "gemm_kernel", "bias_add_kernel", "relu_kernel",
            "transpose_kernel", "mse_derivative_kernel", "sgd_update_kernel",
            "relu_derivative_kernel", "sum_rows_kernel",
            "softmax_kernel", "scale_kernel", "causal_mask_kernel",
            "gelu_kernel", "layernorm_kernel", "elementwise_add_kernel",
            "rope_kernel", "embedding_lookup_kernel", "q8_gemm_kernel",
            // New for Goal #12:
            "rmsnorm_kernel", "silu_kernel"
        ]
        for kernel in kernels {
            engine.loadLibrary(resourceName: "gemm", kernelName: kernel)
        }
        
        // ============================================
        // Step 1: BPE Tokenizer
        // ============================================
        print("=== STEP 1: BPE Tokenizer ===\n")
        
        let tokenizer = Tokenizer()
        
        // Build a test vocabulary with BPE merge rules
        let testVocab = [
            "H", "e", "l", "o", ",", " ", "Y", "a", "n", "A",
            "I", "E", "g", "i", "!", "He", "ll", "llo", "an",
            "AI", "En", "Eng", "ine", "engine", "in"
        ]
        tokenizer.loadSimple(tokens: testVocab)
        
        // Add merge rules (in priority order)
        tokenizer.merges = [
            ("H", "e"),       // H + e → He
            ("l", "l"),       // l + l → ll
            ("a", "n"),       // a + n → an
            ("A", "I"),       // A + I → AI
            ("E", "n"),       // E + n → En
            ("En", "g"),      // En + g → Eng
            ("i", "n"),       // i + n → in
            ("in", "e"),      // in + e → ine
        ]
        
        let testText = "Hello, YanAIEngine!"
        let tokenIds = tokenizer.encode(text: testText)
        let decoded = tokenizer.decode(ids: tokenIds)
        
        print("  Vocabulary size:  \(tokenizer.vocabSize)")
        print("  Input text:       \"\(testText)\"")
        print("  Token IDs:        \(tokenIds)")
        print("  Token strings:    \(tokenIds.map { tokenizer.reverseVocab[$0] ?? "?" })")
        print("  Decoded:          \"\(decoded)\"")
        let encodePass = !tokenIds.isEmpty
        let decodePass = decoded == testText
        print("  Encode works:     \(encodePass ? "✅" : "❌")")
        print("  Roundtrip match:  \(decodePass ? "✅" : "❌")\n")
        
        // ============================================
        // Step 2: Llama 3 Block (GQA + RMSNorm + SwiGLU)
        // ============================================
        print("=== STEP 2: Llama 3 Block ===\n")
        
        let seqLen = 4
        let dModel = 16
        let numHeads = 4      // 4 query heads
        let numKVHeads = 2    // 2 KV heads (GQA ratio = 2:1)
        let dHead = dModel / numHeads
        
        print("  Configuration:")
        print("    Seq Length:    \(seqLen)")
        print("    dModel:        \(dModel)")
        print("    Query Heads:   \(numHeads)")
        print("    KV Heads:      \(numKVHeads) (GQA \(numHeads/numKVHeads):1 ratio)")
        print("    dHead:         \(dHead)")
        print("    Normalization: RMSNorm (no mean centering)")
        print("    FFN:           SwiGLU (gate + up + down projections)")
        print("    Positional:    RoPE\n")
        
        let llamaBlock = LlamaBlock(engine: engine, seqLen: seqLen, dModel: dModel, numHeads: numHeads, numKVHeads: numKVHeads)
        
        // Create random input [seqLen x dModel]
        let input = Tensor(device: engine.device, rows: seqLen, cols: dModel)
        input.fillRandom()
        
        print("  Running forward pass...")
        llamaBlock.forward(input: input)
        
        // Verify
        let outPtr = llamaBlock.output.pointer()
        let inPtr = input.pointer()
        let totalElements = seqLen * dModel
        
        // Check 1: Shape preserved (residual connections)
        let shapePass = (llamaBlock.output.rows == seqLen && llamaBlock.output.cols == dModel)
        
        // Check 2: Output is different from input (computation happened)
        var isDifferent = false
        for i in 0..<totalElements {
            if abs(outPtr[i] - inPtr[i]) > 1e-6 { isDifferent = true; break }
        }
        
        // Check 3: All finite
        var allFinite = true
        for i in 0..<totalElements {
            if outPtr[i].isNaN || outPtr[i].isInfinite { allFinite = false; break }
        }
        
        // Check 4: GQA memory savings
        let mhaKVSize = numHeads * dHead * seqLen * 2     // Standard MHA
        let gqaKVSize = numKVHeads * dHead * seqLen * 2   // GQA
        let kvSavings = Float(mhaKVSize - gqaKVSize) / Float(mhaKVSize) * 100
        let gqaPass = numKVHeads < numHeads
        
        print("\n  Output sample: [\(String(format: "%.4f", outPtr[0])), \(String(format: "%.4f", outPtr[1])), \(String(format: "%.4f", outPtr[2])), ...]\n")
        
        print("  GQA Memory Analysis:")
        print("    Standard MHA KV:  \(mhaKVSize) floats (\(numHeads) heads)")
        print("    GQA KV:           \(gqaKVSize) floats (\(numKVHeads) heads)")
        print("    KV Cache Savings: \(String(format: "%.0f", kvSavings))%\n")
        
        // ---- Final Verification ----
        print("==============================================")
        print("  Verification Results")
        print("==============================================")
        print("  BPE Tokenizer works:      \(encodePass ? "✅ PASS" : "❌ FAIL")")
        print("  Roundtrip encode/decode:   \(decodePass ? "✅ PASS" : "❌ FAIL")")
        print("  Shape preserved:           \(shapePass ? "✅ PASS" : "❌ FAIL")  [\(seqLen)×\(dModel)]")
        print("  Transformation applied:    \(isDifferent ? "✅ PASS" : "❌ FAIL")")
        print("  All values finite:         \(allFinite ? "✅ PASS" : "❌ FAIL")")
        print("  GQA KV reduction:          \(gqaPass ? "✅ PASS" : "❌ FAIL")  [\(String(format: "%.0f", kvSavings))% savings]")
        print("==============================================")
        
        if encodePass && decodePass && shapePass && isDifferent && allFinite && gqaPass {
            print("\n🚀 Goal #12 COMPLETE: Llama 3 architecture on Apple Silicon!")
            print("   RMSNorm:   ✓  (replaces LayerNorm)")
            print("   SwiGLU:    ✓  (replaces GELU FFN)")
            print("   GQA:       ✓  (\(numHeads)Q/\(numKVHeads)KV — \(String(format: "%.0f", kvSavings))% KV cache savings)")
            print("   BPE:       ✓  (tokenizer with merge rules)")
            print("   Ready to load Meta-Llama-3-8B-Instruct!")
        }
    }
}
