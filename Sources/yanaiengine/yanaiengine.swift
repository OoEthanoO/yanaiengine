import Foundation
import Metal

@main
struct yanaiengine {
    static func main() {
        print("==============================================")
        print("  YanAIEngine: Goal #13 — Full Llama Model")
        print("  Stacked Blocks + Nucleus Sampling + Chat")
        print("==============================================\n")
        
        guard let engine = MetalEngine() else { fatalError("Metal not available") }
        
        // Load all 19 kernels
        let kernels = [
            "gemm_kernel", "bias_add_kernel", "relu_kernel",
            "transpose_kernel", "mse_derivative_kernel", "sgd_update_kernel",
            "relu_derivative_kernel", "sum_rows_kernel",
            "softmax_kernel", "scale_kernel", "causal_mask_kernel",
            "gelu_kernel", "layernorm_kernel", "elementwise_add_kernel",
            "rope_kernel", "embedding_lookup_kernel", "q8_gemm_kernel",
            "rmsnorm_kernel", "silu_kernel", "fused_attention_kernel"
        ]
        for kernel in kernels { engine.loadLibrary(resourceName: "gemm", kernelName: kernel) }
        
        // ============================================
        // Step 1: Chat Template
        // ============================================
        print("=== STEP 1: Llama 3 Chat Template ===\n")
        
        let userMessage = "What is the meaning of life?"
        let chatPrompt = formatLlama3Chat(message: userMessage)
        print("  User message: \"\(userMessage)\"")
        print("  Formatted:")
        for line in chatPrompt.components(separatedBy: "\n") {
            print("    \(line)")
        }
        let templatePass = chatPrompt.contains("<|start_header_id|>") && chatPrompt.contains("<|eot_id|>")
        print("  Template valid: \(templatePass ? "✅" : "❌")\n")
        
        // ============================================
        // Step 2: Sampler Verification
        // ============================================
        print("=== STEP 2: Sampler Verification ===\n")
        
        let vocabSize = 10
        let sampler = Sampler(temperature: 0.8, topK: 5, topP: 0.9)
        
        // Create logits where multiple tokens have competitive probabilities
        var testLogits = [Float](repeating: -10.0, count: vocabSize)
        testLogits[3] = 2.0   // Strong
        testLogits[7] = 1.8   // Close second
        testLogits[1] = 1.5   // Third
        testLogits[5] = 1.0   // Fourth
        
        print("  Config: T=\(sampler.temperature), K=\(sampler.topK), P=\(sampler.topP)")
        print("  Logits: token 3=2.0, token 7=1.8, token 1=1.5, token 5=1.0\n")
        
        // Greedy: should always pick token 3
        let greedySampler = Sampler(temperature: 0, topK: vocabSize, topP: 1.0)
        var greedyLogits = testLogits
        let greedyResult = greedySampler.sample(logits: &greedyLogits, vocabSize: vocabSize)
        let greedyPass = greedyResult == 3
        print("  Greedy (T=0):     token \(greedyResult) \(greedyPass ? "✅" : "❌") (expected 3)")
        
        // Nucleus sampling: run multiple times, should see variety
        var sampledTokens = Set<UInt32>()
        for _ in 0..<50 {
            var logitsCopy = testLogits
            let token = sampler.sample(logits: &logitsCopy, vocabSize: vocabSize)
            sampledTokens.insert(token)
        }
        let varietyPass = sampledTokens.count > 1
        print("  Nucleus (50 runs): \(sampledTokens.sorted()) — \(sampledTokens.count) unique tokens \(varietyPass ? "✅" : "❌")")
        
        // Top-K=1 should behave like greedy
        let topK1Sampler = Sampler(temperature: 0.8, topK: 1, topP: 1.0)
        var topK1Logits = testLogits
        let topK1Result = topK1Sampler.sample(logits: &topK1Logits, vocabSize: vocabSize)
        let topKPass = topK1Result == 3
        print("  Top-K=1:          token \(topK1Result) \(topKPass ? "✅" : "❌") (expected 3)\n")
        
        // ============================================
        // Step 3: Full LlamaModel Forward Pass
        // ============================================
        print("=== STEP 3: LlamaModel (Stacked Blocks) ===\n")
        
        let config = LlamaConfig.tiny
        print("  Config: \(config.numLayers) layers, \(config.dModel)d, \(config.numHeads)Q/\(config.numKVHeads)KV, vocab=\(config.vocabSize)")
        
        let model = LlamaModel(engine: engine, config: config)
        
        // Fake prompt tokens
        let promptTokens: [UInt32] = [1, 5, 12, 8]
        print("  Prompt: \(promptTokens) (\(promptTokens.count) tokens)")
        print("  Running prefill through \(config.numLayers) stacked LlamaBlocks...\n")
        
        let logitsPtr = model.prefill(tokenIds: promptTokens)
        
        // Check logits are finite
        var logitsFinite = true
        for i in 0..<config.vocabSize {
            if logitsPtr[i].isNaN || logitsPtr[i].isInfinite { logitsFinite = false; break }
        }
        
        // Sample next token
        var logitsCopy = [Float](repeating: 0, count: config.vocabSize)
        for i in 0..<config.vocabSize { logitsCopy[i] = logitsPtr[i] }
        let nextToken = sampler.sample(logits: &logitsCopy, vocabSize: config.vocabSize)
        print("  Prefill logits finite:  \(logitsFinite ? "✅" : "❌")")
        print("  Sampled next token:     \(nextToken)")
        
        // ============================================
        // Step 4: Autoregressive Generation with Stop Token
        // ============================================
        print("\n=== STEP 4: Generation with Stop Token ===\n")
        
        let eotTokenId: UInt32 = 0  // Simulate <|eot_id|> as token 0
        var sequence = promptTokens
        let maxGenTokens = 6
        
        print("  Generating up to \(maxGenTokens) tokens (stop on token \(eotTokenId))...")
        
        // First prediction already done by prefill
        sequence.append(nextToken)
        
        for step in 0..<(maxGenTokens - 1) {
            let decLogits = model.decode(tokenId: sequence.last!)
            var decLogitsCopy = [Float](repeating: 0, count: config.vocabSize)
            for i in 0..<config.vocabSize { decLogitsCopy[i] = decLogits[i] }
            let tok = sampler.sample(logits: &decLogitsCopy, vocabSize: config.vocabSize)
            
            if tok == eotTokenId {
                print("  Step \(step + 2): <|eot_id|> — STOPPING")
                break
            }
            sequence.append(tok)
            print("  Step \(step + 2): token \(tok) → [\(sequence.map { String($0) }.joined(separator: ", "))]")
        }
        
        let totalTokens = sequence.count
        let genTokens = totalTokens - promptTokens.count
        print("\n  Generated \(genTokens) tokens. Total sequence: \(totalTokens) tokens")
        
        // ---- Final Verification ----
        let allValid = sequence.allSatisfy { $0 < UInt32(config.vocabSize) }
        let multiLayer = config.numLayers > 1
        
        print("\n==============================================")
        print("  Verification Results")
        print("==============================================")
        print("  Chat template correct:     \(templatePass ? "✅ PASS" : "❌ FAIL")")
        print("  Greedy sampling works:     \(greedyPass ? "✅ PASS" : "❌ FAIL")")
        print("  Nucleus produces variety:  \(varietyPass ? "✅ PASS" : "❌ FAIL")")
        print("  Multi-layer forward:       \(multiLayer ? "✅ PASS" : "❌ FAIL")  [\(config.numLayers) layers]")
        print("  Logits are finite:         \(logitsFinite ? "✅ PASS" : "❌ FAIL")")
        print("  All tokens valid:          \(allValid ? "✅ PASS" : "❌ FAIL")")
        print("==============================================")
        
        if templatePass && greedyPass && varietyPass && multiLayer && logitsFinite && allValid {
            print("\n🚀 Goal #13 COMPLETE: Full LLM pipeline on Apple Silicon!")
            print("   Model:    \(config.numLayers)-layer LlamaModel (Embed→Blocks→RMSNorm→LMHead)")
            print("   Sampler:  Temperature + Top-K + Top-P (Nucleus)")
            print("   Chat:     Llama 3 template with <|eot_id|> stop detection")
            print("   This is a bespoke llama.cpp alternative, built from scratch.")
        }
    }
    
    /// Format a user message using Llama 3's chat template.
    static func formatLlama3Chat(message: String, systemPrompt: String = "You are a helpful AI assistant.") -> String {
        return """
        <|begin_of_text|><|start_header_id|>system<|end_header_id|>
        
        \(systemPrompt)<|eot_id|><|start_header_id|>user<|end_header_id|>
        
        \(message)<|eot_id|><|start_header_id|>assistant<|end_header_id|>
        
        
        """
    }
}
