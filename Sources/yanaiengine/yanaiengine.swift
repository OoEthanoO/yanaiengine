import Foundation
import Metal

@main
struct yanaiengine {
    static func main() {
        print("==============================================")
        print("  YanAIEngine: Goal #8 — LLM Generation")
        print("  RoPE + Embedding + TransformerBlock + LMHead")
        print("==============================================\n")
        
        guard let engine = MetalEngine() else { fatalError("Metal not available") }
        
        // Load all kernels
        let kernels = [
            "gemm_kernel", "bias_add_kernel", "relu_kernel",
            "transpose_kernel", "mse_derivative_kernel", "sgd_update_kernel",
            "relu_derivative_kernel", "sum_rows_kernel",
            "softmax_kernel", "scale_kernel", "causal_mask_kernel",
            "gelu_kernel", "layernorm_kernel", "elementwise_add_kernel",
            // New for Goal #8:
            "rope_kernel", "embedding_lookup_kernel"
        ]
        for kernel in kernels {
            engine.loadLibrary(resourceName: "gemm", kernelName: kernel)
        }
        
        // ---- Vocabulary ----
        let vocab: [String] = [
            "the",      // 0
            "AI",       // 1
            "engine",   // 2
            "runs",     // 3
            "on",       // 4
            "silicon",  // 5
            "metal",    // 6
            "gpu",      // 7
            "fast",     // 8
            "now"       // 9
        ]
        let vocabSize = vocab.count
        
        // ---- Configuration ----
        let maxSeqLen = 8    // Max tokens in generated sequence
        let dModel = 8       // Embedding dimension
        let numHeads = 2     // Attention heads (dHead = 4)
        
        print("Configuration:")
        print("  Vocabulary:    \(vocab)")
        print("  Vocab Size:    \(vocabSize)")
        print("  Max Seq Len:   \(maxSeqLen)")
        print("  dModel:        \(dModel)")
        print("  Heads:         \(numHeads) (dHead = \(dModel / numHeads))")
        print("  Positional:    RoPE (Rotary)")
        print("  Decoding:      Greedy (argmax)\n")
        
        // ---- Build the LLM Pipeline ----
        let embedding = EmbeddingLayer(engine: engine, vocabSize: vocabSize, dModel: dModel, maxSeqLen: maxSeqLen)
        let transformer = TransformerBlock(engine: engine, seqLen: maxSeqLen, dModel: dModel, numHeads: numHeads)
        let lmHead = LMHead(engine: engine, dModel: dModel, vocabSize: vocabSize, maxSeqLen: maxSeqLen)
        
        print("LLM Pipeline: Embedding → TransformerBlock(RoPE) → LMHead → argmax\n")
        
        // ---- Autoregressive Generation ----
        let startToken: UInt32 = 0  // "the"
        var sequence: [UInt32] = [startToken]
        let tokensToGenerate = maxSeqLen - 1  // Generate 7 more tokens
        
        print("Starting generation from: \"\(vocab[Int(startToken)])\"")
        print("Generating \(tokensToGenerate) tokens...\n")
        
        for step in 0..<tokensToGenerate {
            let currentLen = sequence.count
            
            // 1. Embed: token IDs → dense vectors [currentLen x dModel]
            embedding.forward(tokenIds: sequence)
            
            // 2. Create a properly-sized input for the transformer
            //    Copy only the current sequence into a [maxSeqLen x dModel] tensor
            //    (pad remaining positions with zeros)
            let transformerInput = Tensor(device: engine.device, rows: maxSeqLen, cols: dModel)
            let srcPtr = embedding.output.pointer()
            let dstPtr = transformerInput.pointer()
            for i in 0..<(currentLen * dModel) {
                dstPtr[i] = srcPtr[i]
            }
            
            // 3. Transformer Block (with RoPE inside MHA)
            transformer.forward(input: transformerInput)
            
            // 4. LM Head: project to vocab logits
            lmHead.forward(input: transformer.output)
            
            // 5. Greedy decode: argmax of last token's logits
            let nextToken = lmHead.argmaxLastToken(seqLen: currentLen)
            sequence.append(nextToken)
            
            // Print progress
            let tokenStr = vocab[Int(nextToken)]
            let seqStr = sequence.map { vocab[Int($0)] }.joined(separator: " ")
            print("  Step \(step + 1): predicted \"\(tokenStr)\" → [\(seqStr)]")
        }
        
        // ---- Final Output ----
        let finalText = sequence.map { vocab[Int($0)] }.joined(separator: " ")
        
        print("\n==============================================")
        print("  Generated Text")
        print("==============================================")
        print("  \"\(finalText)\"")
        print("==============================================\n")
        
        // ---- Verification ----
        let allValid = sequence.allSatisfy { $0 < UInt32(vocabSize) }
        let correctLength = sequence.count == maxSeqLen
        let isDeterministic: Bool = {
            // Run a second pass with the same start token to verify determinism
            var seq2: [UInt32] = [startToken]
            for _ in 0..<tokensToGenerate {
                embedding.forward(tokenIds: seq2)
                let tInput = Tensor(device: engine.device, rows: maxSeqLen, cols: dModel)
                let s = embedding.output.pointer()
                let d = tInput.pointer()
                for i in 0..<(seq2.count * dModel) { d[i] = s[i] }
                transformer.forward(input: tInput)
                lmHead.forward(input: transformer.output)
                seq2.append(lmHead.argmaxLastToken(seqLen: seq2.count))
            }
            return seq2 == sequence
        }()
        
        print("  Verification Results")
        print("==============================================")
        print("  All tokens valid IDs:     \(allValid ? "✅ PASS" : "❌ FAIL")")
        print("  Sequence length correct:  \(correctLength ? "✅ PASS" : "❌ FAIL")  [\(sequence.count)/\(maxSeqLen)]")
        print("  Deterministic output:     \(isDeterministic ? "✅ PASS" : "❌ FAIL")")
        print("==============================================")
        
        if allValid && correctLength && isDeterministic {
            print("\n🚀 Goal #8 COMPLETE: Autoregressive LLM generation on Apple Silicon!")
            print("   Pipeline: Embedding → RoPE → MHA → LayerNorm → FFN(GELU) → LMHead → argmax")
            print("   Your engine can now generate text, one token at a time.")
        }
    }
}
