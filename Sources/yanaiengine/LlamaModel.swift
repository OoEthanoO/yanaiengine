import Metal
import Foundation

/// Full Llama model: Embedding → N × LlamaBlock → RMSNorm → LMHead.
/// Each LlamaBlock has its own KVCache for efficient autoregressive decode.
public class LlamaModel {
    private let engine: MetalEngine
    public let config: LlamaConfig
    
    public let embedding: EmbeddingLayer
    public let blocks: [LlamaBlock]
    public let caches: [KVCache]
    private let finalNormGamma: Tensor
    public let lmHead: LMHead
    
    // Decode-phase LMHead (batchSize=1)
    private let decodeLmHead: LMHead
    
    public init(engine: MetalEngine, config: LlamaConfig) {
        self.engine = engine
        self.config = config
        
        self.embedding = EmbeddingLayer(
            engine: engine,
            vocabSize: config.vocabSize,
            dModel: config.dModel,
            maxSeqLen: config.maxSeqLen
        )
        
        self.blocks = (0..<config.numLayers).map { _ in
            LlamaBlock(
                engine: engine,
                seqLen: config.maxSeqLen,
                dModel: config.dModel,
                numHeads: config.numHeads,
                numKVHeads: config.numKVHeads,
                ffnMultiplier: config.ffnMultiplier
            )
        }
        
        self.caches = (0..<config.numLayers).map { _ in
            KVCache(
                device: engine.device,
                numHeads: config.numKVHeads,
                dHead: config.dModel / config.numHeads,
                maxSeqLen: config.maxSeqLen
            )
        }
        
        self.finalNormGamma = Tensor(device: engine.device, rows: 1, cols: config.dModel)
        finalNormGamma.fill(with: 1.0)
        
        self.lmHead = LMHead(engine: engine, dModel: config.dModel, vocabSize: config.vocabSize, maxSeqLen: config.maxSeqLen)
        self.decodeLmHead = LMHead(engine: engine, dModel: config.dModel, vocabSize: config.vocabSize, maxSeqLen: 1)
    }
    
    /// Prefill: process entire prompt in parallel.
    /// Returns logits for the last token position.
    public func prefill(tokenIds: [UInt32]) -> UnsafeMutablePointer<Float> {
        let seqLen = tokenIds.count
        
        // Embed
        embedding.forward(tokenIds: tokenIds)
        
        // Pad to maxSeqLen
        let hidden = Tensor(device: engine.device, rows: config.maxSeqLen, cols: config.dModel)
        memcpy(hidden.pointer(), embedding.output.pointer(), seqLen * config.dModel * MemoryLayout<Float>.stride)
        
        // Pass through all blocks
        var current = hidden
        for i in 0..<config.numLayers {
            blocks[i].forward(input: current)
            current = blocks[i].output
        }
        
        // Final RMSNorm (CPU for simplicity)
        let outPtr = current.pointer()
        for row in 0..<seqLen {
            let offset = row * config.dModel
            var sumSq: Float = 0
            for c in 0..<config.dModel { sumSq += outPtr[offset + c] * outPtr[offset + c] }
            let rms = 1.0 / sqrt(sumSq / Float(config.dModel) + 1e-5)
            let gPtr = finalNormGamma.pointer()
            for c in 0..<config.dModel { outPtr[offset + c] *= rms * gPtr[c] }
        }
        
        // LMHead
        lmHead.forward(input: current)
        
        // Return pointer to last token's logits
        let logitsPtr = lmHead.logits.pointer()
        let lastRowStart = (seqLen - 1) * config.vocabSize
        return logitsPtr + lastRowStart
    }
    
    /// Decode: process a single new token with KV cache.
    /// Returns pointer to logits [vocabSize].
    public func decode(tokenId: UInt32) -> UnsafeMutablePointer<Float> {
        // [1 x dModel]
        embedding.forwardDecode(tokenId: tokenId)
        var current = embedding.decodeOutput
        
        for i in 0..<config.numLayers {
            current = blocks[i].forwardCached(input: current, cache: caches[i])
        }
        
        // Final RMSNorm (CPU for simplicity)
        let outPtr = current.pointer()
        var sumSq: Float = 0
        for c in 0..<config.dModel { sumSq += outPtr[c] * outPtr[c] }
        let rms = 1.0 / sqrt(sumSq / Float(config.dModel) + 1e-5)
        let gPtr = finalNormGamma.pointer()
        for c in 0..<config.dModel { outPtr[c] *= rms * gPtr[c] }
        
        // LMHead (batchSize=1)
        decodeLmHead.forward(input: current)
        
        return decodeLmHead.logits.pointer()
    }
    
    /// Reset all KV caches for a new conversation.
    public func resetCaches() {
        for cache in caches { cache.reset() }
    }
}

/// Configuration for the Llama model.
public struct LlamaConfig: Sendable {
    public let vocabSize: Int
    public let dModel: Int
    public let numHeads: Int
    public let numKVHeads: Int
    public let numLayers: Int
    public let maxSeqLen: Int
    public let ffnMultiplier: Float
    
    /// Llama 3 8B default config
    public static let llama3_8B = LlamaConfig(
        vocabSize: 128256,
        dModel: 4096,
        numHeads: 32,
        numKVHeads: 8,
        numLayers: 32,
        maxSeqLen: 8192,
        ffnMultiplier: 2.6875
    )
    
    /// Tiny config for testing
    public static let tiny = LlamaConfig(
        vocabSize: 32,
        dModel: 16,
        numHeads: 4,
        numKVHeads: 2,
        numLayers: 2,
        maxSeqLen: 16,
        ffnMultiplier: 2.6875
    )
    
    public init(vocabSize: Int, dModel: Int, numHeads: Int, numKVHeads: Int, numLayers: Int, maxSeqLen: Int, ffnMultiplier: Float) {
        self.vocabSize = vocabSize
        self.dModel = dModel
        self.numHeads = numHeads
        self.numKVHeads = numKVHeads
        self.numLayers = numLayers
        self.maxSeqLen = maxSeqLen
        self.ffnMultiplier = ffnMultiplier
    }
}
