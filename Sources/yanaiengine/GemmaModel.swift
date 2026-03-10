import Metal
import Foundation

/// Google Gemma 2 Model: Embedding -> N x GemmaBlock -> RMSNorm -> LMHead.
/// Implements alternating Sliding Window Attention (SWA).
public class GemmaModel {
    private let engine: MetalEngine
    public let config: GemmaConfig
    
    public let embedding: EmbeddingLayer
    public let blocks: [GemmaBlock]
    public let caches: [KVCache]
    private let finalNormGamma: Tensor
    public let lmHead: LMHead
    
    public init(engine: MetalEngine, config: GemmaConfig) {
        self.engine = engine
        self.config = config
        
        self.embedding = EmbeddingLayer(
            engine: engine,
            vocabSize: config.vocabSize,
            dModel: config.dModel,
            maxSeqLen: config.maxSeqLen
        )
        
        self.blocks = (0..<config.numLayers).map { i in
            let block = GemmaBlock(
                engine: engine,
                seqLen: config.maxSeqLen,
                dModel: config.dModel,
                numHeads: config.numHeads,
                numKVHeads: config.numKVHeads,
                ffnDim: config.ffnDim
            )
            // Alternating sliding window: Even layers use SWA, Odd layers use Global
            if i % 2 == 0 {
                block.windowSize = config.windowSize
            } else {
                block.windowSize = 0 // Global
            }
            block.logitCap = 50.0 // Constant for Gemma 2 attention
            return block
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
    }
    
    public func prefill(tokenIds: [UInt32]) {
        let seqLen = tokenIds.count
        embedding.forward(tokenIds: tokenIds)
        
        var current = embedding.output
        for i in 0..<config.numLayers {
            blocks[i].forward(input: current)
            current = blocks[i].output
            
            // In a real implementation, we'd update caches here for future decoding
            // For Gemma 2 prefill, the fused kernel handles masking/SWA internally
        }
        
        // Final normalization and LMHead
        dispatchRMSNorm(data: current, gamma: finalNormGamma, seqLen: seqLen)
        lmHead.forward(input: current)
    }
    
    private func dispatchRMSNorm(data: Tensor, gamma: Tensor, seqLen: Int) {
        guard let cb = engine.commandQueue.makeCommandBuffer(),
              let enc = cb.makeComputeCommandEncoder() else { fatalError() }
        let pso = engine.getPipelineState(name: "rmsnorm_kernel")
        enc.setComputePipelineState(pso)
        enc.setBuffer(data.buffer, offset: 0, index: 0)
        enc.setBuffer(gamma.buffer, offset: 0, index: 1)
        var r = UInt32(seqLen); var c = UInt32(config.dModel); var eps: Float = 1e-6
        enc.setBytes(&r, length: MemoryLayout<UInt32>.size, index: 2)
        enc.setBytes(&c, length: MemoryLayout<UInt32>.size, index: 3)
        enc.setBytes(&eps, length: MemoryLayout<Float>.size, index: 4)
        enc.dispatchThreads(MTLSize(width: seqLen, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: min(pso.maxTotalThreadsPerThreadgroup, seqLen), height: 1, depth: 1))
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
    }
}

public struct GemmaConfig {
    public let vocabSize: Int
    public let dModel: Int
    public let numHeads: Int
    public let numKVHeads: Int
    public let numLayers: Int
    public let maxSeqLen: Int
    public let ffnDim: Int
    public let windowSize: Int
    
    public static let gemma2_9B = GemmaConfig(
        vocabSize: 256000,
        dModel: 3584,
        numHeads: 16,
        numKVHeads: 8,
        numLayers: 42,
        maxSeqLen: 8192,
        ffnDim: 14336,
        windowSize: 4096
    )
}
