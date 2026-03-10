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
    public let allocator: BlockAllocator
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
        
        self.allocator = BlockAllocator(
            device: engine.device,
            numBlocks: 1024,
            blockSize: 16,
            numKVHeads: config.numKVHeads,
            dHead: config.dModel / config.numHeads
        )
        
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
    
    /// Performs one forward step for a batch of sequences (Continuous Batching).
    /// Returns an array of pointers to the logits of the last token for each sequence.
    public func forwardStep(batch: [SequenceRequest], allocator: BlockAllocator) -> [UnsafeMutablePointer<Float>] {
        let batchSize = batch.count
        var totalTokens = 0
        var qStarts: [UInt32] = [0]
        var contextLens: [UInt32] = []
        var allInputTokens: [UInt32] = []
        
        // 1. Prepare Metadata & Concatenate Inputs
        var maxBlocks = 0
        for req in batch {
            let nextTokens = req.getNextInput()
            allInputTokens.append(contentsOf: nextTokens)
            totalTokens += nextTokens.count
            qStarts.append(UInt32(totalTokens))
            
            let newLen = UInt32(req.totalTokens + nextTokens.count) // total length after this step
            contextLens.append(newLen)
            maxBlocks = max(maxBlocks, req.pagedKVCache[0].pageTable.count + 1) // +1 for safety
        }
        
        // 2. Build Block Tables tensor [batchSize x maxBlocks]
        let blockTablesTensor = Tensor(device: engine.device, rows: batchSize, cols: maxBlocks)
        let btPtr = blockTablesTensor.buffer.contents().bindMemory(to: Int32.self, capacity: batchSize * maxBlocks)
        for i in 0..<batchSize {
            let table = batch[i].pagedKVCache[0].pageTable // Blocks are shared across layers for same seq
            for j in 0..<table.count {
                btPtr[i * maxBlocks + j] = Int32(table[j])
            }
        }
        
        // 3. Embed all tokens at once
        embedding.forward(tokenIds: allInputTokens)
        var current = embedding.output // [totalTokens x dModel]
        
        // 4. Update Physical KV Pools (KV Cache Append)
        // We handle this layer-by-layer during the forward pass by passing the batch info
        // But for simplicity in this implementation, let's assume the Attention kernel
        // will read from the pool. We still need to WRITE the new tokens to the pool first.
        // We'll use a specialized kernel later, or just CPU-side copy for now.
        for layerIdx in 0..<config.numLayers {
            // Write new K/V tokens for this layer to the physical pool
            let kLayer = blocks[layerIdx].keyProj.output
            let vLayer = blocks[layerIdx].valueProj.output
            
            for i in 0..<batchSize {
                let req = batch[i]
                let numNew = Int(qStarts[i+1] - qStarts[i])
                let globalStart = Int(qStarts[i])
                
                for t in 0..<numNew {
                    req.pagedKVCache[layerIdx].appendFromFull(
                        kTensor: kLayer,
                        vTensor: vLayer,
                        tokenIdx: globalStart + t
                    )
                }
            }
            
            // Pass through LlamaBlock
            current = blocks[layerIdx].forwardBatched(
                input: current,
                qStarts: qStarts,
                contextLens: contextLens,
                blockTables: blockTablesTensor,
                maxBlocksPerSeq: maxBlocks,
                allocator: allocator,
                layerIdx: layerIdx
            )
        }
        
        // 5. Final Normalization (Batched)
        dispatchRMSNorm(data: current, gamma: finalNormGamma, rows: totalTokens)
        
        // 6. LMHead (Batched)
        lmHead.forward(input: current)
        let allLogits = lmHead.logits.pointer()
        
        // 7. Extract last-token logits for each request
        var resultPointers: [UnsafeMutablePointer<Float>] = []
        for i in 0..<batchSize {
            let lastTokenGlobalIdx = Int(qStarts[i + 1]) - 1
            resultPointers.append(allLogits + (lastTokenGlobalIdx * config.vocabSize))
        }
        
        return resultPointers
    }
    
    private func dispatchRMSNorm(data: Tensor, gamma: Tensor, rows: Int) {
        guard let cb = engine.commandQueue.makeCommandBuffer(),
              let enc = cb.makeComputeCommandEncoder() else { fatalError() }
        let pso = engine.getPipelineState(name: "rmsnorm_kernel")
        enc.setComputePipelineState(pso)
        enc.setBuffer(data.buffer, offset: 0, index: 0)
        enc.setBuffer(gamma.buffer, offset: 0, index: 1)
        var r = UInt32(rows); var c = UInt32(config.dModel); var eps: Float = 1e-5
        enc.setBytes(&r, length: MemoryLayout<UInt32>.size, index: 2)
        enc.setBytes(&c, length: MemoryLayout<UInt32>.size, index: 3)
        enc.setBytes(&eps, length: MemoryLayout<Float>.size, index: 4)
        enc.dispatchThreads(MTLSize(width: rows, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: min(pso.maxTotalThreadsPerThreadgroup, rows), height: 1, depth: 1))
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
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
