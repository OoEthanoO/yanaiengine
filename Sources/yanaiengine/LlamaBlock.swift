import Metal
import Foundation

/// Llama 3-style Transformer Block with:
/// - RMSNorm (replaces LayerNorm)
/// - Grouped Query Attention (GQA) with RoPE
/// - SwiGLU FFN (replaces GELU FFN)
///
/// Architecture (Pre-Norm):
///   x₁ = x + GQA_MHA(RMSNorm(x))
///   x₂ = x₁ + SwiGLU(RMSNorm(x₁))
///
/// SwiGLU FFN: output = down_proj(silu(gate_proj(x)) ⊙ up_proj(x))
public class LlamaBlock {
    private let engine: MetalEngine
    public let seqLen: Int
    public let dModel: Int
    public let numHeads: Int      // Query heads
    public let numKVHeads: Int    // Key/Value heads (GQA)
    public let dHead: Int
    private let ffnDim: Int       // Intermediate FFN dimension
    
    // Projection layers for GQA attention
    private let queryProj: LinearLayer   // [dModel → numHeads * dHead]
    private let keyProj: LinearLayer     // [dModel → numKVHeads * dHead]
    private let valueProj: LinearLayer   // [dModel → numKVHeads * dHead]
    private let outputProj: LinearLayer  // [numHeads * dHead → dModel]
    
    // SwiGLU FFN: gate_proj, up_proj, down_proj
    private let gateProj: LinearLayer    // [dModel → ffnDim]
    private let upProj: LinearLayer      // [dModel → ffnDim]
    private let downProj: LinearLayer    // [ffnDim → dModel]
    
    // RMSNorm weights (learnable γ)
    private let rmsGamma1: Tensor
    private let rmsGamma2: Tensor
    
    // Scratch tensors
    private let rmsOut1: Tensor
    private let residual1: Tensor
    private let rmsOut2: Tensor
    private let gateOut: Tensor          // For SwiGLU gate
    public let output: Tensor
    
    // Per-head scratch for GQA attention
    private let qkScores: Tensor
    private let headOutput: Tensor
    private let keyTransposed: Tensor
    private let concatOutput: Tensor
    
    // Decode-phase projections (batchSize=1)
    private lazy var dQueryProj = LinearLayer(engine: engine, inputDim: dModel, outputDim: numHeads * dHead, batchSize: 1, useReLU: false)
    private lazy var dKeyProj = LinearLayer(engine: engine, inputDim: dModel, outputDim: numKVHeads * dHead, batchSize: 1, useReLU: false)
    private lazy var dValueProj = LinearLayer(engine: engine, inputDim: dModel, outputDim: numKVHeads * dHead, batchSize: 1, useReLU: false)
    private lazy var dOutputProj = LinearLayer(engine: engine, inputDim: numHeads * dHead, outputDim: dModel, batchSize: 1, useReLU: false)
    
    private lazy var dGateProj = LinearLayer(engine: engine, inputDim: dModel, outputDim: ffnDim, batchSize: 1, useReLU: false)
    private lazy var dUpProj = LinearLayer(engine: engine, inputDim: dModel, outputDim: ffnDim, batchSize: 1, useReLU: false)
    private lazy var dDownProj = LinearLayer(engine: engine, inputDim: ffnDim, outputDim: dModel, batchSize: 1, useReLU: false)
    
    private var decodeWeightsSynced = false
    
    public init(engine: MetalEngine, seqLen: Int, dModel: Int, numHeads: Int, numKVHeads: Int, ffnMultiplier: Float = 2.6875) {
        precondition(dModel % numHeads == 0, "dModel must be divisible by numHeads")
        precondition(numHeads % numKVHeads == 0, "numHeads must be divisible by numKVHeads")
        
        self.engine = engine
        self.seqLen = seqLen
        self.dModel = dModel
        self.numHeads = numHeads
        self.numKVHeads = numKVHeads
        self.dHead = dModel / numHeads
        
        // Llama 3 uses ~2.6875x expansion with rounding to 256
        let rawFFN = Int(Float(dModel) * ffnMultiplier)
        self.ffnDim = ((rawFFN + 255) / 256) * 256  // Round up to 256
        
        // GQA projections
        self.queryProj = LinearLayer(engine: engine, inputDim: dModel, outputDim: numHeads * dHead, batchSize: seqLen, useReLU: false)
        self.keyProj = LinearLayer(engine: engine, inputDim: dModel, outputDim: numKVHeads * dHead, batchSize: seqLen, useReLU: false)
        self.valueProj = LinearLayer(engine: engine, inputDim: dModel, outputDim: numKVHeads * dHead, batchSize: seqLen, useReLU: false)
        self.outputProj = LinearLayer(engine: engine, inputDim: numHeads * dHead, outputDim: dModel, batchSize: seqLen, useReLU: false)
        
        // SwiGLU FFN
        self.gateProj = LinearLayer(engine: engine, inputDim: dModel, outputDim: ffnDim, batchSize: seqLen, useReLU: false)
        self.upProj = LinearLayer(engine: engine, inputDim: dModel, outputDim: ffnDim, batchSize: seqLen, useReLU: false)
        self.downProj = LinearLayer(engine: engine, inputDim: ffnDim, outputDim: dModel, batchSize: seqLen, useReLU: false)
        
        // RMSNorm gamma (initialized to 1.0)
        self.rmsGamma1 = Tensor(device: engine.device, rows: 1, cols: dModel)
        self.rmsGamma2 = Tensor(device: engine.device, rows: 1, cols: dModel)
        rmsGamma1.fill(with: 1.0)
        rmsGamma2.fill(with: 1.0)
        
        // Scratch
        self.rmsOut1 = Tensor(device: engine.device, rows: seqLen, cols: dModel)
        self.residual1 = Tensor(device: engine.device, rows: seqLen, cols: dModel)
        self.rmsOut2 = Tensor(device: engine.device, rows: seqLen, cols: dModel)
        self.gateOut = Tensor(device: engine.device, rows: seqLen, cols: ffnDim)
        self.output = Tensor(device: engine.device, rows: seqLen, cols: dModel)
        
        // Per-head scratch
        self.qkScores = Tensor(device: engine.device, rows: seqLen, cols: seqLen)
        self.headOutput = Tensor(device: engine.device, rows: seqLen, cols: dHead)
        self.keyTransposed = Tensor(device: engine.device, rows: dHead, cols: seqLen)
        self.concatOutput = Tensor(device: engine.device, rows: seqLen, cols: dModel)
    }
    
    /// Forward pass through the Llama 3 block.
    public func forward(input: Tensor) {
        let totalLen = seqLen * dModel
        
        // ====================================
        // Sub-block 1: x₁ = x + GQA_MHA(RMSNorm(x))
        // ====================================
        
        memcpy(rmsOut1.buffer.contents(), input.buffer.contents(), totalLen * MemoryLayout<Float>.stride)
        dispatchRMSNorm(data: rmsOut1, gamma: rmsGamma1)
        
        // GQA Attention
        queryProj.forward(input: rmsOut1)
        keyProj.forward(input: rmsOut1)
        valueProj.forward(input: rmsOut1)
        
        // Fused GPU attention pass
        guard let cb = engine.commandQueue.makeCommandBuffer(),
              let enc = cb.makeComputeCommandEncoder() else { fatalError() }
        
        // Prefer paged attention if a cache is provided and it's a PagedKVCache
        // (For prefill, we'll use a temporary PagedKVCache if needed, 
        // but for now let's keep it simple and focus on the decode/inference path integration)
        let fusedPSO = engine.getPipelineState(name: "fused_attention_kernel")
        enc.setComputePipelineState(fusedPSO)
        enc.setBuffer(queryProj.output.buffer, offset: 0, index: 0)
        enc.setBuffer(keyProj.output.buffer, offset: 0, index: 1)
        enc.setBuffer(valueProj.output.buffer, offset: 0, index: 2)
        enc.setBuffer(concatOutput.buffer, offset: 0, index: 3)
        
        var sl = UInt32(seqLen)
        var dh = UInt32(dHead)
        var sc = 1.0 / sqrt(Float(dHead))
        var cm = true // Causal mask always true for prefill in Llama
        var nh = UInt32(numHeads)
        var nkv = UInt32(numKVHeads)
        var cap: Float = 0.0
        var ws: Int32 = 0
        
        enc.setBytes(&sl, length: MemoryLayout<UInt32>.size, index: 4)
        enc.setBytes(&dh, length: MemoryLayout<UInt32>.size, index: 5)
        enc.setBytes(&sc, length: MemoryLayout<Float>.size, index: 6)
        enc.setBytes(&cm, length: MemoryLayout<Bool>.size, index: 7)
        enc.setBytes(&nh, length: MemoryLayout<UInt32>.size, index: 8)
        enc.setBytes(&nkv, length: MemoryLayout<UInt32>.size, index: 9)
        enc.setBytes(&cap, length: MemoryLayout<Float>.size, index: 10)
        enc.setBytes(&ws, length: MemoryLayout<Int32>.size, index: 11)
        
        // Dispatch threads: 
        //   width = seqLen (one thread per token row)
        //   height = 1
        //   depth = numHeads (one thread per head)
        enc.dispatchThreads(MTLSize(width: seqLen, height: 1, depth: numHeads),
                            threadsPerThreadgroup: MTLSize(width: min(fusedPSO.maxTotalThreadsPerThreadgroup, seqLen), height: 1, depth: 1))
        
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        
        // Output projection
        outputProj.forward(input: concatOutput)
        
        // Residual #1
        dispatchAdd(a: input, b: outputProj.output, out: residual1, length: totalLen)
        
        // ====================================
        // Sub-block 2: x₂ = x₁ + SwiGLU(RMSNorm(x₁))
        // ====================================
        
        memcpy(rmsOut2.buffer.contents(), residual1.buffer.contents(), totalLen * MemoryLayout<Float>.stride)
        dispatchRMSNorm(data: rmsOut2, gamma: rmsGamma2)
        
        // SwiGLU: down_proj(silu(gate_proj(x)) ⊙ up_proj(x))
        gateProj.forward(input: rmsOut2)
        upProj.forward(input: rmsOut2)
        
        // Apply SiLU to gate output
        dispatchSiLU(data: gateProj.output, length: seqLen * ffnDim)
        
        // Element-wise multiply: gate * up
        let gatePtr = gateProj.output.pointer()
        let upPtr = upProj.output.pointer()
        for i in 0..<(seqLen * ffnDim) {
            gatePtr[i] *= upPtr[i]
        }
        
        // Down projection
        downProj.forward(input: gateProj.output)
        
        // Residual #2
        dispatchAdd(a: residual1, b: downProj.output, out: output, length: totalLen)
    }
    
    /// Batched forward pass for Continuous Batching.
    public func forwardBatched(input: Tensor, 
                               qStarts: [UInt32], 
                               contextLens: [UInt32], 
                               blockTables: Tensor, 
                               maxBlocksPerSeq: Int,
                               allocator: BlockAllocator,
                               layerIdx: Int) -> Tensor {
        let totalTokens = input.rows
        let totalLen = totalTokens * dModel
        
        // Scratch for batched operations
        let bRmsOut1 = Tensor(device: engine.device, rows: totalTokens, cols: dModel)
        let bConcatOut = Tensor(device: engine.device, rows: totalTokens, cols: dModel)
        let bResidual1 = Tensor(device: engine.device, rows: totalTokens, cols: dModel)
        let bOutput = Tensor(device: engine.device, rows: totalTokens, cols: dModel)
        
        // 1. RMSNorm
        memcpy(bRmsOut1.buffer.contents(), input.buffer.contents(), totalLen * MemoryLayout<Float>.stride)
        dispatchRMSNorm(data: bRmsOut1, gamma: rmsGamma1, rows: totalTokens)
        
        // 2. Projections
        queryProj.forward(input: bRmsOut1)
        keyProj.forward(input: bRmsOut1)
        valueProj.forward(input: bRmsOut1)
        
        // 3. Batched Ragged Attention
        guard let cb = engine.commandQueue.makeCommandBuffer(),
              let enc = cb.makeComputeCommandEncoder() else { fatalError() }
        
        let pso = engine.getPipelineState(name: "batched_paged_attention_kernel")
        enc.setComputePipelineState(pso)
        enc.setBuffer(queryProj.output.buffer, offset: 0, index: 0)
        enc.setBuffer(allocator.keyBuffer, offset: 0, index: 1)
        enc.setBuffer(allocator.valueBuffer, offset: 0, index: 2)
        enc.setBuffer(bConcatOut.buffer, offset: 0, index: 3)
        
        // Metadata
        let qStartsBuf = engine.device.makeBuffer(bytes: qStarts, length: qStarts.count * MemoryLayout<UInt32>.size, options: .storageModeShared)
        let cLensBuf = engine.device.makeBuffer(bytes: contextLens, length: contextLens.count * MemoryLayout<UInt32>.size, options: .storageModeShared)
        
        enc.setBuffer(qStartsBuf, offset: 0, index: 4)
        enc.setBuffer(cLensBuf, offset: 0, index: 5)
        
        var dh = UInt32(dHead)
        var sc = 1.0 / sqrt(Float(dHead))
        var cm = true
        var nh = UInt32(numHeads)
        var nkv = UInt32(numKVHeads)
        var cap: Float = 0.0
        var bs = UInt32(allocator.blockSize)
        var mb = UInt32(maxBlocksPerSeq)
        
        enc.setBytes(&dh, length: MemoryLayout<UInt32>.size, index: 6)
        enc.setBytes(&sc, length: MemoryLayout<Float>.size, index: 7)
        enc.setBytes(&cm, length: MemoryLayout<Bool>.size, index: 8)
        enc.setBytes(&nh, length: MemoryLayout<UInt32>.size, index: 9)
        enc.setBytes(&nkv, length: MemoryLayout<UInt32>.size, index: 10)
        enc.setBytes(&cap, length: MemoryLayout<Float>.size, index: 11)
        enc.setBuffer(blockTables.buffer, offset: 0, index: 12)
        enc.setBytes(&bs, length: MemoryLayout<UInt32>.size, index: 13)
        enc.setBytes(&mb, length: MemoryLayout<UInt32>.size, index: 14)
        
        enc.dispatchThreads(MTLSize(width: totalTokens, height: 1, depth: numHeads),
                            threadsPerThreadgroup: MTLSize(width: min(pso.maxTotalThreadsPerThreadgroup, totalTokens), height: 1, depth: 1))
        
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        
        // 4. Output projection
        outputProj.forward(input: bConcatOut)
        
        // 5. Residual 1
        dispatchAdd(a: input, b: outputProj.output, out: bResidual1, length: totalLen)
        
        // 6. FFN (Sub-block 2)
        let bRmsOut2 = Tensor(device: engine.device, rows: totalTokens, cols: dModel)
        memcpy(bRmsOut2.buffer.contents(), bResidual1.buffer.contents(), totalLen * MemoryLayout<Float>.stride)
        dispatchRMSNorm(data: bRmsOut2, gamma: rmsGamma2, rows: totalTokens)
        
        gateProj.forward(input: bRmsOut2)
        upProj.forward(input: bRmsOut2)
        
        dispatchSiLU(data: gateProj.output, length: totalTokens * ffnDim)
        
        let gPtr = gateProj.output.pointer()
        let uPtr = upProj.output.pointer()
        for i in 0..<(totalTokens * ffnDim) { gPtr[i] *= uPtr[i] }
        
        downProj.forward(input: gateProj.output)
        
        // Residual 2
        dispatchAdd(a: bResidual1, b: downProj.output, out: bOutput, length: totalLen)
        
        return bOutput
    }
    
    /// Synchronize weights from prefill layers to decode layers (sharing memory).
    private func syncDecodeWeights() {
        if decodeWeightsSynced { return }
        let layers: [(LinearLayer, LinearLayer)] = [
            (queryProj, dQueryProj), (keyProj, dKeyProj), (valueProj, dValueProj), (outputProj, dOutputProj),
            (gateProj, dGateProj), (upProj, dUpProj), (downProj, dDownProj)
        ]
        for (src, dst) in layers {
            memcpy(dst.weights.pointer(), src.weights.pointer(), src.weights.rows * src.weights.cols * MemoryLayout<Float>.stride)
            memcpy(dst.bias.pointer(), src.bias.pointer(), src.bias.cols * MemoryLayout<Float>.stride)
        }
        decodeWeightsSynced = true
    }
    
    /// Forward pass for a single token using KV Cache (O(1) relative to total context).
    public func forwardCached(input: Tensor, cache: KVCache) -> Tensor {
        precondition(input.rows == 1, "forwardCached expects a single token")
        syncDecodeWeights()
        
        let pos = cache.currentPosition
        let outSingle = Tensor(device: engine.device, rows: 1, cols: dModel)
        
        // 1. RMSNorm
        let rmsIn = Tensor(device: engine.device, rows: 1, cols: dModel)
        memcpy(rmsIn.pointer(), input.pointer(), dModel * MemoryLayout<Float>.stride)
        dispatchRMSNorm(data: rmsIn, gamma: rmsGamma1, rows: 1)
        
        // 2. Projections
        dQueryProj.forward(input: rmsIn)
        dKeyProj.forward(input: rmsIn)
        dValueProj.forward(input: rmsIn)
        
        // 3. Update Cache
        cache.appendFromFull(kTensor: dKeyProj.output, vTensor: dValueProj.output, tokenIdx: 0, dModel: numKVHeads * dHead)
        cache.advancePosition()
        let cacheLen = cache.currentPosition
        
        // 4. Attention
        let qPtr = dQueryProj.output.pointer()
        let concatPtr = Tensor(device: engine.device, rows: 1, cols: numHeads * dHead).pointer()
        let kvGroupSize = numHeads / numKVHeads
        
        for h in 0..<numHeads {
            let kvH = h / kvGroupSize
            let headOffset = h * dHead
            
            // Extract & RoPE Query
            var qh = [Float](repeating: 0, count: dHead)
            for d in 0..<dHead { qh[d] = qPtr[headOffset + d] }
            
            for pair in 0..<(dHead / 2) {
                let theta = Float(pos) / pow(10000.0, Float(2 * pair) / Float(dHead))
                let cosT = cos(theta); let sinT = sin(theta)
                let x0 = qh[2 * pair]; let x1 = qh[2 * pair + 1]
                qh[2 * pair] = x0 * cosT - x1 * sinT
                qh[2 * pair + 1] = x0 * sinT + x1 * cosT
            }
            
            // Attention scores
            let kPtr = cache.cachedKeys[kvH].pointer()
            var scores = [Float](repeating: 0, count: cacheLen)
            let scale = 1.0 / sqrt(Float(dHead))
            for t in 0..<cacheLen {
                var dot: Float = 0
                for d in 0..<dHead { dot += qh[d] * kPtr[t * dHead + d] }
                scores[t] = dot * scale
            }
            
            // Softmax
            let m = scores.max() ?? 0
            var s: Float = 0
            for t in 0..<cacheLen { scores[t] = exp(scores[t] - m); s += scores[t] }
            for t in 0..<cacheLen { scores[t] /= s }
            
            // Value sum
            let vPtr = cache.cachedValues[kvH].pointer()
            for d in 0..<dHead {
                var sum: Float = 0
                for t in 0..<cacheLen { sum += scores[t] * vPtr[t * dHead + d] }
                concatPtr[headOffset + d] = sum
            }
        }
        
        let concatTensor = Tensor(device: engine.device, rows: 1, cols: numHeads * dHead)
        memcpy(concatTensor.pointer(), concatPtr, numHeads * dHead * MemoryLayout<Float>.stride)
        
        dOutputProj.forward(input: concatTensor)
        
        // Residual 1
        let x1 = Tensor(device: engine.device, rows: 1, cols: dModel)
        let x1Ptr = x1.pointer()
        let inPtr = input.pointer()
        let attPtr = dOutputProj.output.pointer()
        for i in 0..<dModel { x1Ptr[i] = inPtr[i] + attPtr[i] }
        
        // 5. FFN (SwiGLU)
        let rmsIn2 = Tensor(device: engine.device, rows: 1, cols: dModel)
        memcpy(rmsIn2.pointer(), x1.pointer(), dModel * MemoryLayout<Float>.stride)
        dispatchRMSNorm(data: rmsIn2, gamma: rmsGamma2, rows: 1)
        
        dGateProj.forward(input: rmsIn2)
        dUpProj.forward(input: rmsIn2)
        
        dispatchSiLU(data: dGateProj.output, length: ffnDim)
        let gPtr = dGateProj.output.pointer()
        let uPtr = dUpProj.output.pointer()
        for i in 0..<ffnDim { gPtr[i] *= uPtr[i] }
        
        dDownProj.forward(input: dGateProj.output)
        
        // Residual 2
        let resPtr = dDownProj.output.pointer()
        let finalPtr = outSingle.pointer()
        for i in 0..<dModel { finalPtr[i] = x1Ptr[i] + resPtr[i] }
        
        return outSingle
    }
    
    // MARK: - GPU Dispatch Helpers
    
    private func dispatchRMSNorm(data: Tensor, gamma: Tensor, rows: Int? = nil) {
        guard let cb = engine.commandQueue.makeCommandBuffer(),
              let enc = cb.makeComputeCommandEncoder() else { fatalError() }
        let pso = engine.getPipelineState(name: "rmsnorm_kernel")
        enc.setComputePipelineState(pso)
        enc.setBuffer(data.buffer, offset: 0, index: 0)
        enc.setBuffer(gamma.buffer, offset: 0, index: 1)
        var r = UInt32(rows ?? seqLen); var c = UInt32(dModel); var eps: Float = 1e-5
        enc.setBytes(&r, length: MemoryLayout<UInt32>.size, index: 2)
        enc.setBytes(&c, length: MemoryLayout<UInt32>.size, index: 3)
        enc.setBytes(&eps, length: MemoryLayout<Float>.size, index: 4)
        enc.dispatchThreads(MTLSize(width: Int(r), height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: min(pso.maxTotalThreadsPerThreadgroup, Int(r)), height: 1, depth: 1))
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
    }
    
    private func dispatchSiLU(data: Tensor, length: Int) {
        guard let cb = engine.commandQueue.makeCommandBuffer(),
              let enc = cb.makeComputeCommandEncoder() else { fatalError() }
        let pso = engine.getPipelineState(name: "silu_kernel")
        enc.setComputePipelineState(pso)
        enc.setBuffer(data.buffer, offset: 0, index: 0)
        var len = UInt32(length)
        enc.setBytes(&len, length: MemoryLayout<UInt32>.size, index: 1)
        enc.dispatchThreads(MTLSize(width: length, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: min(pso.maxTotalThreadsPerThreadgroup, length), height: 1, depth: 1))
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
    }
    
    private func dispatchAdd(a: Tensor, b: Tensor, out: Tensor, length: Int) {
        guard let cb = engine.commandQueue.makeCommandBuffer(),
              let enc = cb.makeComputeCommandEncoder() else { fatalError() }
        let pso = engine.getPipelineState(name: "elementwise_add_kernel")
        enc.setComputePipelineState(pso)
        enc.setBuffer(a.buffer, offset: 0, index: 0)
        enc.setBuffer(b.buffer, offset: 0, index: 1)
        enc.setBuffer(out.buffer, offset: 0, index: 2)
        var len = UInt32(length)
        enc.setBytes(&len, length: MemoryLayout<UInt32>.size, index: 3)
        enc.dispatchThreads(MTLSize(width: length, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: min(pso.maxTotalThreadsPerThreadgroup, length), height: 1, depth: 1))
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
    }
}
