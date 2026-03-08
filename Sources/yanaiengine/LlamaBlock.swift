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
        
        let qPtr = queryProj.output.pointer()
        let kPtr = keyProj.output.pointer()
        let vPtr = valueProj.output.pointer()
        let concatPtr = concatOutput.pointer()
        
        let kvGroupSize = numHeads / numKVHeads  // Queries per KV head
        
        for h in 0..<numHeads {
            let kvHead = h / kvGroupSize  // Which KV head this Q head uses
            let qOffset = h * dHead
            let kvOffset = kvHead * dHead
            let kvDim = numKVHeads * dHead
            
            // Extract Q_h, K_h, V_h slices and apply RoPE
            let qhBuffer = Tensor(device: engine.device, rows: seqLen, cols: dHead)
            let khBuffer = Tensor(device: engine.device, rows: seqLen, cols: dHead)
            let vhBuffer = Tensor(device: engine.device, rows: seqLen, cols: dHead)
            let qhPtr = qhBuffer.pointer()
            let khPtr = khBuffer.pointer()
            let vhPtr = vhBuffer.pointer()
            
            for row in 0..<seqLen {
                for col in 0..<dHead {
                    qhPtr[row * dHead + col] = qPtr[row * (numHeads * dHead) + qOffset + col]
                    khPtr[row * dHead + col] = kPtr[row * kvDim + kvOffset + col]
                    vhPtr[row * dHead + col] = vPtr[row * kvDim + kvOffset + col]
                }
            }
            
            // GPU attention pass for this head
            guard let cb = engine.commandQueue.makeCommandBuffer(),
                  let enc = cb.makeComputeCommandEncoder() else { fatalError() }
            
            // RoPE on Q_h and K_h
            let ropePSO = engine.getPipelineState(name: "rope_kernel")
            var ropeSeq = UInt32(seqLen)
            var ropeDHead = UInt32(dHead)
            let numPairs = dHead / 2
            
            enc.setComputePipelineState(ropePSO)
            enc.setBuffer(qhBuffer.buffer, offset: 0, index: 0)
            enc.setBytes(&ropeSeq, length: MemoryLayout<UInt32>.size, index: 1)
            enc.setBytes(&ropeDHead, length: MemoryLayout<UInt32>.size, index: 2)
            enc.dispatchThreads(MTLSize(width: numPairs, height: seqLen, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: 8, height: 8, depth: 1))
            
            enc.setBuffer(khBuffer.buffer, offset: 0, index: 0)
            enc.dispatchThreads(MTLSize(width: numPairs, height: seqLen, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: 8, height: 8, depth: 1))
            
            // Transpose K_h → [dHead x seqLen]
            let transPSO = engine.getPipelineState(name: "transpose_kernel")
            enc.setComputePipelineState(transPSO)
            enc.setBuffer(khBuffer.buffer, offset: 0, index: 0)
            enc.setBuffer(keyTransposed.buffer, offset: 0, index: 1)
            var tR = UInt32(seqLen); var tC = UInt32(dHead)
            enc.setBytes(&tR, length: MemoryLayout<UInt32>.size, index: 2)
            enc.setBytes(&tC, length: MemoryLayout<UInt32>.size, index: 3)
            enc.dispatchThreads(MTLSize(width: dHead, height: seqLen, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: 8, height: 8, depth: 1))
            
            // Scores = Q_h × K_h^T [seqLen × seqLen]
            let gemmPSO = engine.getPipelineState(name: "gemm_kernel")
            enc.setComputePipelineState(gemmPSO)
            enc.setBuffer(qhBuffer.buffer, offset: 0, index: 0)
            enc.setBuffer(keyTransposed.buffer, offset: 0, index: 1)
            enc.setBuffer(qkScores.buffer, offset: 0, index: 2)
            var gM = UInt32(seqLen); var gK = UInt32(dHead); var gN = UInt32(seqLen)
            enc.setBytes(&gM, length: MemoryLayout<UInt32>.size, index: 3)
            enc.setBytes(&gK, length: MemoryLayout<UInt32>.size, index: 4)
            enc.setBytes(&gN, length: MemoryLayout<UInt32>.size, index: 5)
            enc.dispatchThreads(MTLSize(width: seqLen, height: seqLen, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: 8, height: 8, depth: 1))
            
            // Scale
            let scalePSO = engine.getPipelineState(name: "scale_kernel")
            enc.setComputePipelineState(scalePSO)
            enc.setBuffer(qkScores.buffer, offset: 0, index: 0)
            var scaleLen = UInt32(seqLen * seqLen)
            var scaleVal = 1.0 / sqrt(Float(dHead))
            enc.setBytes(&scaleLen, length: MemoryLayout<UInt32>.size, index: 1)
            enc.setBytes(&scaleVal, length: MemoryLayout<Float>.size, index: 2)
            enc.dispatchThreads(MTLSize(width: seqLen * seqLen, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: min(scalePSO.maxTotalThreadsPerThreadgroup, seqLen * seqLen), height: 1, depth: 1))
            
            // Causal mask
            let maskPSO = engine.getPipelineState(name: "causal_mask_kernel")
            enc.setComputePipelineState(maskPSO)
            enc.setBuffer(qkScores.buffer, offset: 0, index: 0)
            var maskN = UInt32(seqLen)
            enc.setBytes(&maskN, length: MemoryLayout<UInt32>.size, index: 1)
            enc.dispatchThreads(MTLSize(width: seqLen, height: seqLen, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: 8, height: 8, depth: 1))
            
            // Softmax
            let softPSO = engine.getPipelineState(name: "softmax_kernel")
            enc.setComputePipelineState(softPSO)
            enc.setBuffer(qkScores.buffer, offset: 0, index: 0)
            var sR = UInt32(seqLen); var sC = UInt32(seqLen)
            enc.setBytes(&sR, length: MemoryLayout<UInt32>.size, index: 1)
            enc.setBytes(&sC, length: MemoryLayout<UInt32>.size, index: 2)
            enc.dispatchThreads(MTLSize(width: seqLen, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: min(softPSO.maxTotalThreadsPerThreadgroup, seqLen), height: 1, depth: 1))
            
            // Head output = softmax(scores) × V_h [seqLen × dHead]
            enc.setComputePipelineState(gemmPSO)
            enc.setBuffer(qkScores.buffer, offset: 0, index: 0)
            enc.setBuffer(vhBuffer.buffer, offset: 0, index: 1)
            enc.setBuffer(headOutput.buffer, offset: 0, index: 2)
            var hM = UInt32(seqLen); var hK = UInt32(seqLen); var hN = UInt32(dHead)
            enc.setBytes(&hM, length: MemoryLayout<UInt32>.size, index: 3)
            enc.setBytes(&hK, length: MemoryLayout<UInt32>.size, index: 4)
            enc.setBytes(&hN, length: MemoryLayout<UInt32>.size, index: 5)
            enc.dispatchThreads(MTLSize(width: dHead, height: seqLen, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: 8, height: 8, depth: 1))
            
            enc.endEncoding()
            cb.commit()
            cb.waitUntilCompleted()
            
            // Copy head output into concat buffer
            let hoPtr = headOutput.pointer()
            for row in 0..<seqLen {
                for col in 0..<dHead {
                    concatPtr[row * (numHeads * dHead) + qOffset + col] = hoPtr[row * dHead + col]
                }
            }
        }
        
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
