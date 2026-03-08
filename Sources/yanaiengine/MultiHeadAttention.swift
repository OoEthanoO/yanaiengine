import Metal
import Foundation

/// Multi-Head Attention: splits Q, K, V into h heads and runs batched attention.
/// Attention(Q, K, V) = Concat(head_1, ..., head_h) * Wo
/// where head_i = softmax( (Qi * Ki^T) / sqrt(d_k) ) * Vi
public class MultiHeadAttention {
    private let engine: MetalEngine
    public let seqLen: Int
    public let dModel: Int
    public let numHeads: Int
    public let dHead: Int  // dModel / numHeads
    
    // Projection layers: input -> Q, K, V (full dModel width)
    private let queryProj: LinearLayer
    public let keyProj: LinearLayer
    public let valueProj: LinearLayer
    private let outputProj: LinearLayer  // Concat(heads) -> output
    
    // Scratch tensors
    private let qkScores: Tensor         // [seqLen x seqLen] per head (reused)
    private let headOutput: Tensor       // [seqLen x dHead] per head (reused)
    private let concatOutput: Tensor     // [seqLen x dModel] — concatenated heads
    private let keyTransposed: Tensor    // [dHead x seqLen] per head (reused)
    public let output: Tensor            // Final output [seqLen x dModel]
    
    private let useCausalMask: Bool
    
    public init(engine: MetalEngine, seqLen: Int, dModel: Int, numHeads: Int, useCausalMask: Bool = true) {
        precondition(dModel % numHeads == 0, "dModel must be divisible by numHeads")
        self.engine = engine
        self.seqLen = seqLen
        self.dModel = dModel
        self.numHeads = numHeads
        self.dHead = dModel / numHeads
        self.useCausalMask = useCausalMask
        
        // Full-width projections (no ReLU)
        self.queryProj = LinearLayer(engine: engine, inputDim: dModel, outputDim: dModel, batchSize: seqLen, useReLU: false)
        self.keyProj = LinearLayer(engine: engine, inputDim: dModel, outputDim: dModel, batchSize: seqLen, useReLU: false)
        self.valueProj = LinearLayer(engine: engine, inputDim: dModel, outputDim: dModel, batchSize: seqLen, useReLU: false)
        self.outputProj = LinearLayer(engine: engine, inputDim: dModel, outputDim: dModel, batchSize: seqLen, useReLU: false)
        
        // Per-head scratch (reused across heads in serial)
        self.qkScores = Tensor(device: engine.device, rows: seqLen, cols: seqLen)
        self.headOutput = Tensor(device: engine.device, rows: seqLen, cols: dHead)
        self.keyTransposed = Tensor(device: engine.device, rows: dHead, cols: seqLen)
        
        // Concatenation buffer and final output
        self.concatOutput = Tensor(device: engine.device, rows: seqLen, cols: dModel)
        self.output = Tensor(device: engine.device, rows: seqLen, cols: dModel)
    }
    
    /// Forward pass: Multi-Head Scaled Dot-Product Attention.
    public func forward(input: Tensor) {
        // Phase 1: Project input into Q, K, V [seqLen x dModel]
        queryProj.forward(input: input)
        keyProj.forward(input: input)
        valueProj.forward(input: input)
        
        let qPtr = queryProj.output.pointer()
        let kPtr = keyProj.output.pointer()
        let vPtr = valueProj.output.pointer()
        let concatPtr = concatOutput.pointer()
        
        // Phase 2: For each head, extract the slice and run attention
        for h in 0..<numHeads {
            let headOffset = h * dHead
            
            // Extract Q_h, K_h slices into head-local tensors for GEMM
            // Q_h[row][col] = Q[row][headOffset + col]
            let qhBuffer = Tensor(device: engine.device, rows: seqLen, cols: dHead)
            let khBuffer = Tensor(device: engine.device, rows: seqLen, cols: dHead)
            let vhBuffer = Tensor(device: engine.device, rows: seqLen, cols: dHead)
            let qhPtr = qhBuffer.pointer()
            let khPtr = khBuffer.pointer()
            let vhPtr = vhBuffer.pointer()
            
            for row in 0..<seqLen {
                for col in 0..<dHead {
                    qhPtr[row * dHead + col] = qPtr[row * dModel + headOffset + col]
                    khPtr[row * dHead + col] = kPtr[row * dModel + headOffset + col]
                    vhPtr[row * dHead + col] = vPtr[row * dModel + headOffset + col]
                }
            }
            
            // Run attention for this head on GPU
            guard let commandBuffer = engine.commandQueue.makeCommandBuffer(),
                  let encoder = commandBuffer.makeComputeCommandEncoder() else { fatalError() }
            
            // Apply RoPE to Q_h and K_h
            let ropePSO = engine.getPipelineState(name: "rope_kernel")
            var ropeSeq = UInt32(seqLen)
            var ropeDHead = UInt32(dHead)
            let numPairs = dHead / 2
            
            // RoPE on Q_h
            encoder.setComputePipelineState(ropePSO)
            encoder.setBuffer(qhBuffer.buffer, offset: 0, index: 0)
            encoder.setBytes(&ropeSeq, length: MemoryLayout<UInt32>.size, index: 1)
            encoder.setBytes(&ropeDHead, length: MemoryLayout<UInt32>.size, index: 2)
            encoder.dispatchThreads(MTLSize(width: numPairs, height: seqLen, depth: 1),
                                    threadsPerThreadgroup: MTLSize(width: 8, height: 8, depth: 1))
            
            // RoPE on K_h
            encoder.setComputePipelineState(ropePSO)
            encoder.setBuffer(khBuffer.buffer, offset: 0, index: 0)
            encoder.setBytes(&ropeSeq, length: MemoryLayout<UInt32>.size, index: 1)
            encoder.setBytes(&ropeDHead, length: MemoryLayout<UInt32>.size, index: 2)
            encoder.dispatchThreads(MTLSize(width: numPairs, height: seqLen, depth: 1),
                                    threadsPerThreadgroup: MTLSize(width: 8, height: 8, depth: 1))
            
            // Transpose K_h -> [dHead x seqLen]
            let transPSO = engine.getPipelineState(name: "transpose_kernel")
            encoder.setComputePipelineState(transPSO)
            encoder.setBuffer(khBuffer.buffer, offset: 0, index: 0)
            encoder.setBuffer(keyTransposed.buffer, offset: 0, index: 1)
            var tRows = UInt32(seqLen)
            var tCols = UInt32(dHead)
            encoder.setBytes(&tRows, length: MemoryLayout<UInt32>.size, index: 2)
            encoder.setBytes(&tCols, length: MemoryLayout<UInt32>.size, index: 3)
            encoder.dispatchThreads(MTLSize(width: dHead, height: seqLen, depth: 1),
                                    threadsPerThreadgroup: MTLSize(width: 8, height: 8, depth: 1))
            
            // Scores = Q_h * K_h^T: [seqLen x dHead] * [dHead x seqLen] = [seqLen x seqLen]
            let gemmPSO = engine.getPipelineState(name: "gemm_kernel")
            encoder.setComputePipelineState(gemmPSO)
            encoder.setBuffer(qhBuffer.buffer, offset: 0, index: 0)
            encoder.setBuffer(keyTransposed.buffer, offset: 0, index: 1)
            encoder.setBuffer(qkScores.buffer, offset: 0, index: 2)
            var gM = UInt32(seqLen); var gK = UInt32(dHead); var gN = UInt32(seqLen)
            encoder.setBytes(&gM, length: MemoryLayout<UInt32>.size, index: 3)
            encoder.setBytes(&gK, length: MemoryLayout<UInt32>.size, index: 4)
            encoder.setBytes(&gN, length: MemoryLayout<UInt32>.size, index: 5)
            encoder.dispatchThreads(MTLSize(width: seqLen, height: seqLen, depth: 1),
                                    threadsPerThreadgroup: MTLSize(width: 8, height: 8, depth: 1))
            
            // Scale by 1/sqrt(dHead)
            let scalePSO = engine.getPipelineState(name: "scale_kernel")
            encoder.setComputePipelineState(scalePSO)
            encoder.setBuffer(qkScores.buffer, offset: 0, index: 0)
            var sf = 1.0 / sqrt(Float(dHead))
            encoder.setBytes(&sf, length: MemoryLayout<Float>.size, index: 1)
            var sLen = UInt32(seqLen * seqLen)
            encoder.setBytes(&sLen, length: MemoryLayout<UInt32>.size, index: 2)
            encoder.dispatchThreads(MTLSize(width: seqLen * seqLen, height: 1, depth: 1),
                                    threadsPerThreadgroup: MTLSize(width: min(scalePSO.maxTotalThreadsPerThreadgroup, seqLen * seqLen), height: 1, depth: 1))
            
            // Causal mask
            if useCausalMask {
                let maskPSO = engine.getPipelineState(name: "causal_mask_kernel")
                encoder.setComputePipelineState(maskPSO)
                encoder.setBuffer(qkScores.buffer, offset: 0, index: 0)
                var dim = UInt32(seqLen)
                encoder.setBytes(&dim, length: MemoryLayout<UInt32>.size, index: 1)
                encoder.dispatchThreads(MTLSize(width: seqLen, height: seqLen, depth: 1),
                                        threadsPerThreadgroup: MTLSize(width: 8, height: 8, depth: 1))
            }
            
            // Softmax (1D: one thread per row)
            let softmaxPSO = engine.getPipelineState(name: "softmax_kernel")
            encoder.setComputePipelineState(softmaxPSO)
            encoder.setBuffer(qkScores.buffer, offset: 0, index: 0)
            var smR = UInt32(seqLen); var smC = UInt32(seqLen)
            encoder.setBytes(&smR, length: MemoryLayout<UInt32>.size, index: 1)
            encoder.setBytes(&smC, length: MemoryLayout<UInt32>.size, index: 2)
            encoder.dispatchThreads(MTLSize(width: seqLen, height: 1, depth: 1),
                                    threadsPerThreadgroup: MTLSize(width: min(softmaxPSO.maxTotalThreadsPerThreadgroup, seqLen), height: 1, depth: 1))
            
            // Head output = scores * V_h: [seqLen x seqLen] * [seqLen x dHead] = [seqLen x dHead]
            encoder.setComputePipelineState(gemmPSO)
            encoder.setBuffer(qkScores.buffer, offset: 0, index: 0)
            encoder.setBuffer(vhBuffer.buffer, offset: 0, index: 1)
            encoder.setBuffer(headOutput.buffer, offset: 0, index: 2)
            var hM = UInt32(seqLen); var hK = UInt32(seqLen); var hN = UInt32(dHead)
            encoder.setBytes(&hM, length: MemoryLayout<UInt32>.size, index: 3)
            encoder.setBytes(&hK, length: MemoryLayout<UInt32>.size, index: 4)
            encoder.setBytes(&hN, length: MemoryLayout<UInt32>.size, index: 5)
            encoder.dispatchThreads(MTLSize(width: dHead, height: seqLen, depth: 1),
                                    threadsPerThreadgroup: MTLSize(width: 8, height: 8, depth: 1))
            
            encoder.endEncoding()
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
            
            // Copy head output into concat buffer at correct offset
            let hoPtr = headOutput.pointer()
            for row in 0..<seqLen {
                for col in 0..<dHead {
                    concatPtr[row * dModel + headOffset + col] = hoPtr[row * dHead + col]
                }
            }
        }
        
        // Phase 3: Output projection: [seqLen x dModel] -> [seqLen x dModel]
        outputProj.forward(input: concatOutput)
        
        // Copy to output
        let outProjPtr = outputProj.output.pointer()
        let outPtr = output.pointer()
        memcpy(outPtr, outProjPtr, seqLen * dModel * MemoryLayout<Float>.stride)
    }
    
    // ====================================================
    // KV-Cached Single-Token Decode
    // ====================================================
    
    // Decode-phase scratch (allocated lazily)
    private lazy var decodeQProj = LinearLayer(engine: engine, inputDim: dModel, outputDim: dModel, batchSize: 1, useReLU: false)
    private lazy var decodeKProj = LinearLayer(engine: engine, inputDim: dModel, outputDim: dModel, batchSize: 1, useReLU: false)
    private lazy var decodeVProj = LinearLayer(engine: engine, inputDim: dModel, outputDim: dModel, batchSize: 1, useReLU: false)
    private lazy var decodeOutProj = LinearLayer(engine: engine, inputDim: dModel, outputDim: dModel, batchSize: 1, useReLU: false)
    private lazy var decodeOutput = Tensor(device: engine.device, rows: 1, cols: dModel)
    private lazy var decodeConcatOutput = Tensor(device: engine.device, rows: 1, cols: dModel)
    private var decodeWeightsCopied = false
    
    /// Copy weights from prefill projections to decode projections (one-time setup).
    private func ensureDecodeWeights() {
        if decodeWeightsCopied { return }
        // Share the same weight data
        memcpy(decodeQProj.weights.pointer(), queryProj.weights.pointer(),
               queryProj.weights.rows * queryProj.weights.cols * MemoryLayout<Float>.stride)
        memcpy(decodeQProj.bias.pointer(), queryProj.bias.pointer(),
               queryProj.bias.cols * MemoryLayout<Float>.stride)
        memcpy(decodeKProj.weights.pointer(), keyProj.weights.pointer(),
               keyProj.weights.rows * keyProj.weights.cols * MemoryLayout<Float>.stride)
        memcpy(decodeKProj.bias.pointer(), keyProj.bias.pointer(),
               keyProj.bias.cols * MemoryLayout<Float>.stride)
        memcpy(decodeVProj.weights.pointer(), valueProj.weights.pointer(),
               valueProj.weights.rows * valueProj.weights.cols * MemoryLayout<Float>.stride)
        memcpy(decodeVProj.bias.pointer(), valueProj.bias.pointer(),
               valueProj.bias.cols * MemoryLayout<Float>.stride)
        memcpy(decodeOutProj.weights.pointer(), outputProj.weights.pointer(),
               outputProj.weights.rows * outputProj.weights.cols * MemoryLayout<Float>.stride)
        memcpy(decodeOutProj.bias.pointer(), outputProj.bias.pointer(),
               outputProj.bias.cols * MemoryLayout<Float>.stride)
        decodeWeightsCopied = true
    }
    
    /// Forward pass for a single new token using KV cache.
    /// Input: [1 x dModel]. Updates cache and produces output [1 x dModel].
    public func forwardCached(input: Tensor, cache: KVCache) -> Tensor {
        ensureDecodeWeights()
        
        let pos = cache.currentPosition
        
        // Project single token → Q, K, V [1 x dModel]
        decodeQProj.forward(input: input)
        decodeKProj.forward(input: input)
        decodeVProj.forward(input: input)
        
        let qPtr = decodeQProj.output.pointer()
        let concatPtr = decodeConcatOutput.pointer()
        
        // Append new K/V to cache for all heads
        cache.appendFromFull(kTensor: decodeKProj.output, vTensor: decodeVProj.output, tokenIdx: 0, dModel: dModel)
        cache.advancePosition()
        let cacheLen = cache.currentPosition  // now includes the new token
        
        // For each head: Q_h [1 x dHead] × cachedK_h^T [dHead x cacheLen] → scores [1 x cacheLen]
        for h in 0..<numHeads {
            let headOffset = h * dHead
            
            // Extract Q_h from full Q
            var qh = [Float](repeating: 0, count: dHead)
            for d in 0..<dHead { qh[d] = qPtr[headOffset + d] }
            
            // Apply RoPE to Q_h at position `pos`
            for pair in 0..<(dHead / 2) {
                let theta = Float(pos) / pow(10000.0, Float(2 * pair) / Float(dHead))
                let cosT = cos(theta)
                let sinT = sin(theta)
                let x0 = qh[2 * pair]
                let x1 = qh[2 * pair + 1]
                qh[2 * pair] = x0 * cosT - x1 * sinT
                qh[2 * pair + 1] = x0 * sinT + x1 * cosT
            }
            
            // Compute scores: Q_h [1 x dHead] dot cachedK [cacheLen x dHead]
            let cachedKPtr = cache.cachedKeys[h].pointer()
            var scores = [Float](repeating: 0, count: cacheLen)
            let scaleFactor = 1.0 / sqrt(Float(dHead))
            
            for t in 0..<cacheLen {
                var dot: Float = 0
                for d in 0..<dHead {
                    dot += qh[d] * cachedKPtr[t * dHead + d]
                }
                scores[t] = dot * scaleFactor
            }
            
            // Causal mask: position `pos` can see positions 0..pos
            if useCausalMask {
                for t in 0..<cacheLen {
                    if t > pos { scores[t] = -Float.infinity }
                }
            }
            
            // Softmax over scores
            let maxScore = scores.max() ?? 0
            var expSum: Float = 0
            for t in 0..<cacheLen {
                scores[t] = exp(scores[t] - maxScore)
                expSum += scores[t]
            }
            for t in 0..<cacheLen {
                scores[t] /= expSum
            }
            
            // Weighted sum: output_h = scores × cachedV [cacheLen x dHead]
            let cachedVPtr = cache.cachedValues[h].pointer()
            for d in 0..<dHead {
                var val: Float = 0
                for t in 0..<cacheLen {
                    val += scores[t] * cachedVPtr[t * dHead + d]
                }
                concatPtr[headOffset + d] = val
            }
        }
        
        // Output projection [1 x dModel] -> [1 x dModel]
        decodeOutProj.forward(input: decodeConcatOutput)
        
        let outPtr = decodeOutput.pointer()
        memcpy(outPtr, decodeOutProj.output.pointer(), dModel * MemoryLayout<Float>.stride)
        
        return decodeOutput
    }
}
