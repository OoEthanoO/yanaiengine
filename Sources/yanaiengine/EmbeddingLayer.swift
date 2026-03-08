import Metal
import Foundation

/// Embedding Layer: maps integer token IDs to dense vectors.
/// Weight matrix: [vocabSize x dModel]. Lookup by row index.
public class EmbeddingLayer {
    public let weights: Tensor
    public let output: Tensor
    private let engine: MetalEngine
    public let vocabSize: Int
    public let dModel: Int
    private let maxSeqLen: Int
    
    // Token IDs buffer (uint32)
    private let tokenBuffer: MTLBuffer
    
    public init(engine: MetalEngine, vocabSize: Int, dModel: Int, maxSeqLen: Int) {
        self.engine = engine
        self.vocabSize = vocabSize
        self.dModel = dModel
        self.maxSeqLen = maxSeqLen
        
        self.weights = Tensor(device: engine.device, rows: vocabSize, cols: dModel)
        self.output = Tensor(device: engine.device, rows: maxSeqLen, cols: dModel)
        
        // Allocate buffer for token IDs
        self.tokenBuffer = engine.device.makeBuffer(
            length: maxSeqLen * MemoryLayout<UInt32>.stride,
            options: .storageModeShared
        )!
        
        // Initialize with small random weights
        weights.fillRandom()
    }
    
    /// Lookup: for each token ID, copy the corresponding row from weights.
    public func forward(tokenIds: [UInt32]) {
        let seqLen = tokenIds.count
        
        // Copy token IDs to GPU buffer
        let ptr = tokenBuffer.contents().bindMemory(to: UInt32.self, capacity: maxSeqLen)
        for i in 0..<seqLen {
            ptr[i] = tokenIds[i]
        }
        
        guard let cb = engine.commandQueue.makeCommandBuffer(),
              let enc = cb.makeComputeCommandEncoder() else { fatalError() }
        
        let pso = engine.getPipelineState(name: "embedding_lookup_kernel")
        enc.setComputePipelineState(pso)
        enc.setBuffer(tokenBuffer, offset: 0, index: 0)
        enc.setBuffer(weights.buffer, offset: 0, index: 1)
        enc.setBuffer(output.buffer, offset: 0, index: 2)
        var sl = UInt32(seqLen)
        var dm = UInt32(dModel)
        enc.setBytes(&sl, length: MemoryLayout<UInt32>.size, index: 3)
        enc.setBytes(&dm, length: MemoryLayout<UInt32>.size, index: 4)
        enc.dispatchThreads(MTLSize(width: dModel, height: seqLen, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: 8, height: 8, depth: 1))
        
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
    }
}
