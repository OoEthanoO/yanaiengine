import Metal
import Foundation

/// LM Head: projects the final hidden state back to vocabulary logits.
/// Linear layer: [dModel → vocabSize], plus CPU-side argmax for greedy decoding.
public class LMHead {
    private let projection: LinearLayer
    private let engine: MetalEngine
    public let vocabSize: Int
    public let dModel: Int
    
    public init(engine: MetalEngine, dModel: Int, vocabSize: Int, maxSeqLen: Int) {
        self.engine = engine
        self.vocabSize = vocabSize
        self.dModel = dModel
        self.projection = LinearLayer(engine: engine, inputDim: dModel, outputDim: vocabSize, batchSize: maxSeqLen, useReLU: false)
    }
    
    /// Forward: project hidden states to logits [seqLen x vocabSize]
    public func forward(input: Tensor) {
        projection.forward(input: input)
        applySoftCapping()
    }
    
    /// Gemma 2: Apply logit soft-capping to the final vocab logits (cap=30.0)
    private func applySoftCapping() {
        guard let cb = engine.commandQueue.makeCommandBuffer(),
              let enc = cb.makeComputeCommandEncoder() else { fatalError() }
              
        let pso = engine.getPipelineState(name: "logit_softcap_kernel")
        enc.setComputePipelineState(pso)
        enc.setBuffer(projection.output.buffer, offset: 0, index: 0)
        
        var cap: Float = 30.0
        var len = UInt32(projection.output.rows * projection.output.cols)
        
        enc.setBytes(&cap, length: MemoryLayout<Float>.size, index: 1)
        enc.setBytes(&len, length: MemoryLayout<UInt32>.size, index: 2)
        
        enc.dispatchThreads(MTLSize(width: Int(len), height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: min(pso.maxTotalThreadsPerThreadgroup, Int(len)), height: 1, depth: 1))
                            
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
    }
    
    /// Get the logits tensor (output of the projection)
    public var logits: Tensor {
        return projection.output
    }
    
    /// Greedy decode: return the argmax token ID for the last position in the sequence.
    public func argmaxLastToken(seqLen: Int) -> UInt32 {
        let ptr = projection.output.pointer()
        let lastRowStart = (seqLen - 1) * vocabSize
        
        var maxVal: Float = -Float.infinity
        var maxIdx: UInt32 = 0
        
        for i in 0..<vocabSize {
            let val = ptr[lastRowStart + i]
            if val > maxVal {
                maxVal = val
                maxIdx = UInt32(i)
            }
        }
    
        return maxIdx
    }
}
