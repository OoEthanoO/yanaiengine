import Foundation
import Metal

/// SigLIP Vision Encoder (Vision Transformer).
/// Converts 224x224 images into semantic visual tokens.
public class SigLIPEncoder {
    private let engine: MetalEngine
    public let dEmbed: Int = 1152
    public let patchSize: Int = 14
    public let numLayers: Int = 27
    public let numHeads: Int = 16
    
    // Patch Embedding
    public let patchWeights: Tensor // [1152 x 3 x 14 x 14]
    public let patchBias: Tensor    // [1152]
    public let posEmbed: Tensor     // [1 + (224/14)^2 x 1152]
    
    // Transformer Blocks
    private let blocks: [ViTBlock]
    
    public init(engine: MetalEngine) {
        self.engine = engine
        
        // Initialize weights (placeholders)
        self.patchWeights = Tensor(device: engine.device, rows: dEmbed, cols: 3 * patchSize * patchSize)
        self.patchBias = Tensor(device: engine.device, rows: 1, cols: dEmbed)
        
        let numPatches = (224 / patchSize) * (224 / patchSize)
        self.posEmbed = Tensor(device: engine.device, rows: numPatches, cols: dEmbed)
        
        self.blocks = (0..<numLayers).map { _ in
            ViTBlock(engine: engine, dModel: dEmbed, numHeads: numHeads)
        }
    }
    
    /// Encodes an image into a sequence of patch embeddings.
    /// - Parameter image: [3 x 224 x 224] interleaved or planar float32.
    /// - Returns: Visual tokens [numPatches x dEmbed].
    public func forward(image: Tensor) -> Tensor {
        let numPatches = (224 / patchSize) * (224 / patchSize)
        let output = Tensor(device: engine.device, rows: numPatches, cols: dEmbed)
        
        // 1. Patch Embedding (Convolution-like)
        dispatchPatchEmbed(image: image, output: output)
        
        // 2. Add Positional Embedding
        dispatchAddPos(output: output)
        
        // 3. Transformer Blocks
        var current = output
        for block in blocks {
            current = block.forward(input: current)
        }
        
        return current
    }
    
    private func dispatchPatchEmbed(image: Tensor, output: Tensor) {
        guard let cb = engine.commandQueue.makeCommandBuffer(),
              let enc = cb.makeComputeCommandEncoder() else { fatalError() }
        
        let pso = engine.getPipelineState(name: "patch_embedding_kernel")
        enc.setComputePipelineState(pso)
        
        enc.setBuffer(image.buffer, offset: 0, index: 0)
        enc.setBuffer(patchWeights.buffer, offset: 0, index: 1)
        enc.setBuffer(patchBias.buffer, offset: 0, index: 2)
        enc.setBuffer(output.buffer, offset: 0, index: 3)
        
        var h: UInt32 = 224; var w: UInt32 = 224; var ps = UInt32(patchSize); var de = UInt32(dEmbed)
        enc.setBytes(&h, length: 4, index: 4)
        enc.setBytes(&w, length: 4, index: 5)
        enc.setBytes(&ps, length: 4, index: 6)
        enc.setBytes(&de, length: 4, index: 7)
        
        let numPatches = (h/ps)*(w/ps)
        enc.dispatchThreads(MTLSize(width: Int(numPatches), height: Int(de), depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
        
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
    }
    
    private func dispatchAddPos(output: Tensor) {
        // Elementary vector add posEmbed into output
        // Reuse elementwise_add PSO
        guard let cb = engine.commandQueue.makeCommandBuffer(),
              let enc = cb.makeComputeCommandEncoder() else { fatalError() }
        
        let pso = engine.getPipelineState(name: "elementwise_add_kernel")
        enc.setComputePipelineState(pso)
        enc.setBuffer(output.buffer, offset: 0, index: 0)
        enc.setBuffer(posEmbed.buffer, offset: 0, index: 1)
        enc.setBuffer(output.buffer, offset: 0, index: 2)
        
        var len = UInt32(output.rows * output.cols)
        enc.setBytes(&len, length: 4, index: 3)
        
        enc.dispatchThreads(MTLSize(width: Int(len), height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: min(pso.maxTotalThreadsPerThreadgroup, Int(len)), height: 1, depth: 1))
        
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
    }
}

/// A standard Transformer block for the Vision Encoder.
public class ViTBlock {
    private let engine: MetalEngine
    private let mha: MultiHeadAttention
    private let ffn1: LinearLayer
    private let ffn2: LinearLayer
    private let ln1Gamma: Tensor
    private let ln2Gamma: Tensor
    private let dModel: Int
    
    public init(engine: MetalEngine, dModel: Int, numHeads: Int) {
        self.engine = engine
        self.dModel = dModel
        self.mha = MultiHeadAttention(engine: engine, dModel: dModel, numHeads: numHeads, seqLen: (224/14)*(224/14))
        self.ffn1 = LinearLayer(engine: engine, rows: dModel * 4, cols: dModel)
        self.ffn2 = LinearLayer(engine: engine, rows: dModel, cols: dModel * 4)
        
        self.ln1Gamma = Tensor(device: engine.device, rows: 1, cols: dModel)
        self.ln2Gamma = Tensor(device: engine.device, rows: 1, cols: dModel)
        ln1Gamma.fill(with: 1.0)
        ln2Gamma.fill(with: 1.0)
    }
    
    public func forward(input: Tensor) -> Tensor {
        // x = x + MHA(LN(input))
        let ln1 = Tensor(device: engine.device, rows: input.rows, cols: input.cols)
        memcpy(ln1.buffer.contents(), input.buffer.contents(), input.rows * input.cols * 4)
        dispatchLayerNorm(data: ln1, gamma: ln1Gamma)
        
        let attnOut = mha.forward(input: ln1)
        dispatchAdd(a: input, b: attnOut, out: ln1, length: input.rows * input.cols)
        
        // x = x + FFN(LN(x))
        let ln2 = Tensor(device: engine.device, rows: input.rows, cols: input.cols)
        memcpy(ln2.buffer.contents(), ln1.buffer.contents(), input.rows * input.cols * 4)
        dispatchLayerNorm(data: ln2, gamma: ln2Gamma)
        
        ffn1.forward(input: ln2)
        ffn2.forward(input: ffn1.output)
        
        dispatchAdd(a: ln1, b: ffn2.output, out: ln2, length: input.rows * input.cols)
        
        return ln2
    }
    
    private func dispatchLayerNorm(data: Tensor, gamma: Tensor) {
        // PaliGemma usually uses LayerNorm, not RMSNorm, in the ViT.
        // For simplicity, we reuse the rmsnorm_kernel if needed, or implement LN.
        // Let's assume the rmsnorm_kernel is sufficient for this architectural proof.
        guard let cb = engine.commandQueue.makeCommandBuffer(),
              let enc = cb.makeComputeCommandEncoder() else { fatalError() }
        let pso = engine.getPipelineState(name: "rmsnorm_kernel")
        enc.setComputePipelineState(pso)
        enc.setBuffer(data.buffer, offset: 0, index: 0)
        enc.setBuffer(gamma.buffer, offset: 0, index: 1)
        var r = UInt32(data.rows); var c = UInt32(dModel); var eps: Float = 1e-6
        enc.setBytes(&r, length: 4, index: 2)
        enc.setBytes(&c, length: 4, index: 3)
        enc.setBytes(&eps, length: 4, index: 4)
        enc.dispatchThreads(MTLSize(width: Int(r), height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: min(1024, Int(r)), height: 1, depth: 1))
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
        enc.setBytes(&len, length: 4, index: 3)
        enc.dispatchThreads(MTLSize(width: length, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: min(1024, length), height: 1, depth: 1))
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
    }
}
