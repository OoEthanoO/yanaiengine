import Metal
import Foundation

/// Quantized Linear Layer: Y = X * dequantize(W_int8) + b
/// Weights are stored as INT8 with per-row FP32 scale factors.
/// Memory bandwidth consumed is ~4x lower than FP32 LinearLayer.
public class QuantizedLinearLayer {
    private let engine: MetalEngine
    public let inputDim: Int
    public let outputDim: Int
    public let batchSize: Int
    
    /// INT8 quantized weights [inputDim x outputDim]
    public let quantizedWeights: QuantizedTensor
    
    /// FP32 bias [1 x outputDim]
    public let bias: Tensor
    
    /// FP32 output [batchSize x outputDim]
    public let output: Tensor
    
    /// Original FP32 memory that would be needed
    public var fp32WeightBytes: Int { return quantizedWeights.fp32Bytes }
    
    /// Actual INT8 memory used
    public var int8WeightBytes: Int { return quantizedWeights.quantizedBytes }
    
    public init(engine: MetalEngine, from layer: LinearLayer) {
        self.engine = engine
        self.inputDim = layer.weights.rows
        self.outputDim = layer.weights.cols
        self.batchSize = layer.output.rows
        
        // Quantize the FP32 weights → INT8
        self.quantizedWeights = QuantizedTensor(device: engine.device, rows: inputDim, cols: outputDim)
        quantizedWeights.quantize(from: layer.weights)
        
        // Copy bias as-is (FP32)
        self.bias = Tensor(device: engine.device, rows: 1, cols: outputDim)
        memcpy(bias.pointer(), layer.bias.pointer(), outputDim * MemoryLayout<Float>.stride)
        
        // Allocate output buffer
        self.output = Tensor(device: engine.device, rows: batchSize, cols: outputDim)
    }
    
    /// Forward pass using INT8 dequantize-on-the-fly GEMM.
    public func forward(input: Tensor) {
        guard let cb = engine.commandQueue.makeCommandBuffer(),
              let enc = cb.makeComputeCommandEncoder() else { fatalError() }
        
        // Q8 GEMM: output = input * dequantize(weights)
        let gemmPSO = engine.getPipelineState(name: "q8_gemm_kernel")
        enc.setComputePipelineState(gemmPSO)
        enc.setBuffer(input.buffer, offset: 0, index: 0)
        enc.setBuffer(quantizedWeights.buffer, offset: 0, index: 1)
        enc.setBuffer(quantizedWeights.scales, offset: 0, index: 2)
        enc.setBuffer(output.buffer, offset: 0, index: 3)
        var m = UInt32(batchSize)
        var k = UInt32(inputDim)
        var n = UInt32(outputDim)
        enc.setBytes(&m, length: MemoryLayout<UInt32>.size, index: 4)
        enc.setBytes(&k, length: MemoryLayout<UInt32>.size, index: 5)
        enc.setBytes(&n, length: MemoryLayout<UInt32>.size, index: 6)
        enc.dispatchThreads(MTLSize(width: outputDim, height: batchSize, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: 8, height: 8, depth: 1))
        
        // Bias add
        let biasPSO = engine.getPipelineState(name: "bias_add_kernel")
        enc.setComputePipelineState(biasPSO)
        enc.setBuffer(output.buffer, offset: 0, index: 0)
        enc.setBuffer(bias.buffer, offset: 0, index: 1)
        var biasRows = UInt32(batchSize)
        var biasCols = UInt32(outputDim)
        enc.setBytes(&biasRows, length: MemoryLayout<UInt32>.size, index: 2)
        enc.setBytes(&biasCols, length: MemoryLayout<UInt32>.size, index: 3)
        enc.dispatchThreads(MTLSize(width: outputDim, height: batchSize, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: 8, height: 8, depth: 1))
        
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
    }
}
