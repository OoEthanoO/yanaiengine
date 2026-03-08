import Metal
import Foundation

/// Quantized Tensor: stores weights as INT8 with per-row FP32 scale factors.
/// Memory usage: rows * cols bytes (INT8) + rows * 4 bytes (scales) vs rows * cols * 4 bytes (FP32).
/// Compression ratio: ~4x for large matrices.
public class QuantizedTensor {
    public let rows: Int
    public let cols: Int
    
    /// INT8 weight buffer [rows x cols] — 1 byte per element
    public let buffer: MTLBuffer
    
    /// FP32 scale factors [rows] — one per row
    public let scales: MTLBuffer
    
    /// Memory used by quantized representation (bytes)
    public var quantizedBytes: Int {
        return rows * cols + rows * MemoryLayout<Float>.stride
    }
    
    /// Memory that would be used by FP32 (bytes)
    public var fp32Bytes: Int {
        return rows * cols * MemoryLayout<Float>.stride
    }
    
    public init(device: MTLDevice, rows: Int, cols: Int) {
        self.rows = rows
        self.cols = cols
        self.buffer = device.makeBuffer(length: rows * cols, options: .storageModeShared)!
        self.scales = device.makeBuffer(length: rows * MemoryLayout<Float>.stride, options: .storageModeShared)!
    }
    
    /// Quantize a FP32 Tensor into this INT8 representation.
    /// Per-row symmetric quantization: scale = max(|row|) / 127
    public func quantize(from source: Tensor) {
        precondition(source.rows == rows && source.cols == cols, "Shape mismatch")
        
        let srcPtr = source.pointer()
        let dstPtr = buffer.contents().bindMemory(to: Int8.self, capacity: rows * cols)
        let scalePtr = scales.contents().bindMemory(to: Float.self, capacity: rows)
        
        for r in 0..<rows {
            // Find max absolute value in this row
            var maxAbs: Float = 0
            for c in 0..<cols {
                let val = abs(srcPtr[r * cols + c])
                if val > maxAbs { maxAbs = val }
            }
            
            // Compute scale factor (avoid division by zero)
            let scale = maxAbs > 0 ? maxAbs / 127.0 : 1.0
            scalePtr[r] = scale
            
            // Quantize: float → round(float / scale) clamped to [-127, 127]
            for c in 0..<cols {
                let val = srcPtr[r * cols + c]
                let quantized = Int32(round(val / scale))
                let clamped = max(-127, min(127, quantized))
                dstPtr[r * cols + c] = Int8(clamped)
            }
        }
    }
    
    /// Pointer to the INT8 weight data.
    public func int8Pointer() -> UnsafeMutablePointer<Int8> {
        return buffer.contents().bindMemory(to: Int8.self, capacity: rows * cols)
    }
    
    /// Pointer to the FP32 scale data.
    public func scalePointer() -> UnsafeMutablePointer<Float> {
        return scales.contents().bindMemory(to: Float.self, capacity: rows)
    }
}
