import Metal
import Foundation

/// Model weight loader: injects weights from a SafetensorsReader into engine layers.
public class ModelLoader {
    
    /// Load FP32 weights from safetensors into a LinearLayer.
    /// Handles the weight matrix and optional bias.
    public static func loadLinearLayer(
        reader: SafetensorsReader,
        weightName: String,
        biasName: String?,
        into layer: LinearLayer
    ) throws {
        // Load weight matrix
        guard let weightInfo = reader.tensorInfo(name: weightName) else {
            throw SafetensorsError.tensorNotFound(weightName)
        }
        guard let weightPtr = reader.tensorData(name: weightName) else {
            throw SafetensorsError.tensorNotFound(weightName)
        }
        
        let expectedRows = layer.weights.rows
        let expectedCols = layer.weights.cols
        
        if weightInfo.dtype == "F32" {
            let srcFloats = weightPtr.bindMemory(to: Float.self, capacity: expectedRows * expectedCols)
            let dstFloats = layer.weights.pointer()
            
            // Check if we need to transpose (PyTorch stores [out, in], we store [in, out])
            if weightInfo.shape.count == 2 && weightInfo.shape[0] == expectedCols && weightInfo.shape[1] == expectedRows {
                // Transpose: source is [out x in], dest is [in x out]
                for r in 0..<expectedRows {
                    for c in 0..<expectedCols {
                        dstFloats[r * expectedCols + c] = srcFloats[c * expectedRows + r]
                    }
                }
            } else {
                // Direct copy (same layout)
                memcpy(dstFloats, srcFloats, expectedRows * expectedCols * MemoryLayout<Float>.stride)
            }
        } else if weightInfo.dtype == "F16" {
            // FP16 → FP32 conversion
            let srcHalf = weightPtr.bindMemory(to: UInt16.self, capacity: expectedRows * expectedCols)
            let dstFloats = layer.weights.pointer()
            
            let needsTranspose = weightInfo.shape.count == 2 && weightInfo.shape[0] == expectedCols && weightInfo.shape[1] == expectedRows
            
            for r in 0..<expectedRows {
                for c in 0..<expectedCols {
                    let srcIdx = needsTranspose ? (c * expectedRows + r) : (r * expectedCols + c)
                    dstFloats[r * expectedCols + c] = float16ToFloat32(srcHalf[srcIdx])
                }
            }
        }
        
        // Load bias if present
        if let biasName = biasName,
           let biasPtr = reader.tensorData(name: biasName) {
            let biasInfo = reader.tensorInfo(name: biasName)!
            if biasInfo.dtype == "F32" {
                let srcBias = biasPtr.bindMemory(to: Float.self, capacity: expectedCols)
                memcpy(layer.bias.pointer(), srcBias, expectedCols * MemoryLayout<Float>.stride)
            } else if biasInfo.dtype == "F16" {
                let srcBias = biasPtr.bindMemory(to: UInt16.self, capacity: expectedCols)
                let dstBias = layer.bias.pointer()
                for i in 0..<expectedCols {
                    dstBias[i] = float16ToFloat32(srcBias[i])
                }
            }
        }
    }
    
    /// Convert IEEE FP16 (half) to FP32 (float).
    private static func float16ToFloat32(_ h: UInt16) -> Float {
        let sign = UInt32(h >> 15) & 0x1
        let exponent = UInt32(h >> 10) & 0x1F
        let mantissa = UInt32(h) & 0x3FF
        
        var result: UInt32
        if exponent == 0 {
            if mantissa == 0 {
                result = sign << 31  // ±0
            } else {
                // Subnormal
                var e = exponent
                var m = mantissa
                while (m & 0x400) == 0 { m <<= 1; e -= 1 }
                e += 1
                m &= ~UInt32(0x400)
                result = (sign << 31) | ((e + 112) << 23) | (m << 13)
            }
        } else if exponent == 31 {
            result = (sign << 31) | 0x7F800000 | (mantissa << 13)  // Inf/NaN
        } else {
            result = (sign << 31) | ((exponent + 112) << 23) | (mantissa << 13)
        }
        
        return Float(bitPattern: result)
    }
    
    /// Load weights for an MoELayer from expert-partitioned tensors.
    public static func loadMoELayer(
        reader: SafetensorsReader,
        layerPrefix: String,
        into moe: MoELayer
    ) throws {
        // Load Router weights
        try loadLinearLayer(
            reader: reader,
            weightName: "\(layerPrefix).mlp.gate.weight",
            biasName: nil,
            into: moe.router.weights.rowAsLinear() // Helper needed or direct copy
        )
        
        // Load Experts
        for i in 0..<moe.experts.count {
            let expertPrefix = "\(layerPrefix).mlp.experts.\(i)"
            try loadLinearLayer(reader: reader, weightName: "\(expertPrefix).w1.weight", biasName: nil, into: moe.experts[i].gateProj)
            try loadLinearLayer(reader: reader, weightName: "\(expertPrefix).w3.weight", biasName: nil, into: moe.experts[i].upProj)
            try loadLinearLayer(reader: reader, weightName: "\(expertPrefix).w2.weight", biasName: nil, into: moe.experts[i].downProj)
        }
    }
}

extension Tensor {
    /// Returns a temporary LinearLayer-like object to use the loading logic.
    func rowAsLinear() -> LinearLayer {
        // This is a hack for the loader: it creates a "view" of the tensor as a LinearLayer
        // so we can use loadLinearLayer on it.
        return LinearLayer(engine: MetalEngine.shared, inputDim: self.rows, outputDim: self.cols, batchSize: 1, existingBuffer: self.buffer)
    }
}
