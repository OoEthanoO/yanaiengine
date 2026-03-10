import Metal
import Foundation

/// Key-Value Cache for efficient autoregressive inference.
/// Stores historical K and V vectors per head to avoid recomputation.
public class KVCache {
    public let numHeads: Int
    public let dHead: Int
    public let maxSeqLen: Int
    
    // Per-head cache buffers: [maxSeqLen x dHead]
    public var cachedKeys: [Tensor]
    public var cachedValues: [Tensor]
    
    // How many positions have been filled
    public var currentPosition: Int = 0
    
    public init(device: MTLDevice, numHeads: Int, dHead: Int, maxSeqLen: Int) {
        self.numHeads = numHeads
        self.dHead = dHead
        self.maxSeqLen = maxSeqLen
        
        self.cachedKeys = (0..<numHeads).map { _ in
            Tensor(device: device, rows: maxSeqLen, cols: dHead)
        }
        self.cachedValues = (0..<numHeads).map { _ in
            Tensor(device: device, rows: maxSeqLen, cols: dHead)
        }
    }
    
    /// Append a single token's K and V for a specific head at the current position.
    public func append(key: UnsafePointer<Float>, value: UnsafePointer<Float>, head: Int) {
        let kPtr = cachedKeys[head].pointer()
        let vPtr = cachedValues[head].pointer()
        let offset = currentPosition * dHead
        
        for d in 0..<dHead {
            kPtr[offset + d] = key[d]
            vPtr[offset + d] = value[d]
        }
    }
    
    /// Append K/V for all heads from full-width K and V tensors at a given row.
    /// K and V are [seqLen x dModel], we extract row `tokenIdx` and split across heads.
    /// Supports sliding window via circular indexing.
    public func appendFromFull(kTensor: Tensor, vTensor: Tensor, tokenIdx: Int, dModel: Int, windowSize: Int = 0) {
        let kPtr = kTensor.pointer()
        let vPtr = vTensor.pointer()
        
        // If windowSize > 0, we treat the cache as a circular buffer
        let targetPos = (windowSize > 0) ? (currentPosition % windowSize) : currentPosition
        
        for h in 0..<numHeads {
            let headOffset = h * dHead
            let ckPtr = cachedKeys[h].pointer()
            let cvPtr = cachedValues[h].pointer()
            let cacheOffset = targetPos * dHead
            
            for d in 0..<dHead {
                ckPtr[cacheOffset + d] = kPtr[tokenIdx * dModel + headOffset + d]
                cvPtr[cacheOffset + d] = vPtr[tokenIdx * dModel + headOffset + d]
            }
        }
    }
    
    /// Advance the position counter by 1.
    public func advancePosition() {
        currentPosition += 1
    }
    
    /// Reset the cache (for a new generation session).
    public func reset() {
        currentPosition = 0
    }
}
