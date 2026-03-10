import Metal
import Foundation

/// A global physical memory manager for KV Cache blocks.
/// Inspired by vLLM, it pre-allocates a massive memory pool and manages it in fixed-size blocks.
public class BlockAllocator {
    private let device: MTLDevice
    public let blockSize: Int        // Tokens per block (e.g., 16)
    public let numBlocks: Int       // Total number of blocks in the pool
    public let dHead: Int           // Dimension of each head
    public let numKVHeads: Int      // Number of KV heads
    
    // The physical memory pool: [numBlocks x numKVHeads x blockSize x dHead]
    // To simplify kernel access, we combine all heads into one large buffer.
    public let keyBuffer: MTLBuffer
    public let valueBuffer: MTLBuffer
    
    private var freeBlocks: [Int]
    private var allocatedBlocks: Set<Int> = []
    
    public init(device: MTLDevice, numBlocks: Int, blockSize: Int, numKVHeads: Int, dHead: Int) {
        self.device = device
        self.numBlocks = numBlocks
        self.blockSize = blockSize
        self.numKVHeads = numKVHeads
        self.dHead = dHead
        
        let totalElements = numBlocks * numKVHeads * blockSize * dHead
        let byteCount = totalElements * MemoryLayout<Float>.stride
        
        guard let kBuf = device.makeBuffer(length: byteCount, options: .storageModeShared),
              let vBuf = device.makeBuffer(length: byteCount, options: .storageModeShared) else {
            fatalError("Failed to allocate BlockAllocator pools of size \(byteCount)")
        }
        
        self.keyBuffer = kBuf
        self.valueBuffer = vBuf
        self.freeBlocks = Array(0..<numBlocks).reversed() // Stack-like for fast pop
    }
    
    /// Allocate a physical block and return its index.
    public func allocate() -> Int? {
        guard let blockIdx = freeBlocks.popLast() else { return nil }
        allocatedBlocks.insert(blockIdx)
        return blockIdx
    }
    
    /// Release a physical block.
    public func deallocate(blockIdx: Int) {
        if allocatedBlocks.remove(blockIdx) != nil {
            freeBlocks.append(blockIdx)
        }
    }
    
    /// Calculate current memory usage percentage.
    public var usage: Float {
        return Float(allocatedBlocks.count) / Float(numBlocks)
    }
    
    /// Reset the entire allocator.
    public func reset() {
        freeBlocks = Array(0..<numBlocks).reversed()
        allocatedBlocks.removeAll()
    }
}
