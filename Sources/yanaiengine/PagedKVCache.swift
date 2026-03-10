import Metal
import Foundation

/// Virtualized Key-Value Cache.
/// Instead of contiguous tensors, it holds a Page Table mapping to physical blocks in a BlockAllocator.
public class PagedKVCache {
    private let allocator: BlockAllocator
    public var pageTable: [Int] = [] // Logical page index -> Physical block index
    
    public var currentPosition: Int = 0
    public let numKVHeads: Int
    public let dHead: Int
    public let blockSize: Int
    
    public init(allocator: BlockAllocator) {
        self.allocator = allocator
        self.numKVHeads = allocator.numKVHeads
        self.dHead = allocator.dHead
        self.blockSize = allocator.blockSize
    }
    
    /// Append a single token's KV across all heads.
    /// Manages dynamic block allocation as the sequence grows.
    public func appendFromFull(kTensor: Tensor, vTensor: Tensor, tokenIdx: Int) {
        let kPtr = kTensor.pointer()
        let vPtr = vTensor.pointer()
        let dModel = numKVHeads * dHead
        
        // 1. Ensure we have a block for the current position
        if currentPosition % blockSize == 0 {
            guard let newBlock = allocator.allocate() else {
                fatalError("BlockAllocator out of memory!")
            }
            pageTable.append(newBlock)
        }
        
        // 2. Resolve target block and offset
        let pageIdx = currentPosition / blockSize
        let physicalBlockIdx = pageTable[pageIdx]
        let offsetInBlock = currentPosition % blockSize
        
        // 3. Write data to the physical pool
        let kPool = allocator.keyBuffer.contents().bindMemory(to: Float.self, capacity: allocator.numBlocks * numKVHeads * blockSize * dHead)
        let vPool = allocator.valueBuffer.contents().bindMemory(to: Float.self, capacity: allocator.numBlocks * numKVHeads * blockSize * dHead)
        
        for h in 0..<numKVHeads {
            // Memory layout: [numBlocks][numKVHeads][blockSize][dHead]
            let blockOffset = physicalBlockIdx * (numKVHeads * blockSize * dHead)
            let headOffsetInBlock = h * (blockSize * dHead)
            let tokenOffsetInHead = offsetInBlock * dHead
            
            let poolOffset = blockOffset + headOffsetInBlock + tokenOffsetInHead
            let modelOffset = tokenIdx * dModel + h * dHead
            
            for d in 0..<dHead {
                kPool[poolOffset + d] = kPtr[modelOffset + d]
                vPool[poolOffset + d] = vPtr[modelOffset + d]
            }
        }
        
        currentPosition += 1
    }
    
    /// Release all physical blocks back to the pool.
    public func reset() {
        for blockIdx in pageTable {
            allocator.deallocate(blockIdx: blockIdx)
        }
        pageTable.removeAll()
        currentPosition = 0
    }
    
    deinit {
        reset()
    }
}
