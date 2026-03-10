import Foundation
import Metal

func testPagedAttention() {
    print("🚀 Starting PagedAttention Verification...")
    
    let engine = MetalEngine()
    let device = engine.device
    
    // 1. Initialize Block Allocator
    // Pool: 1024 blocks, 16 tokens per block, 8 heads, 64 dimension
    let allocator = BlockAllocator(device: device, numBlocks: 1024, blockSize: 16, numKVHeads: 8, dHead: 64)
    print("💎 BlockAllocator initialized with \(allocator.numBlocks) physical blocks.")
    
    // 2. Initialize PagedKVCache
    let cache = PagedKVCache(allocator: allocator)
    print("🧠 PagedKVCache created.")
    
    // 3. Simulate sequence growth
    let dModel = 8 * 64
    let dummyK = Tensor(device: device, rows: 1, cols: dModel)
    let dummyV = Tensor(device: device, rows: 1, cols: dModel)
    dummyK.fill(with: 1.0)
    dummyV.fill(with: 2.0)
    
    print("📈 Appending 20 logical tokens (should trigger second physical block)...")
    for i in 0..<20 {
        cache.appendFromFull(kTensor: dummyK, vTensor: dummyV, tokenIdx: 0)
    }
    
    print("📊 Current position: \(cache.currentPosition)")
    print("📉 Page Table size: \(cache.pageTable.count)")
    print("🏗️ Allocator usage: \(allocator.usage * 100)%")
    
    precondition(cache.pageTable.count == 2, "Page Table should have 2 entries for 20 tokens (block size 16)")
    print("✅ Virtual mapping confirmed.")
    
    // 4. Verify Metal Kernel Access
    print("🔍 Checking if paged_fused_attention_kernel is accessible...")
    _ = engine.getPipelineState(name: "paged_fused_attention_kernel")
    print("✅ paged_fused_attention_kernel confirmed.")
    
    print("🎉 PagedAttention verification passed!")
}

testPagedAttention()
