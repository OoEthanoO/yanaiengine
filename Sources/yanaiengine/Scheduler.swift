import Foundation
import Metal

/// Repesents a single generation request in the system.
public class SequenceRequest: Identifiable {
    public let id: UUID = UUID()
    public var promptTokens: [UInt32]
    public var generatedTokens: [UInt32] = []
    public let pagedKVCache: [PagedKVCache] // One per layer
    public let sampler: Sampler
    public let maxTokens: Int
    public let stopTokenId: UInt32
    
    // Vision Tokens (Projected Embeddings)
    public var visualEmbeddings: Tensor? // [numPatches x dModel]
    
    // Callback to stream tokens back to the client
    public let onTokenGenerated: (String) -> Void
    public let onCompletion: () -> Void
    
    public var isFinished: Bool = false
    public var isPrefillDone: Bool = false
    
    public var totalTokens: Int {
        let visionCount = visualEmbeddings?.rows ?? 0
        return visionCount + promptTokens.count + generatedTokens.count
    }
    
    public init(promptTokens: [UInt32], 
                numLayers: Int, 
                allocator: BlockAllocator,
                sampler: Sampler,
                maxTokens: Int, 
                stopTokenId: UInt32,
                onTokenGenerated: @escaping (String) -> Void,
                onCompletion: @escaping () -> Void,
                visualEmbeddings: Tensor? = nil) {
        self.promptTokens = promptTokens
        self.sampler = sampler
        self.maxTokens = maxTokens
        self.stopTokenId = stopTokenId
        self.onTokenGenerated = onTokenGenerated
        self.onCompletion = onCompletion
        self.visualEmbeddings = visualEmbeddings
        
        // Initialize independent paged cache per layer
        self.pagedKVCache = (0..<numLayers).map { _ in
            PagedKVCache(allocator: allocator)
        }
    }
    
    /// Returns the input token(s) for the next forward pass.
    /// If prefill is not done, returns all prompt tokens.
    /// Otherwise returns the last generated token.
    public func getNextInput() -> [UInt32] {
        if !isPrefillDone {
            return promptTokens
        } else {
            return [generatedTokens.last ?? promptTokens.last!]
        }
    }
}

/// The Continuous Batching Scheduler (ORCA-style).
/// Orchestrates the execution loop and manages multiple sequences simultaneously.
public actor Scheduler {
    private let model: LlamaModel
    private let engine: MetalEngine
    private let allocator: BlockAllocator
    private let tokenizer: Tokenizer?
    
    private var waitingQueue: [SequenceRequest] = []
    private var runningQueue: [SequenceRequest] = []
    
    private var isLoopRunning: Bool = false
    
    public init(model: LlamaModel, engine: MetalEngine, allocator: BlockAllocator, tokenizer: Tokenizer?) {
        self.model = model
        self.engine = engine
        self.allocator = allocator
        self.tokenizer = tokenizer
    }
    
    /// Submit a new request to the waiting queue.
    public func submit(request: SequenceRequest) {
        waitingQueue.append(request)
        
        // Start the execution loop if not already running
        if !isLoopRunning {
            Task {
                await runLoop()
            }
        }
    }
    
    /// Main execution loop.
    private func runLoop() async {
        isLoopRunning = true
        
        while !waitingQueue.isEmpty || !runningQueue.isEmpty {
            // 1. Scheduling: Move waiting requests to running queue if memory allows
            while !waitingQueue.isEmpty {
                let req = waitingQueue[0]
                // Simple heuristic: check if we have some blocks free (at least for initial prefill)
                if allocator.usage < 0.9 {
                    runningQueue.append(waitingQueue.removeFirst())
                } else {
                    break // Memory wall
                }
            }
            
            if runningQueue.isEmpty {
                try? await Task.sleep(nanoseconds: 10_000_000) // 10ms
                continue
            }
            
            // 2. Step: One forward pass for the entire batch
            autoreleasepool {
                step()
            }
            
            // 3. Cleanup: Remove finished requests
            runningQueue.removeAll { req in
                if req.isFinished {
                    req.onCompletion()
                    return true
                }
                return false
            }
        }
        
        isLoopRunning = false
    }
    
    /// Performs a single heterogeneous forward pass on the GPU.
    private func step() {
        // Collect batch of sequences
        let batch = runningQueue
        
        // Orchestrate batch forward pass
        let allLogitsPointers = model.forwardStep(batch: batch, allocator: allocator)
        
        for i in 0..<batch.count {
            let req = batch[i]
            let logits = allLogitsPointers[i]
            
            // Sample
            var logitsCopy = [Float](repeating: 0, count: model.config.vocabSize)
            for j in 0..<model.config.vocabSize { logitsCopy[j] = logits[j] }
            let nextToken = req.sampler.sample(logits: &logitsCopy, vocabSize: model.config.vocabSize)
            
            req.generatedTokens.append(nextToken)
            
            // Output handling
            let text = tokenizer?.decode(ids: [nextToken]) ?? ""
            req.onTokenGenerated(text)
            
            // State update
            req.isPrefillDone = true
            
            // Completion check
            if nextToken == req.stopTokenId || req.generatedTokens.count >= req.maxTokens {
                req.isFinished = true
            }
        }
    }
}
