import Foundation
import Metal

/// Repesents a single generation request in the system.
public class SequenceRequest: Identifiable {
    public let id: UUID = UUID()
    public var promptTokens: [UInt32]
    public var generatedTokens: [UInt32] = []
    public let pagedKVCache: [PagedKVCache] // Target model cache
    public let draftKVCache: [PagedKVCache]? // Draft model cache
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
                draftNumLayers: Int? = nil,
                draftAllocator: BlockAllocator? = nil,
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
        
        if let draftNumLayers = draftNumLayers, let draftAllocator = draftAllocator {
            self.draftKVCache = (0..<draftNumLayers).map { _ in
                PagedKVCache(allocator: draftAllocator)
            }
        } else {
            self.draftKVCache = nil
        }
    }
    
    public var pendingTokensForTarget: [UInt32] = []
    
    /// Returns the input token(s) for the next forward pass.
    public func getNextInput(isDraft: Bool) -> [UInt32] {
        if !isPrefillDone {
            return promptTokens
        } else {
            if isDraft {
                // Draft model just needs the last generated token
                return [generatedTokens.last ?? promptTokens.last!]
            } else {
                // Target model needs all pending drafted tokens (or just the last generated if not speculative)
                let tokens = pendingTokensForTarget.isEmpty ? [generatedTokens.last ?? promptTokens.last!] : pendingTokensForTarget
                return tokens
            }
        }
    }
}

/// The Continuous Batching Scheduler (ORCA-style).
/// Orchestrates the execution loop and manages multiple sequences simultaneously.
public actor Scheduler {
    private let draftModel: LlamaModel?
    private let model: LlamaModel
    private let engine: MetalEngine
    private let allocator: BlockAllocator
    private let tokenizer: Tokenizer?
    
    private var waitingQueue: [SequenceRequest] = []
    private var runningQueue: [SequenceRequest] = []
    
    private var isLoopRunning: Bool = false
    
    public init(model: LlamaModel, draftModel: LlamaModel? = nil, engine: MetalEngine, allocator: BlockAllocator, tokenizer: Tokenizer?) {
        self.draftModel = draftModel
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
        let batch = runningQueue
        if batch.isEmpty { return }
        
        let isSpeculative = draftModel != nil
        
        if isSpeculative, let draftModel = draftModel {
            let gamma = 4
            
            // Per-sequence state
            var draftTokens: [[UInt32]] = Array(repeating: [], count: batch.count)
            var draftProbs: [[[Float]]] = Array(repeating: [], count: batch.count)
            let targetCachePositions: [Int] = batch.map { $0.pagedKVCache[0].currentPosition }
            let wasPrefillDone: [Bool] = batch.map { $0.isPrefillDone }
            
            // 1. DRAFT PHASE
            for _ in 0..<gamma {
                let allDraftLogits = draftModel.forwardStep(batch: batch, allocator: draftModel.allocator, isDraft: true)
                for i in 0..<batch.count {
                    let req = batch[i]
                    if req.isFinished { continue }
                    
                    let tokenLogitsPtr = allDraftLogits[i].last!
                    var logitsCopy = [Float](repeating: 0, count: model.config.vocabSize)
                    for j in 0..<model.config.vocabSize { logitsCopy[j] = tokenLogitsPtr[j] }
                    
                    let probs = SpeculativeSampler.softmax(logits: logitsCopy)
                    let token = SpeculativeSampler.sampleFromProbs(probs)
                    
                    draftTokens[i].append(token)
                    draftProbs[i].append(probs)
                    
                    req.generatedTokens.append(token)
                    req.isPrefillDone = true
                }
            }
            
            // 2. VERIFICATION SETUP
            for i in 0..<batch.count {
                let req = batch[i]
                if req.isFinished { continue }
                
                let originalPrefillDone = wasPrefillDone[i]
                req.generatedTokens.removeLast(draftTokens[i].count)
                req.isPrefillDone = originalPrefillDone 
                
                if !req.isPrefillDone {
                    req.pendingTokensForTarget = req.promptTokens + draftTokens[i]
                } else {
                    let lastVerified = req.generatedTokens.last ?? req.promptTokens.last!
                    req.pendingTokensForTarget = [lastVerified] + draftTokens[i]
                }
            }
            
            // 3. TARGET MODEL PARALLEL FORWARD
            let allTargetLogits = model.forwardStep(batch: batch, allocator: allocator, isDraft: false)
            
            // 4. VERIFICATION & ROLLBACK
            for i in 0..<batch.count {
                let req = batch[i]
                if req.isFinished { continue }
                
                req.isPrefillDone = true
                req.pendingTokensForTarget = []
                
                let targetLogitsPtrs = allTargetLogits[i]
                let numDrafted = draftTokens[i].count
                let numOutputLogits = targetLogitsPtrs.count
                var targetLogitsArray: [[Float]] = []
                
                let startIndexLogits = max(0, numOutputLogits - numDrafted - 1)
                
                for ptrIdx in startIndexLogits..<numOutputLogits {
                    let ptr = targetLogitsPtrs[ptrIdx]
                    var logitsCopy = [Float](repeating: 0, count: model.config.vocabSize)
                    for j in 0..<model.config.vocabSize { logitsCopy[j] = ptr[j] }
                    targetLogitsArray.append(logitsCopy)
                }
                
                let acceptedSequence = SpeculativeSampler.verifySequence(
                    draftedTokens: draftTokens[i],
                    draftProbs: draftProbs[i],
                    targetLogits: targetLogitsArray,
                    vocabSize: model.config.vocabSize
                )
                
                // Rollback caches
                let finalLen = targetCachePositions[i] + acceptedSequence.count - 1
                for layerIdx in 0..<model.config.numLayers {
                    req.pagedKVCache[layerIdx].rollback(to: finalLen)
                }
                
                if let draftCaches = req.draftKVCache {
                    for layerIdx in 0..<draftModel.config.numLayers {
                        draftCaches[layerIdx].rollback(to: finalLen)
                    }
                }
                
                req.generatedTokens.append(contentsOf: acceptedSequence)
                
                for t in acceptedSequence {
                    let text = tokenizer?.decode(ids: [t]) ?? ""
                    req.onTokenGenerated(text)
                    if t == req.stopTokenId {
                        req.isFinished = true
                        break
                    }
                }
                
                if req.generatedTokens.count >= req.maxTokens {
                    req.isFinished = true
                }
            }
            
        } else {
            // Standard Autoregressive Decode
            let allLogitsPointers = model.forwardStep(batch: batch, allocator: allocator, isDraft: false)
            
            for i in 0..<batch.count {
                let req = batch[i]
                let logitsPtrs = allLogitsPointers[i]
                if logitsPtrs.isEmpty { continue }
                let logits = logitsPtrs.last!
                
                var logitsCopy = [Float](repeating: 0, count: model.config.vocabSize)
                for j in 0..<model.config.vocabSize { logitsCopy[j] = logits[j] }
                let nextToken = req.sampler.sample(logits: &logitsCopy, vocabSize: model.config.vocabSize)
                
                req.generatedTokens.append(nextToken)
                let text = tokenizer?.decode(ids: [nextToken]) ?? ""
                req.onTokenGenerated(text)
                
                req.isPrefillDone = true
                
                if nextToken == req.stopTokenId || req.generatedTokens.count >= req.maxTokens {
                    req.isFinished = true
                }
            }
        }
    }
}
