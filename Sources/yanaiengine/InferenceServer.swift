import Foundation
import Hummingbird
import NIOHTTP1

/// An asynchronous HTTP server that exposes a Gemini-compatible API for YanAIEngine.
public actor InferenceServer {
    private let draftModel: LlamaModel?
    private let model: LlamaModel
    private let scheduler: Scheduler
    private let tokenizer: Tokenizer?
    private let stopTokenId: UInt32
    
    // Multimodal Components
    private let visionEncoder: SigLIPEncoder
    private let projector: MultimodalProjector
    
    public init(model: LlamaModel, draftModel: LlamaModel? = nil, scheduler: Scheduler, tokenizer: Tokenizer? = nil, stopTokenId: UInt32 = 128009) {
        self.draftModel = draftModel
        self.model = model
        self.scheduler = scheduler
        self.tokenizer = tokenizer
        self.stopTokenId = stopTokenId
        
        self.visionEncoder = SigLIPEncoder(engine: model.blocks[0].outputProj.weights.buffer.device.description.contains("Apple") ? MetalEngine.shared : MetalEngine.shared) // Need cleaner engine access
        self.projector = MultimodalProjector(engine: MetalEngine.shared, visionDim: 1152, llmDim: model.config.dModel)
    }
    
    /// Start the server on the specified port.
    public func start(port: Int = 8080) async throws {
        let router = Router()
        
        // Endpoint: Single-shot generation
        router.post("/v1beta/models/yanai-model:generateContent") { request, context in
            let geminiReq = try await request.decode(as: GeminiRequest.self, context: context)
            let response = try await self.handleGenerateContent(geminiReq)
            let data = try JSONEncoder().encode(response)
            return Response(
                status: .ok,
                headers: [.contentType: "application/json"],
                body: .init(byteBuffer: ByteBuffer(data: data))
            )
        }
        
        // Endpoint: Streaming generation (SSE)
        router.post("/v1beta/models/yanai-model:streamGenerateContent") { request, context in
            let geminiReq = try await request.decode(as: GeminiRequest.self, context: context)
            
            // Get a stream from the actor
            let tokenStream = await self.generateStream(request: geminiReq)
            
            return Response(
                status: .ok,
                headers: [.contentType: "text/event-stream"],
                body: .init { writer in
                    var writer = writer
                    for await tokenText in tokenStream {
                        let chunk = GeminiStreamChunk(
                            candidates: [
                                GeminiCandidate(
                                    content: GeminiContent(role: "model", parts: [GeminiPart(text: tokenText, inlineData: nil)]),
                                    finishReason: nil,
                                    avgLogprobs: nil
                                )
                            ],
                            usageMetadata: nil
                        )
                        if let data = try? JSONEncoder().encode(chunk), let jsonString = String(data: data, encoding: .utf8) {
                            let sseEvent = "data: \(jsonString)\n\n"
                            try await writer.write(ByteBuffer(string: sseEvent))
                        }
                    }
                    
                    // Send final chunk
                    let finalChunk = GeminiStreamChunk(
                        candidates: [GeminiCandidate(content: GeminiContent(role: "model", parts: []), finishReason: "STOP", avgLogprobs: nil)],
                        usageMetadata: nil
                    )
                    if let data = try? JSONEncoder().encode(finalChunk), let jsonString = String(data: data, encoding: .utf8) {
                        try await writer.write(ByteBuffer(string: "data: \(jsonString)\n\n"))
                    }
                }
            )
        }
        
        let app = Application(router: router, configuration: .init(address: .hostname("0.0.0.0", port: port)))
        print("🚀 YanAIEngine Inference Server listening on http://localhost:\(port)")
        try await app.run()
    }
    
    // MARK: - Handlers
    
    private func handleGenerateContent(_ request: GeminiRequest) async throws -> GeminiResponse {
        let prompt = bridgeGeminiToLlama(request.contents)
        let config = request.generationConfig
        
        let maxTokens = config?.maxOutputTokens ?? 512
        let sampler = Sampler()
        sampler.temperature = config?.temperature ?? 0.8
        sampler.topP = config?.topP ?? 0.9
        sampler.topK = config?.topK ?? 50
        
        // Wait for generation to complete
        let generatedText = try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<String, Error>) in
            guard let tokens = tokenizer?.encode(text: prompt) else {
                continuation.resume(throwing: NSError(domain: "Server", code: 1, userInfo: [NSLocalizedDescriptionKey: "Tokenization failed"]))
                return
            }
            
            var result = ""
            // Process Images if present
            var visualEmbeds: Tensor? = nil
            if let base64Image = request.contents.first?.parts.compactMap({ $0.inlineData?.data }).first {
                visualEmbeds = self.processImage(base64Image)
            }
            
            let seq = SequenceRequest(
                promptTokens: tokens,
                numLayers: model.config.numLayers,
                allocator: model.allocator,
                draftNumLayers: draftModel?.config.numLayers,
                draftAllocator: draftModel?.allocator,
                sampler: sampler,
                maxTokens: maxTokens,
                stopTokenId: stopTokenId,
                onTokenGenerated: { text in result += text },
                onCompletion: { continuation.resume(returning: result) },
                visualEmbeddings: visualEmbeds
            )
            
            Task {
                await scheduler.submit(request: seq)
            }
        }
        
        return GeminiResponse(
            candidates: [
                GeminiCandidate(
                    content: GeminiContent(role: "model", parts: [GeminiPart(text: generatedText, inlineData: nil)]),
                    finishReason: "STOP",
                    avgLogprobs: nil
                )
            ],
            usageMetadata: nil
        )
    }
    
    func generateStream(request: GeminiRequest) -> AsyncStream<String> {
        let prompt = bridgeGeminiToLlama(request.contents)
        let config = request.generationConfig
        let maxTokens = config?.maxOutputTokens ?? 512
        
        let sampler = Sampler()
        sampler.temperature = config?.temperature ?? 0.8
        sampler.topP = config?.topP ?? 0.9
        sampler.topK = config?.topK ?? 50
        
        return AsyncStream { continuation in
            guard let tokens = tokenizer?.encode(text: prompt) else {
                continuation.finish()
                return
            }
            
            // Process Images if present
            var visualEmbeds: Tensor? = nil
            if let base64Image = request.contents.first?.parts.compactMap({ $0.inlineData?.data }).first {
                visualEmbeds = self.processImage(base64Image)
            }
            
            // This is messy - we need to fix the allocator access.
            // For now, assume model has access to allocator.
            let seq = SequenceRequest(
                promptTokens: tokens,
                numLayers: model.config.numLayers,
                allocator: model.allocator,
                draftNumLayers: draftModel?.config.numLayers,
                draftAllocator: draftModel?.allocator,
                sampler: sampler,
                maxTokens: maxTokens,
                stopTokenId: stopTokenId,
                onTokenGenerated: { text in continuation.yield(text) },
                onCompletion: { continuation.finish() },
                visualEmbeddings: visualEmbeds
            )
            
            Task {
                await scheduler.submit(request: seq)
            }
        }
    }
    
    // MARK: - Logic Bridge
    
    /// Convert Gemini's multi-turn content array into a single Llama 3 chat prompt.
    private func bridgeGeminiToLlama(_ contents: [GeminiContent]) -> String {
        var prompt = "<|begin_of_text|>"
        for content in contents {
            let role = content.role ?? "user"
            let text = content.parts.compactMap { $0.text }.joined(separator: "\n")
            prompt += "<|start_header_id|>\(role)<|end_header_id|>\n\n\(text)<|eot_id|>"
        }
        return prompt
    }
    
    /// Decodes base64 image, runs SigLIP, and projects to LLM space.
    private func processImage(_ base64: String) -> Tensor {
        // 1. Decode base64 to Tensor [3 x 224 x 224]
        // (Placeholder: In a real app we'd use ImageIO or similar)
        let imgTensor = Tensor(device: MetalEngine.shared.device, rows: 3, cols: 224 * 224)
        
        // 2. Vision Encoder
        let vTokens = visionEncoder.forward(image: imgTensor)
        
        // 3. Project to LLM Space
        return projector.forward(visionTokens: vTokens)
    }
    
    /// Core generation loop (serial execution on the GPU).
    private func generate(prompt: String, maxTokens: Int) async throws -> String {
        let sampler = Sampler()
        // Since this is an actor, this whole method is serial.
        // We'll use the existing LlamaModel & Sampler logic.
        
        // Tokenize
        guard let tokens = tokenizer?.encode(text: prompt) else { return "No tokenizer available" }
        
        model.resetCaches()
        var sequence = tokens
        
        // Prefill
        let logitsPtr = model.prefill(tokenIds: tokens)
        var logitsCopy = [Float](repeating: 0, count: model.config.vocabSize)
        for i in 0..<model.config.vocabSize { logitsCopy[i] = logitsPtr[i] }
        let nextToken = sampler.sample(logits: &logitsCopy, vocabSize: model.config.vocabSize)
        sequence.append(nextToken)
        
        if nextToken == stopTokenId { // <|eot_id|>
            return tokenizer?.decode(ids: [nextToken]) ?? ""
        }
        
        // Decode loop
        var result = tokenizer?.decode(ids: [nextToken]) ?? ""
        for _ in 0..<maxTokens {
            let decLogits = model.decode(tokenId: sequence.last!)
            for i in 0..<model.config.vocabSize { logitsCopy[i] = decLogits[i] }
            let tok = sampler.sample(logits: &logitsCopy, vocabSize: model.config.vocabSize)
            
            if tok == stopTokenId { break }
            sequence.append(tok)
            result += tokenizer?.decode(ids: [tok]) ?? ""
        }
        
        return result
    }
    
    private func runGenerationStreaming(prompt: String, maxTokens: Int, callback: @Sendable (String) async throws -> Void) async throws {
        let sampler = Sampler()
        guard let tokens = tokenizer?.encode(text: prompt) else { return }
        
        model.resetCaches()
        var sequence = tokens
        
        // Prefill
        let logitsPtr = model.prefill(tokenIds: tokens)
        var logitsCopy = [Float](repeating: 0, count: model.config.vocabSize)
        for i in 0..<model.config.vocabSize { logitsCopy[i] = logitsPtr[i] }
        let nextToken = sampler.sample(logits: &logitsCopy, vocabSize: model.config.vocabSize)
        sequence.append(nextToken)
        
        let initialText = tokenizer?.decode(ids: [nextToken]) ?? ""
        try await callback(initialText)
        
        if nextToken == stopTokenId { return }
        
        // Decode loop
        for _ in 0..<maxTokens {
            let decLogits = model.decode(tokenId: sequence.last!)
            for i in 0..<model.config.vocabSize { logitsCopy[i] = decLogits[i] }
            let tok = sampler.sample(logits: &logitsCopy, vocabSize: model.config.vocabSize)
            
            if tok == stopTokenId { break }
            sequence.append(tok)
            let tokenText = tokenizer?.decode(ids: [tok]) ?? ""
            try await callback(tokenText)
        }
    }
}
