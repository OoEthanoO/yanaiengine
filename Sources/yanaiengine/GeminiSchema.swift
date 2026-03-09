import Foundation

// MARK: - Gemini Request Schema

/// Gemini API request: POST /v1beta/models/yanai-model:generateContent
public struct GeminiRequest: Codable, Sendable {
    public let contents: [GeminiContent]
    public let generationConfig: GeminiGenerationConfig?
}

public struct GeminiContent: Codable, Sendable {
    public let role: String? // "user" or "model" (Llama uses "user", "assistant", "system")
    public let parts: [GeminiPart]
}

public struct GeminiPart: Codable, Sendable {
    public let text: String
}

public struct GeminiGenerationConfig: Codable, Sendable {
    public let temperature: Float?
    public let topP: Float?
    public let topK: Int?
    public let maxOutputTokens: Int?
    public let stopSequences: [String]?
}

// MARK: - Gemini Response Schema

/// Gemini API response
public struct GeminiResponse: Codable, Sendable {
    public let candidates: [GeminiCandidate]
    public let usageMetadata: GeminiUsageMetadata?
}

public struct GeminiCandidate: Codable, Sendable {
    public let content: GeminiContent
    public let finishReason: String?
    public let avgLogprobs: Double?
}

public struct GeminiUsageMetadata: Codable, Sendable {
    public let promptTokenCount: Int
    public let candidatesTokenCount: Int
    public let totalTokenCount: Int
}

// MARK: - Streaming Schema (SSE)

/// A single chunk in the streamGenerateContent response
public struct GeminiStreamChunk: Codable, Sendable {
    public let candidates: [GeminiCandidate]
    public let usageMetadata: GeminiUsageMetadata?
}
