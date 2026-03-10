import Foundation
import Metal

/// The Multimodal Projector (The Bridge).
/// Translates SigLIP vision tokens into Gemma LLM space.
public class MultimodalProjector {
    private let engine: MetalEngine
    private let projection: LinearLayer
    
    public init(engine: MetalEngine, visionDim: Int = 1152, llmDim: Int = 3584) {
        self.engine = engine
        // PaliGemma usually uses a simple Linear layer or a small MLP.
        self.projection = LinearLayer(engine: engine, rows: llmDim, cols: visionDim)
    }
    
    /// Projects vision tokens into LLM space.
    /// - Parameter visionTokens: [numPatches x visionDim]
    /// - Returns: Projected tokens [numPatches x llmDim]
    public func forward(visionTokens: Tensor) -> Tensor {
        projection.forward(input: visionTokens)
        return projection.output
    }
}
