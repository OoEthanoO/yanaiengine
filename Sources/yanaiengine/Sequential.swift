import Foundation
import Metal

/// A Sequential model that orchestrates a chain of layers.
/// It solves the non-linear XOR problem by propagating gradients through multiple layers.
public class Sequential {
    public private(set) var layers: [LinearLayer]
    
    public init(layers: [LinearLayer]) {
        self.layers = layers
    }
    
    /// Performs a forward pass through all layers.
    /// - Parameter input: The initial input tensor.
    /// - Returns: The output of the final layer.
    public func forward(input: Tensor) -> Tensor {
        var currentInput = input
        for layer in layers {
            layer.forward(input: currentInput)
            currentInput = layer.output
        }
        return currentInput
    }
    
    /// Performs a backward pass (Backpropagation) through all layers in reverse.
    /// - Parameters:
    ///   - lossGradient: The gradient of the loss with respect to the output.
    ///   - learningRate: The step size for SGD.
    public func backward(lossGradient: Tensor, learningRate: Float) {
        computeGradients(lossGradient: lossGradient)
        applyUpdates(learningRate: learningRate)
    }
    
    /// Computes gradients for all layers but does not update them.
    public func computeGradients(lossGradient: Tensor) {
        var currentGradient = lossGradient
        for layer in layers.reversed() {
            currentGradient = layer.computeGradients(upstreamGradient: currentGradient)
        }
    }
    
    /// Applies updates to all layers.
    public func applyUpdates(learningRate: Float) {
        for layer in layers {
            layer.applyUpdates(learningRate: learningRate)
        }
    }
}
