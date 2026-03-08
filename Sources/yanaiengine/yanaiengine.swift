import Foundation
import Metal

@main
struct yanaiengine {
    static func main() {
        print("==============================================")
        print("Starting YanAIEngine: Goal #3 (Training Loop)")
        print("==============================================\n")
        
        // Initialize Metal Engine
        guard let engine = MetalEngine() else {
            fatalError("Failed to initialize Metal Engine")
        }
        
        // Load all required kernels
        print("Loading Metal kernels...")
        engine.loadLibrary(resourceName: "gemm", kernelName: "gemm_kernel")
        engine.loadLibrary(resourceName: "gemm", kernelName: "bias_add_kernel")
        engine.loadLibrary(resourceName: "gemm", kernelName: "relu_kernel")
        engine.loadLibrary(resourceName: "gemm", kernelName: "transpose_kernel")
        engine.loadLibrary(resourceName: "gemm", kernelName: "mse_derivative_kernel")
        engine.loadLibrary(resourceName: "gemm", kernelName: "sgd_update_kernel")
        
        // Define dimensions
        let batchSize = 1
        let inputDim = 2
        let outputDim = 1
        
        // 1. Setup Training Data (X -> Y)
        // We want the network to learn that [1, 2] -> [5]
        let input = Tensor(device: engine.device, rows: batchSize, cols: inputDim)
        input.pointer()[0] = 1.0
        input.pointer()[1] = 2.0
        
        let target = Tensor(device: engine.device, rows: batchSize, cols: outputDim)
        target.pointer()[0] = 5.0
        
        // 2. Initialize Layer
        let linear = LinearLayer(engine: engine, inputDim: inputDim, outputDim: outputDim, batchSize: batchSize)
        // Set specific weights to start
        linear.weights.pointer()[0] = 0.1
        linear.weights.pointer()[1] = 0.1
        linear.bias.fill(with: 0.0)
        
        // 3. Loss Gradient Buffer (dY)
        let lossGradient = Tensor(device: engine.device, rows: batchSize, cols: outputDim)
        
        // 4. Training Loop
        let epochs = 100
        let learningRate: Float = 0.1
        
        print("\nStarting Training for \(epochs) epochs (LR: \(learningRate))...")
        
        for epoch in 1...epochs {
            // A. Forward Pass
            linear.forward(input: input)
            
            // B. Calculate Loss Gradient (dY = Output - Target) on GPU
            guard let commandBuffer = engine.commandQueue.makeCommandBuffer(),
                  let encoder = commandBuffer.makeComputeCommandEncoder() else {
                fatalError("Failed to create command buffer")
            }
            
            let lossPSO = engine.getPipelineState(name: "mse_derivative_kernel")
            encoder.setComputePipelineState(lossPSO)
            encoder.setBuffer(linear.output.buffer, offset: 0, index: 0)
            encoder.setBuffer(target.buffer, offset: 0, index: 1)
            encoder.setBuffer(lossGradient.buffer, offset: 0, index: 2)
            var length = UInt32(batchSize * outputDim)
            encoder.setBytes(&length, length: MemoryLayout<UInt32>.size, index: 3)
            
            encoder.dispatchThreads(MTLSize(width: 1, height: 1, depth: 1), 
                                    threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
            encoder.endEncoding()
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
            
            // C. Backward Pass (Calculates dW and updates Weights)
            linear.backward(upstreamGradient: lossGradient, learningRate: learningRate)
            
            // D. Log Progress
            if epoch % 10 == 0 || epoch == 1 {
                let currentOut = linear.output.pointer()[0]
                let loss = 0.5 * pow(currentOut - 5.0, 2)
                print("Epoch \(epoch): Loss = \(String(format: "%.6f", loss)), Output = \(String(format: "%.4f", currentOut))")
            }
        }
        
        print("\nTraining Complete!")
        print("Final Output for [1, 2]: \(linear.output.pointer()[0]) (Target: 5.0)")
        print("Final Weights: \(linear.weights.pointer()[0]), \(linear.weights.pointer()[1])")
    }
}
