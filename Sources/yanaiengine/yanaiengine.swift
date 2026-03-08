import Foundation
import Metal

@main
struct yanaiengine {
    static func main() {
        print("==============================================")
        print("Starting YanAIEngine: Goal #4 (The XOR Problem)")
        print("==============================================\n")
        
        // Initialize Metal Engine
        guard let engine = MetalEngine() else {
            fatalError("Failed to initialize Metal Engine")
        }
        
        // Load all required kernels for multi-layer training
        print("Loading Metal kernels...")
        let kernels = [
            "gemm_kernel", "bias_add_kernel", "relu_kernel", 
            "transpose_kernel", "mse_derivative_kernel", 
            "sgd_update_kernel", "relu_derivative_kernel",
            "sum_rows_kernel"
        ]
        for kernel in kernels {
            engine.loadLibrary(resourceName: "gemm", kernelName: kernel)
        }
        
        // 1. Setup XOR Dataset
        let batchSize = 4
        let inputDim = 2
        let outputDim = 1
        
        let inputs = Tensor(device: engine.device, rows: batchSize, cols: inputDim)
        let targets = Tensor(device: engine.device, rows: batchSize, cols: outputDim)
        
        // Data: [0,0]->[0], [0,1]->[1], [1,0]->[1], [1,1]->[0]
        let ptrIn = inputs.pointer()
        ptrIn[0] = 0; ptrIn[1] = 0
        ptrIn[2] = 0; ptrIn[3] = 1
        ptrIn[4] = 1; ptrIn[5] = 0
        ptrIn[6] = 1; ptrIn[7] = 1
        
        let ptrTar = targets.pointer()
        ptrTar[0] = 0
        ptrTar[1] = 1
        ptrTar[2] = 1
        ptrTar[3] = 0
        
        // 2. Initialize Sequential Model (2 -> 16 -> 1)
        // CRITICAL: The hidden layer uses ReLU, but the output layer is linear (ReLU disabled)
        // This prevents the "Dying ReLU" problem and allows stable convergence for binary targets.
        print("Architecting MLP: 2 Inputs -> 16 Hidden Nodes (ReLU) -> 1 Output (Linear)...")
        let layer1 = LinearLayer(engine: engine, inputDim: 2, outputDim: 16, batchSize: batchSize, useReLU: true)
        let layer2 = LinearLayer(engine: engine, inputDim: 16, outputDim: 1, batchSize: batchSize, useReLU: false)
        
        let model = Sequential(layers: [layer1, layer2])
        
        // 3. Training Loop
        let epochs = 2000
        let learningRate: Float = 0.1
        let lossGradient = Tensor(device: engine.device, rows: batchSize, cols: outputDim)
        
        print("\nStarting Training for \(epochs) epochs...")
        
        for epoch in 1...epochs {
            // A. Forward Pass
            let output = model.forward(input: inputs)
            
            // B. Calculate Loss Gradient (dY = Output - Target)
            guard let commandBuffer = engine.commandQueue.makeCommandBuffer(),
                  let encoder = commandBuffer.makeComputeCommandEncoder() else { fatalError() }
            
            let lossPSO = engine.getPipelineState(name: "mse_derivative_kernel")
            encoder.setComputePipelineState(lossPSO)
            encoder.setBuffer(output.buffer, offset: 0, index: 0)
            encoder.setBuffer(targets.buffer, offset: 0, index: 1)
            encoder.setBuffer(lossGradient.buffer, offset: 0, index: 2)
            var n = UInt32(batchSize * outputDim)
            encoder.setBytes(&n, length: MemoryLayout<UInt32>.size, index: 3)
            
            encoder.dispatchThreads(MTLSize(width: batchSize, height: 1, depth: 1), 
                                    threadsPerThreadgroup: MTLSize(width: batchSize, height: 1, depth: 1))
            encoder.endEncoding()
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
            
            // C. Backward Pass (Chain Rule across layers)
            model.backward(lossGradient: lossGradient, learningRate: learningRate)
            
            // D. Log Progress
            if epoch % 200 == 0 || epoch == 1 {
                var totalLoss: Float = 0
                let ptrOut = output.pointer()
                for i in 0..<batchSize {
                    totalLoss += 0.5 * pow(ptrOut[i] - ptrTar[i], 2)
                }
                print("Epoch \(epoch): Loss = \(String(format: "%.6f", totalLoss / Float(batchSize)))")
            }
        }
        
        // 4. Final Inference
        print("\nFinal XOR Predictions:")
        let finalOutput = model.forward(input: inputs)
        let res = finalOutput.pointer()
        print("[0, 0] -> \(String(format: "%.4f", res[0])) (Target: 0.0)")
        print("[0, 1] -> \(String(format: "%.4f", res[1])) (Target: 1.0)")
        print("[1, 0] -> \(String(format: "%.4f", res[2])) (Target: 1.0)")
        print("[1, 1] -> \(String(format: "%.4f", res[3])) (Target: 0.0)")
        
        print("\nGoal #4 Complete: Non-Linear XOR Solved natively on Apple GPU!")
    }
}
