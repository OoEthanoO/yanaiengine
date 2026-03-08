import Foundation
import Metal

@main
struct yanaiengine {
    static func main() {
        print("==============================================")
        print("Starting YanAIEngine: Goal #2 (Forward Pass)")
        print("==============================================\n")
        
        // Initialize Metal Engine
        guard let engine = MetalEngine() else {
            fatalError("Failed to initialize Metal Engine")
        }
        
        // Load all required kernels for Goal #2
        print("Loading Metal kernels...")
        engine.loadLibrary(resourceName: "gemm", kernelName: "gemm_kernel")
        engine.loadLibrary(resourceName: "gemm", kernelName: "bias_add_kernel")
        engine.loadLibrary(resourceName: "gemm", kernelName: "relu_kernel")
        
        // Define dimensions
        let batchSize = 2
        let inputDim = 3
        let outputDim = 2
        
        // 1. Setup Input Tensor
        print("\nPreparing input tensor labels...")
        let input = Tensor(device: engine.device, rows: batchSize, cols: inputDim)
        let ptrIn = input.pointer()
        // Input: [ [1, 2, 3], [-4, -5, -6] ]
        ptrIn[0] = 1; ptrIn[1] = 2; ptrIn[2] = 3
        ptrIn[3] = -4; ptrIn[4] = -5; ptrIn[5] = -6
        
        print("Input Tensor (2x3):")
        input.printMatrix()
        
        // 2. Initialize Linear Layer
        print("Initializing Linear Layer (Weights: 3x2, Bias: 1x2)...")
        let linear = LinearLayer(engine: engine, inputDim: inputDim, outputDim: outputDim, batchSize: batchSize)
        
        // Set deterministic weights for verification
        // Weight matrix: [ [0.5, -0.5], [1.0, -1.0], [1.5, -1.5] ]
        let ptrW = linear.weights.pointer()
        ptrW[0] = 0.5; ptrW[1] = -0.5
        ptrW[2] = 1.0; ptrW[3] = -1.0
        ptrW[4] = 1.5; ptrW[5] = -1.5
        
        // Set deterministic bias: [ 0.1, 0.1 ]
        let ptrB = linear.bias.pointer()
        ptrB[0] = 0.1; ptrB[1] = 0.1
        
        print("Weights (3x2):")
        linear.weights.printMatrix()
        print("Bias (1x2):")
        linear.bias.printMatrix()
        
        // 3. Execute Forward Pass (Chained on GPU)
        print("Executing Forward Pass: ReLU(X * W + b)...")
        linear.forward(input: input)
        
        // 4. Verify Results
        print("\nForward Pass Results (CPU Pointer):")
        linear.output.printMatrix()
        
        // Math Verification:
        // Row 1: [1, 2, 3] * [weights] = [ 1*0.5+2*1+3*1.5, 1*-0.5+2*-1+3*-1.5 ] = [ 0.5+2+4.5, -0.5-2-4.5 ] = [ 7, -7 ]
        // Add Bias: [ 7+0.1, -7+0.1 ] = [ 7.1, -6.9 ]
        // ReLU: [ 7.1, 0.0 ]
        
        // Row 2: [-4, -5, -6] * [weights] = [ -4*0.5-5*1-6*1.5, -4*-0.5-5*-1-6*-1.5 ] = [ -2-5-9, 2+5+9 ] = [ -16, 16 ]
        // Add Bias: [ -16+0.1, 16+0.1 ] = [ -15.9, 16.1 ]
        // ReLU: [ 0.0, 16.1 ]
        
        print("Expected Results:")
        print("[7.1000, 0.0000]")
        print("[0.0000, 16.1000]")
        print("---")
    }
}
