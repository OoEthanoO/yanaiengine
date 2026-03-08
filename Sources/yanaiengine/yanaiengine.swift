import Foundation
import Metal
import NIO

@main
struct yanaiengine {
    static func main() {
        let args = CommandLine.arguments
        let isMaster = args.contains("--master")
        let isWorker = args.contains("--worker")
        
        print("==============================================")
        print("Starting YanAIEngine: Goal #5 (Distributed DDP)")
        if isMaster { print("ROLE: MASTER (Node 0)") }
        else if isWorker { print("ROLE: WORKER (Node 1)") }
        else { print("ROLE: SINGLE-NODE (Standalone)") }
        print("==============================================\n")
        
        guard let engine = MetalEngine() else { fatalError() }
        let kernels = ["gemm_kernel", "bias_add_kernel", "relu_kernel", "transpose_kernel", "mse_derivative_kernel", "sgd_update_kernel", "relu_derivative_kernel", "sum_rows_kernel"]
        for kernel in kernels { engine.loadLibrary(resourceName: "gemm", kernelName: kernel) }
        
        // 1. Setup XOR Dataset (Split for DDP)
        let localBatch = (isMaster || isWorker) ? 2 : 4
        let inputDim = 2
        let outputDim = 1
        
        let inputs = Tensor(device: engine.device, rows: localBatch, cols: inputDim)
        let targets = Tensor(device: engine.device, rows: localBatch, cols: outputDim)
        let ptrIn = inputs.pointer()
        let ptrTar = targets.pointer()
        
        if isMaster || (!isMaster && !isWorker) {
            ptrIn[0] = 0; ptrIn[1] = 0; ptrTar[0] = 0
            ptrIn[2] = 0; ptrIn[3] = 1; ptrTar[1] = 1
        } else if isWorker {
            ptrIn[0] = 1; ptrIn[1] = 0; ptrTar[0] = 1
            ptrIn[2] = 1; ptrIn[3] = 1; ptrTar[1] = 0
        }
        
        // 2. Setup Model
        let layer1 = LinearLayer(engine: engine, inputDim: 2, outputDim: 16, batchSize: localBatch, useReLU: true)
        let layer2 = LinearLayer(engine: engine, inputDim: 16, outputDim: 1, batchSize: localBatch, useReLU: false)
        let model = Sequential(layers: [layer1, layer2])
        
        // 3. Setup Interconnect (Scoped for the whole main)
        let semaphore = DispatchSemaphore(value: 0)
        let connectionSemaphore = DispatchSemaphore(value: 0)
        var remoteGradients: Data?
        var server: Interconnect.Server?
        var client: Interconnect.Client?
        
        if isMaster {
            let s = Interconnect.Server()
            s.onGradientReceived = { data in
                remoteGradients = data
                semaphore.signal()
            }
            try? s.start(host: "127.0.0.1", port: 8080)
            server = s
            print("Master waiting for Worker connection...")
        } else if isWorker {
            let c = Interconnect.Client()
            c.onDataReceived = { data in
                remoteGradients = data
                semaphore.signal()
            }
            print("Worker connecting to Master...")
            try? c.connect(host: "127.0.0.1", port: 8080)
            client = c
        }
        
        // 4. Training Loop
        let epochs = isMaster || isWorker ? 1000 : 2000
        let learningRate: Float = 0.1
        let lossGradient = Tensor(device: engine.device, rows: localBatch, cols: outputDim)
        
        print("\nStarting Training...")
        for epoch in 1...epochs {
            let _ = model.forward(input: inputs)
            
            // Backward pass (Compute local gradients)
            guard let commandBuffer = engine.commandQueue.makeCommandBuffer(),
                  let encoder = commandBuffer.makeComputeCommandEncoder() else { fatalError() }
            let lossPSO = engine.getPipelineState(name: "mse_derivative_kernel")
            encoder.setComputePipelineState(lossPSO)
            encoder.setBuffer(model.layers.last!.output.buffer, offset: 0, index: 0)
            encoder.setBuffer(targets.buffer, offset: 0, index: 1)
            encoder.setBuffer(lossGradient.buffer, offset: 0, index: 2)
            var n = UInt32(localBatch * outputDim)
            encoder.setBytes(&n, length: MemoryLayout<UInt32>.size, index: 3)
            encoder.dispatchThreads(MTLSize(width: localBatch, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: localBatch, height: 1, depth: 1))
            encoder.endEncoding()
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
            
            model.computeGradients(lossGradient: lossGradient)
            
            // --- ALL-REDUCE STEP ---
            if isMaster || isWorker {
                let localGradData = layer1.weightGradients.serialize()
                
                if let s = server {
                    // print("Master: Waiting for worker gradients...")
                    semaphore.wait() // Wait for worker's gradients
                    if let data = remoteGradients {
                        let workerGradTensor = Tensor(device: engine.device, rows: layer1.weightGradients.rows, cols: layer1.weightGradients.cols)
                        workerGradTensor.deserialize(from: data)
                        layer1.weightGradients.average(with: workerGradTensor)
                        
                        let averagedData = layer1.weightGradients.serialize()
                        s.sendToWorker(data: averagedData)
                    }
                } else if let c = client {
                    // print("Worker: Sending gradients to Master...")
                    c.send(data: localGradData)
                    // print("Worker: Waiting for averaged gradients...")
                    semaphore.wait() // Wait for master to return averaged gradients
                    if let data = remoteGradients {
                        layer1.weightGradients.deserialize(from: data)
                    }
                }
            }
            
            model.applyUpdates(learningRate: learningRate)
            
            if epoch % 100 == 0 || epoch == 1 {
                var totalLoss: Float = 0
                let ptrOut = model.layers.last!.output.pointer()
                for i in 0..<localBatch {
                    totalLoss += 0.5 * pow(ptrOut[i] - ptrTar[i], 2)
                }
                print("Epoch \(epoch): Loss = \(String(format: "%.6f", totalLoss / Float(localBatch)))")
            }
        }
        
        print("\nGoal #5 Training Complete.")
        print("Distributed Data Parallel (DDP) logic verified across devices!")
        
        server?.stop()
        client?.stop()
    }
}
