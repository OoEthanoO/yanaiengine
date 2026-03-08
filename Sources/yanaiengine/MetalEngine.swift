import Metal
import Foundation

/// Handles the Metal initialization (GPU Handshake)
public class MetalEngine {
    public let device: MTLDevice
    public let commandQueue: MTLCommandQueue
    private var pipelineStates: [String: MTLComputePipelineState] = [:]
    
    public init?() {
        // Step 2: Grab the default GPU
        guard let defaultDevice = MTLCreateSystemDefaultDevice() else {
            print("Metal is not supported on this device")
            return nil
        }
        self.device = defaultDevice
        
        // Step 2: Set up the command queue (the highway for instructions)
        guard let queue = device.makeCommandQueue() else {
            print("Failed to create command queue")
            return nil
        }
        self.commandQueue = queue
    }
    
    /// Step 2, 3 & 5: Loads the Metal library using a multi-stage search strategy.
    /// This ensures compatibility across CLI (terminal) and App (Xcode) environments.
    public func loadLibrary(resourceName: String = "gemm", kernelName: String = "gemm_kernel") {
        do {
            var library: MTLLibrary?
            
            // 1. Try to find the compiled library in the module's bundle (Ideal for Xcode/Bundled Apps)
            if let libURL = Bundle.module.url(forResource: "default", withExtension: "metallib") {
                library = try? device.makeLibrary(URL: libURL)
            }
            
            // 2. Try to find the source file in the bundle (Standard for SPM Resources)
            if library == nil {
                if let sourcePath = Bundle.module.path(forResource: resourceName, ofType: "metal") {
                    let source = try String(contentsOfFile: sourcePath)
                    library = try? device.makeLibrary(source: source, options: nil)
                }
            }
            
            // 3. Fallback to hardcoded relative paths (For local CLI development without bundles)
            if library == nil {
                let fallbackPaths = [
                    "Sources/yanaiengine/\(resourceName).metal",
                    "\(resourceName).metal"
                ]
                for path in fallbackPaths {
                    if FileManager.default.fileExists(atPath: path) {
                        let source = try String(contentsOfFile: path)
                        library = try? device.makeLibrary(source: source, options: nil)
                        if library != nil { break }
                    }
                }
            }
            
            // 4. Ultimate fallback to standard makeDefaultLibrary()
            if library == nil {
                library = device.makeDefaultLibrary()
            }
            
            guard let finalLibrary = library else {
                fatalError("ERROR: Could not find Metal shader '\(resourceName)' (tried Bundle.module, local paths, and makeDefaultLibrary).")
            }
            
            guard let function = finalLibrary.makeFunction(name: kernelName) else {
                fatalError("ERROR: Failed to find Metal function: '\(kernelName)' in library.")
            }
            
            let pso = try device.makeComputePipelineState(function: function)
            self.pipelineStates[kernelName] = pso
            print("Successfully initialized '\(kernelName)' kernel.")
        } catch {
            fatalError("CRITICAL: Metal loading error: \(error)")
        }
    }
    
    /// Returns a cached pipeline state for the given kernel name.
    public func getPipelineState(name: String) -> MTLComputePipelineState {
        guard let pso = pipelineStates[name] else {
            fatalError("Pipeline state for '\(name)' has not been loaded. Call loadLibrary() first.")
        }
        return pso
    }
}
