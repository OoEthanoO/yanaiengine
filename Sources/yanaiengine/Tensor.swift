import Metal
import Foundation

/// A Tensor structure that uses a Metal Buffer for its underlying memory.
/// This allows zero-copy data sharing between the CPU and the GPU on Apple Silicon.
public struct Tensor {
    public let buffer: MTLBuffer
    public let rows: Int
    public let cols: Int
    
    /// Initializes a new Tensor with the given dimensions allocated on the specified Metal device.
    ///
    /// - Parameters:
    ///   - device: The Metal device to allocate memory on.
    ///   - rows: The number of rows in the matrix.
    ///   - cols: The number of columns in the matrix.
    public init(device: MTLDevice, rows: Int, cols: Int) {
        self.rows = rows
        self.cols = cols
        let byteCount = rows * cols * MemoryLayout<Float>.stride
        guard let buffer = device.makeBuffer(length: byteCount, options: .storageModeShared) else {
            fatalError("Failed to allocate an MTLBuffer of size \(byteCount)")
        }
        self.buffer = buffer
    }
    
    public init(device: MTLDevice, rows: Int, cols: Int, existingBuffer: MTLBuffer) {
        self.rows = rows
        self.cols = cols
        self.buffer = existingBuffer
    }
    
    /// Returns a typed pointer to the start of the buffer's contents, allowing CPU-side manipulation.
    public func pointer() -> UnsafeMutablePointer<Float> {
        return buffer.contents().bindMemory(to: Float.self, capacity: rows * cols)
    }
    
    /// Fills the entire tensor with a constant value.
    public func fill(with value: Float) {
        let ptr = pointer()
        for i in 0..<(rows * cols) {
            ptr[i] = value
        }
    }
    
    /// Fills the entire tensor with random Float values between -1.0 and 1.0.
    public func fillRandom() {
        let ptr = pointer()
        for i in 0..<(rows * cols) {
            ptr[i] = Float.random(in: -1.0...1.0)
        }
    }
    
    /// Prints the tensor values for easy debugging.
    public func printMatrix() {
        let ptr = pointer()
        for r in 0..<rows {
            var rowString = "["
            for c in 0..<cols {
                let val = ptr[r * cols + c]
                rowString += String(format: "%.4f", val)
                if c < cols - 1 { rowString += ", " }
            }
            rowString += "]"
            print(rowString)
        }
        print("---")
    }
    
    // MARK: - Distributed Utilities
    
    /// Serializes the tensor data into a Data object for network transmission.
    public func serialize() -> Data {
        let byteCount = rows * cols * MemoryLayout<Float>.stride
        return Data(bytes: buffer.contents(), count: byteCount)
    }
    
    /// Deserializes data from a Data object into the tensor's buffer.
    public func deserialize(from data: Data) {
        let byteCount = rows * cols * MemoryLayout<Float>.stride
        guard data.count == byteCount else {
            fatalError("Data size mismatch during deserialization")
        }
        data.withUnsafeBytes { (rawPtr: UnsafeRawBufferPointer) in
            guard let baseAddress = rawPtr.baseAddress else { return }
            memcpy(buffer.contents(), baseAddress, byteCount)
        }
    }
    
    /// Calculates the mean of this tensor and another tensor, storing the result in this tensor.
    /// Used for gradient averaging in Distributed Data Parallel (DDP).
    public func average(with other: Tensor) {
        let ptrSelf = pointer()
        let ptrOther = other.pointer()
        for i in 0..<(rows * cols) {
            ptrSelf[i] = (ptrSelf[i] + ptrOther[i]) / 2.0
        }
    }
}
