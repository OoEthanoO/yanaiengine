import Foundation

/// Metadata for a single tensor in a .safetensors file.
public struct SafetensorInfo {
    public let name: String
    public let dtype: String        // "F32", "F16", "BF16", "I8", etc.
    public let shape: [Int]
    public let dataOffset: Int      // Byte offset from start of data block
    public let dataLength: Int      // Length in bytes
}

/// Zero-copy .safetensors file reader using POSIX mmap.
///
/// File format:
///   [8 bytes: LE uint64 header_size]
///   [header_size bytes: JSON header]
///   [remaining bytes: raw tensor data]
public class SafetensorsReader {
    
    /// Parsed tensor metadata, keyed by tensor name.
    public private(set) var tensors: [String: SafetensorInfo] = [:]
    
    /// The memory-mapped file pointer
    private var mmapPtr: UnsafeMutableRawPointer?
    private var mmapSize: Int = 0
    private var fileDescriptor: Int32 = -1
    
    /// Offset from file start to where tensor data begins
    private var dataStartOffset: Int = 0
    
    public init() {}
    
    deinit {
        close()
    }
    
    /// Open and parse a .safetensors file using mmap.
    public func open(path: String) throws {
        // Open the file
        fileDescriptor = Darwin.open(path, O_RDONLY)
        guard fileDescriptor >= 0 else {
            throw SafetensorsError.fileNotFound(path)
        }
        
        // Get file size
        var stat = stat()
        guard fstat(fileDescriptor, &stat) == 0 else {
            Darwin.close(fileDescriptor)
            throw SafetensorsError.statFailed
        }
        mmapSize = Int(stat.st_size)
        
        // Memory-map the entire file (read-only, private)
        mmapPtr = mmap(nil, mmapSize, PROT_READ, MAP_PRIVATE, fileDescriptor, 0)
        guard mmapPtr != MAP_FAILED else {
            Darwin.close(fileDescriptor)
            throw SafetensorsError.mmapFailed
        }
        
        // Step 1: Read 8-byte LE uint64 header size
        let headerSizePtr = mmapPtr!.bindMemory(to: UInt64.self, capacity: 1)
        let headerSize = Int(UInt64(littleEndian: headerSizePtr.pointee))
        
        guard headerSize > 0 && headerSize < mmapSize else {
            throw SafetensorsError.invalidHeader
        }
        
        // Step 2: Read the JSON header
        let jsonStart = mmapPtr! + 8
        let jsonData = Data(bytes: jsonStart, count: headerSize)
        
        guard let header = try JSONSerialization.jsonObject(with: jsonData) as? [String: Any] else {
            throw SafetensorsError.invalidJSON
        }
        
        // Data block starts right after the header
        dataStartOffset = 8 + headerSize
        
        // Step 3: Parse tensor metadata
        for (name, value) in header {
            // Skip __metadata__ key
            if name == "__metadata__" { continue }
            
            guard let info = value as? [String: Any],
                  let dtype = info["dtype"] as? String,
                  let shape = info["shape"] as? [Int],
                  let offsets = info["data_offsets"] as? [Int],
                  offsets.count == 2 else {
                continue
            }
            
            let byteStart = offsets[0]
            let byteEnd = offsets[1]
            
            tensors[name] = SafetensorInfo(
                name: name,
                dtype: dtype,
                shape: shape,
                dataOffset: byteStart,
                dataLength: byteEnd - byteStart
            )
        }
    }
    
    /// Get a read-only pointer to the raw bytes of a named tensor.
    /// Returns nil if tensor not found.
    public func tensorData(name: String) -> UnsafeRawPointer? {
        guard let info = tensors[name], let ptr = mmapPtr else { return nil }
        return UnsafeRawPointer(ptr + dataStartOffset + info.dataOffset)
    }
    
    /// Get tensor info by name.
    public func tensorInfo(name: String) -> SafetensorInfo? {
        return tensors[name]
    }
    
    /// Close the mmap and file descriptor.
    public func close() {
        if let ptr = mmapPtr, ptr != MAP_FAILED {
            munmap(ptr, mmapSize)
            mmapPtr = nil
        }
        if fileDescriptor >= 0 {
            Darwin.close(fileDescriptor)
            fileDescriptor = -1
        }
    }
}

/// Errors from safetensors parsing.
public enum SafetensorsError: Error, CustomStringConvertible {
    case fileNotFound(String)
    case statFailed
    case mmapFailed
    case invalidHeader
    case invalidJSON
    case tensorNotFound(String)
    
    public var description: String {
        switch self {
        case .fileNotFound(let path): return "File not found: \(path)"
        case .statFailed: return "Failed to stat file"
        case .mmapFailed: return "mmap failed"
        case .invalidHeader: return "Invalid safetensors header"
        case .invalidJSON: return "Invalid JSON in header"
        case .tensorNotFound(let name): return "Tensor '\(name)' not found"
        }
    }
}
