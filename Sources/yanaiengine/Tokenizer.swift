import Foundation

/// BPE (Byte-Pair Encoding) Tokenizer.
/// Reads a HuggingFace `tokenizer.json` vocabulary and performs BPE encoding/decoding.
public class Tokenizer {
    
    /// Token string → ID
    public var vocab: [String: UInt32] = [:]
    
    /// ID → Token string
    public var reverseVocab: [UInt32: String] = [:]
    
    /// BPE merge rules: (pair) → priority (lower = merge first)
    public var merges: [(String, String)] = []
    
    public var vocabSize: Int { return vocab.count }
    
    public init() {}
    
    /// Load vocabulary and merges from a HuggingFace tokenizer.json file.
    public func load(from path: String) throws {
        let data = try Data(contentsOf: URL(fileURLWithPath: path))
        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw TokenizerError.invalidFormat
        }
        
        // Extract vocabulary from model.vocab
        if let model = json["model"] as? [String: Any],
           let vocabDict = model["vocab"] as? [String: Int] {
            for (token, id) in vocabDict {
                let uid = UInt32(id)
                vocab[token] = uid
                reverseVocab[uid] = token
            }
        }
        
        // Extract merge rules from model.merges
        if let model = json["model"] as? [String: Any],
           let mergeList = model["merges"] as? [String] {
            for mergeStr in mergeList {
                let parts = mergeStr.split(separator: " ", maxSplits: 1)
                if parts.count == 2 {
                    merges.append((String(parts[0]), String(parts[1])))
                }
            }
        }
    }
    
    /// Load a simple vocabulary from a dictionary (for testing).
    public func loadSimple(tokens: [String]) {
        for (i, token) in tokens.enumerated() {
            let uid = UInt32(i)
            vocab[token] = uid
            reverseVocab[uid] = token
        }
    }
    
    /// Encode a string into token IDs using BPE.
    public func encode(text: String) -> [UInt32] {
        // Step 1: Convert text to initial tokens (individual characters/bytes)
        var tokens: [String] = text.unicodeScalars.map { String($0) }
        
        if tokens.isEmpty { return [] }
        
        // Step 2: Iteratively apply BPE merges
        for (left, right) in merges {
            var i = 0
            while i < tokens.count - 1 {
                if tokens[i] == left && tokens[i + 1] == right {
                    tokens[i] = left + right
                    tokens.remove(at: i + 1)
                } else {
                    i += 1
                }
            }
        }
        
        // Step 3: Map tokens to IDs
        var ids: [UInt32] = []
        for token in tokens {
            if let id = vocab[token] {
                ids.append(id)
            } else {
                // Unknown token: try byte-level fallback
                for byte in token.utf8 {
                    let byteToken = String(format: "<0x%02X>", byte)
                    if let id = vocab[byteToken] {
                        ids.append(id)
                    }
                    // Skip truly unknown tokens
                }
            }
        }
        
        return ids
    }
    
    /// Decode token IDs back to a string.
    public func decode(ids: [UInt32]) -> String {
        var result = ""
        for id in ids {
            if let token = reverseVocab[id] {
                result += token
            }
        }
        return result
    }
}

public enum TokenizerError: Error {
    case invalidFormat
    case fileNotFound
}
