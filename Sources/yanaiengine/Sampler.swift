import Foundation

/// Advanced token sampler with Temperature, Top-K, and Top-P (Nucleus) sampling.
public class Sampler {
    public var temperature: Float
    public var topK: Int
    public var topP: Float
    
    public init(temperature: Float = 0.8, topK: Int = 50, topP: Float = 0.9) {
        self.temperature = temperature
        self.topK = topK
        self.topP = topP
    }
    
    /// Sample a token from logits using Temperature + Top-K + Top-P.
    /// Returns the selected token ID.
    public func sample(logits: UnsafeMutablePointer<Float>, vocabSize: Int) -> UInt32 {
        // Step 1: Temperature scaling
        if temperature > 0 && temperature != 1.0 {
            for i in 0..<vocabSize {
                logits[i] /= temperature
            }
        }
        
        // Greedy mode: temperature = 0
        if temperature <= 0 {
            return argmax(logits: logits, vocabSize: vocabSize)
        }
        
        // Step 2: Softmax to get probabilities
        let probs = softmax(logits: logits, vocabSize: vocabSize)
        
        // Step 3: Sort by probability (descending)
        var indexed = (0..<vocabSize).map { (index: UInt32($0), prob: probs[$0]) }
        indexed.sort { $0.prob > $1.prob }
        
        // Step 4: Top-K truncation
        let k = min(topK, vocabSize)
        if k < vocabSize {
            for i in k..<vocabSize {
                indexed[i].prob = 0
            }
        }
        
        // Step 5: Top-P (Nucleus) truncation
        var cumProb: Float = 0
        var cutoffIdx = k
        for i in 0..<k {
            cumProb += indexed[i].prob
            if cumProb >= topP {
                cutoffIdx = i + 1
                break
            }
        }
        for i in cutoffIdx..<vocabSize {
            indexed[i].prob = 0
        }
        
        // Step 6: Renormalize
        var totalProb: Float = 0
        for i in 0..<cutoffIdx { totalProb += indexed[i].prob }
        if totalProb > 0 {
            for i in 0..<cutoffIdx { indexed[i].prob /= totalProb }
        }
        
        // Step 7: Random sample from the distribution
        let r = Float.random(in: 0..<1)
        var cumulative: Float = 0
        for i in 0..<cutoffIdx {
            cumulative += indexed[i].prob
            if r < cumulative {
                return indexed[i].index
            }
        }
        
        // Fallback: return the most probable
        return indexed[0].index
    }
    
    /// Simple argmax (greedy decoding).
    public func argmax(logits: UnsafeMutablePointer<Float>, vocabSize: Int) -> UInt32 {
        var maxVal: Float = -Float.infinity
        var maxIdx: UInt32 = 0
        for i in 0..<vocabSize {
            if logits[i] > maxVal {
                maxVal = logits[i]
                maxIdx = UInt32(i)
            }
        }
        return maxIdx
    }
    
    /// Numerically stable softmax.
    private func softmax(logits: UnsafeMutablePointer<Float>, vocabSize: Int) -> [Float] {
        var maxVal: Float = -Float.infinity
        for i in 0..<vocabSize { if logits[i] > maxVal { maxVal = logits[i] } }
        
        var probs = [Float](repeating: 0, count: vocabSize)
        var sum: Float = 0
        for i in 0..<vocabSize {
            probs[i] = exp(logits[i] - maxVal)
            sum += probs[i]
        }
        for i in 0..<vocabSize { probs[i] /= sum }
        return probs
    }
}
