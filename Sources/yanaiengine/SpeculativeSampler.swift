import Foundation

/// Core logic for Speculative Decoding (The Draft-Verify Pipeline).
public class SpeculativeSampler {
    
    public enum SamplingResult {
        /// Target model agreed with the draft model's probability distribution
        case accept
        /// Target model rejected the drafted token; returns a resampled token
        case reject(newToken: UInt32)
    }
    
    /// Rejection sampling algorithm to verify a token drafted by a smaller model.
    ///
    /// - Parameters:
    ///   - draftedToken: The token id predicted by the draft model.
    ///   - draftProbs: The probability distribution of the draft model.
    ///   - targetProbs: The probability distribution of the target model.
    /// - Returns: `SamplingResult` accepting the token or rejecting with a corrected token.
    public static func verifyToken(
        draftedToken: UInt32,
        draftProbs: [Float],
        targetProbs: [Float]
    ) -> SamplingResult {
        let p_x = draftProbs[Int(draftedToken)]
        let q_x = targetProbs[Int(draftedToken)]
        
        let ratio = q_x / p_x
        let r = Float.random(in: 0..<1)
        
        // 1. Acceptance test: Did target model basically agree?
        if r < ratio {
            return .accept
        }
        
        // 2. Rejection: We must sample a new token from an adjusted distribution
        var adjustedProbs = [Float](repeating: 0, count: targetProbs.count)
        var sum: Float = 0
        
        for i in 0..<targetProbs.count {
            // Adjusted probability: max(0, q(x) - p(x))
            let diff = max(0, targetProbs[i] - draftProbs[i])
            adjustedProbs[i] = diff
            sum += diff
        }
        
        // 3. Normalize adjusted distribution
        if sum > 0 {
            for i in 0..<adjustedProbs.count {
                adjustedProbs[i] /= sum
            }
        } else {
            // Math fallback (shouldn't happen if distributions are valid and p_x rejected)
            adjustedProbs = targetProbs
        }
        
        // 4. Sample corrected token
        let newToken = sampleFromProbs(adjustedProbs)
        return .reject(newToken: newToken)
    }
    
    /// Basic multinomial sampling from a probability distribution.
    public static func sampleFromProbs(_ probs: [Float]) -> UInt32 {
        let r = Float.random(in: 0..<1)
        var cdf: Float = 0
        for i in 0..<probs.count {
            cdf += probs[i]
            if r <= cdf {
                return UInt32(i)
            }
        }
        return UInt32(probs.count - 1) // Fallback to last token
    }
    
    /// Utility to convert unnormalized logits to probabilities safely.
    public static func softmax(logits: [Float]) -> [Float] {
        guard let maxLogit = logits.max() else { return logits.map { _ in 0 } }
        var probs = [Float](repeating: 0, count: logits.count)
        var sum: Float = 0
        for i in 0..<logits.count {
            let expVal = exp(logits[i] - maxLogit)
            probs[i] = expVal
            sum += expVal
        }
        if sum > 0 {
            for i in 0..<logits.count {
                probs[i] /= sum
            }
        }
        return probs
    }
    
    /// Process a sequence of draft tokens against target logits.
    /// Returns the final accepted sequence of tokens.
    public static func verifySequence(
        draftedTokens: [UInt32],
        draftProbs: [[Float]],
        targetLogits: [[Float]],
        vocabSize: Int
    ) -> [UInt32] {
        var acceptedSequence: [UInt32] = []
        
        // We verify each token sequentially up to the point of rejection
        for i in 0..<draftedTokens.count {
            let token = draftedTokens[i]
            let p_probs = draftProbs[i]
            let q_probs = softmax(logits: targetLogits[i])
            
            let result = verifyToken(draftedToken: token, draftProbs: p_probs, targetProbs: q_probs)
            
            switch result {
            case .accept:
                acceptedSequence.append(token)
            case .reject(let newToken):
                acceptedSequence.append(newToken)
                // Stop verification exactly at the first rejected token
                return acceptedSequence
            }
        }
        
        // If all draft tokens accepted, we must still sample the FINAL token (the n+1 token) 
        // from the last output of the target model to continue the chain.
        let finalLogits = targetLogits.last!
        let finalProbs = softmax(logits: finalLogits)
        let finalToken = sampleFromProbs(finalProbs)
        acceptedSequence.append(finalToken)
        
        return acceptedSequence
    }
}
