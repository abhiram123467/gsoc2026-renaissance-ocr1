class OCREvaluator:
    """
    Computes standard OCR evaluation metrics: 
    Character Error Rate (CER) and Word Error Rate (WER).
    """
    
    @staticmethod
    def _levenshtein_distance(ref, hyp):
        """Pure Python Levenshtein distance calculation."""
        if len(ref) < len(hyp):
            return OCREvaluator._levenshtein_distance(hyp, ref)
        if len(hyp) == 0:
            return len(ref)
            
        previous_row = range(len(hyp) + 1)
        for i, c1 in enumerate(ref):
            current_row = [i + 1]
            for j, c2 in enumerate(hyp):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        return previous_row[-1]

    @classmethod
    def calculate_cer(cls, references, hypotheses):
        """
        Calculates Character Error Rate across a batch.
        Lower is better (0.0 = perfect match).
        """
        total_distance = 0
        total_chars = 0
        
        for ref, hyp in zip(references, hypotheses):
            total_distance += cls._levenshtein_distance(ref, hyp)
            total_chars += len(ref)
            
        if total_chars == 0:
            return 0.0
        return total_distance / total_chars

if __name__ == "__main__":
    # Quick sanity check
    ref = ["hello", "world"]
    hyp = ["helo", "word"]
    cer = OCREvaluator.calculate_cer(ref, hyp)
    print(f"Sanity Check CER: {cer:.4f} (Expected: ~0.2000)")