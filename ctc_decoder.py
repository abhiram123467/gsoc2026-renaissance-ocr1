import torch

class CTCDecoder:
    """
    Decodes the raw emission probabilities from the CRNN into readable text 
    using Greedy CTC Decoding.
    """
    def __init__(self, vocab_encoder, blank_id=0):
        self.vocab_encoder = vocab_encoder
        self.blank_id = blank_id

    def greedy_decode(self, emissions):
        """
        Args:
            emissions: Tensor of shape (Time, Batch, Classes) from CRNN.
        Returns:
            List of decoded string texts.
        """
        # Get the most likely class at each time step
        predictions = torch.argmax(emissions, dim=-1) # Shape: (Time, Batch)
        
        # Transpose to (Batch, Time) for easier iteration
        predictions = predictions.transpose(0, 1) 

        decoded_texts = []
        for i in range(predictions.size(0)):
            pred_sequence = predictions[i].tolist()
            
            collapsed = []
            prev_char = None
            
            # CTC Rules: Collapse sequential repeats, then remove blanks
            for char_id in pred_sequence:
                if char_id != prev_char:
                    if char_id != self.blank_id:
                        collapsed.append(char_id)
                prev_char = char_id
                
            # Use Issue #2's VocabEncoder to convert IDs back to text
            text = self.vocab_encoder.decode(collapsed)
            decoded_texts.append(text)
            
        return decoded_texts

if __name__ == "__main__":
    print("CTC Decoder successfully initialized.")