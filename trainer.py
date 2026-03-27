import torch
import torch.nn as nn
import torch.optim as optim

class OCRTrainer:
    """
    Orchestrates the training loop for the CRNN model using CTC Loss.
    """
    def __init__(self, model, vocab_size, learning_rate=0.001, blank_id=0):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # PyTorch's built-in CTC Loss
        # zero_infinity=True prevents infinite loss when targets are too long
        self.criterion = nn.CTCLoss(blank=blank_id, zero_infinity=True)

    def train_step(self, images, targets, target_lengths):
        """
        Executes a single forward and backward pass.
        Args:
            images: Batch of image tensors (B, C, H, W)
            targets: Flattened tensor of target indices
            target_lengths: Tensor containing the length of each target sequence
        """
        self.model.train()
        self.optimizer.zero_grad()

        # Forward pass: shape (Time, Batch, Classes)
        emissions = self.model(images.to(self.device))
        
        # CTC Loss requires input_lengths (Time for each sequence in batch)
        # Assuming emissions shape is (T, B, C)
        input_lengths = torch.full(
            size=(emissions.size(1),), 
            fill_value=emissions.size(0), 
            dtype=torch.long
        )

        # Calculate loss
        loss = self.criterion(
            emissions.log_softmax(2), 
            targets.to(self.device), 
            input_lengths, 
            target_lengths.to(self.device)
        )

        # Backpropagation
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients in LSTMs
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0)
        
        self.optimizer.step()

        return loss.item()

if __name__ == "__main__":
    print("Trainer module ready for integration.")