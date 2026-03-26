"""
Issue #3: CRNN Backbone Model
GSoC 2026 - RenAIssance OCR Project
CNN feature extractor + Bidirectional RNN for sequence recognition
"""

import torch
import torch.nn as nn


class CRNN(nn.Module):
    """
    CRNN: Convolutional Recurrent Neural Network for OCR.

    Architecture:
        CNN  → extracts visual features (B, C, H, W)
        Map  → collapses height dimension (B, W, C)
        RNN  → models sequence context (B, W, num_classes)
        FC   → projects to vocabulary size

    Input:  (B, 1, 32, W)  — grayscale, fixed H=32, variable W
    Output: (W', B, num_classes) — CTC-compatible format
    """

    def __init__(self, num_classes: int, hidden_size: int = 256):
        """
        Args:
            num_classes: vocab size including CTC blank (from VocabEncoder.num_classes)
            hidden_size: RNN hidden units (default 256)
        """
        super(CRNN, self).__init__()
        self.num_classes = num_classes
        self.hidden_size = hidden_size

        # ── CNN Backbone ─────────────────────────────────────────────────────
        self.cnn = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 64, kernel_size=3, padding=1),   # (B,64,32,W)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                            # (B,64,16,W/2)

            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1), # (B,128,16,W/2)
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                            # (B,128,8,W/4)

            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),# (B,256,8,W/4)
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            # Block 4
            nn.Conv2d(256, 256, kernel_size=3, padding=1),# (B,256,8,W/4)
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (2, 1)),                  # (B,256,4,W/4)

            # Block 5
            nn.Conv2d(256, 512, kernel_size=3, padding=1),# (B,512,4,W/4)
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            # Block 6
            nn.Conv2d(512, 512, kernel_size=3, padding=1),# (B,512,4,W/4)
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (2, 1)),                  # (B,512,2,W/4)

            # Block 7 - collapse height to 1
            nn.Conv2d(512, 512, kernel_size=2, padding=0), # (B,512,1,W/4-1)
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        # ── Bidirectional LSTM ────────────────────────────────────────────────
        self.rnn = nn.LSTM(
            input_size=512,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.1,
        )

        # ── Output Projection ─────────────────────────────────────────────────
        self.fc = nn.Linear(hidden_size * 2, num_classes)  # *2 for bidirectional

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 1, 32, W) input image tensor
        Returns:
            (T, B, num_classes) log-softmax output for CTC loss
            where T = sequence length (W dimension after CNN)
        """
        # CNN features
        features = self.cnn(x)              # (B, 512, 1, W')

        # Remove height dimension (it's 1 after CNN)
        B, C, H, W = features.size()
        assert H == 1, f"Expected H=1 after CNN, got H={H}"
        features = features.squeeze(2)      # (B, 512, W')
        features = features.permute(0, 2, 1)  # (B, W', 512)

        # RNN sequence modelling
        rnn_out, _ = self.rnn(features)     # (B, W', hidden*2)

        # Project to vocab
        logits = self.fc(rnn_out)           # (B, W', num_classes)

        # CTC expects (T, B, num_classes)
        logits = logits.permute(1, 0, 2)    # (W', B, num_classes)

        return torch.log_softmax(logits, dim=2)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── Quick self-test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    num_classes = 96  # from VocabEncoder.num_classes

    model = CRNN(num_classes=num_classes, hidden_size=256)
    print(model)
    print(f"\nTrainable parameters: {model.count_parameters():,}")

    # Simulate a batch of images (B=4, C=1, H=32, W=128)
    dummy_input = torch.randn(4, 1, 32, 128)
    output = model(dummy_input)

    print(f"\nInput  shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}  (T, B, num_classes)")
    assert output.shape[1] == 4, "Batch size mismatch!"
    assert output.shape[2] == num_classes, "Class count mismatch!"
    print("✅ CRNN forward pass test passed!")
