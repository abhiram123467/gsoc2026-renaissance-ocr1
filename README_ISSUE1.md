# Issue #1: Dynamic-Padding DataLoader for OCR

Implementation of a PyTorch DataLoader that handles variable-width OCR line images with dynamic padding.

## Features

✅ **Dynamic Padding**: Automatically pads images to the maximum width in each batch  
✅ **Fixed Height**: Maintains H=32 pixels for all images  
✅ **Aspect Ratio Preservation**: Resizes images while preserving aspect ratio  
✅ **Grayscale Processing**: Converts all images to single-channel grayscale  
✅ **Normalization**: Applies standard normalization (mean=0.5, std=0.5)  
✅ **Efficient Batching**: Zero-copy padding with left-aligned images  

## Output Format

Each batch returns a dictionary with:
- **`images`**: Tensor of shape `(B, 1, H=32, W_max)` - padded images
- **`labels`**: List of text strings (ground truth)
- **`lengths`**: Tensor of original widths before padding `(B,)`

## Usage

### Basic Usage

```python
from dataloader import create_dataloader

# Prepare your data
image_paths = ["path/to/img1.png", "path/to/img2.png", ...]
labels = ["hello", "world", ...]

# Create DataLoader
dataloader = create_dataloader(
    image_paths=image_paths,
    labels=labels,
    batch_size=32,
    height=32,
    shuffle=True,
    num_workers=4
)

# Iterate through batches
for batch in dataloader:
    images = batch['images']      # (B, 1, 32, W_max)
    labels = batch['labels']       # List of strings
    widths = batch['lengths']      # (B,) tensor
    
    # Your training code here
    # ...
```

### Advanced Options

```python
dataloader = create_dataloader(
    image_paths=image_paths,
    labels=labels,
    batch_size=32,
    height=32,              # Fixed height (default: 32)
    shuffle=True,           # Shuffle data (default: True)
    num_workers=4,          # Parallel workers (default: 4)
    augment=False           # Data augmentation (default: False)
)
```

## Implementation Details

### Image Processing Pipeline

1. **Load Image**: Opens image file and converts to grayscale (1 channel)
2. **Resize**: Scales image to height=32 while preserving aspect ratio
3. **Normalize**: Applies (pixel/255 - mean) / std normalization
4. **Tensor Conversion**: Converts to PyTorch tensor with shape (1, H, W)

### Dynamic Padding Strategy

- Images are **left-aligned** in the batch
- Right side is **padded with zeros** to match max width
- Each batch has different W_max based on its widest image
- Original widths are preserved in `lengths` tensor for downstream processing

### Why This Design?

- **Memory Efficient**: Only pads to the max width in each batch, not global max
- **CTC Compatible**: Provides original widths needed for CTC loss computation
- **Variable Speed**: Wider batches process slower, narrower batches faster
- **No Information Loss**: Preserves original aspect ratios

## Example Output

```
Batch with 4 images:
├─ Image 1: width=100 → padded to 150
├─ Image 2: width=150 → padded to 150
├─ Image 3: width=80  → padded to 150
└─ Image 4: width=120 → padded to 150

Output shape: (4, 1, 32, 150)
Lengths tensor: [100, 150, 80, 120]
```

## Testing

Run the built-in tests:

```bash
python dataloader.py
```

This will create dummy images with different widths and verify:
- Correct padding behavior
- Proper shape outputs
- Length preservation

## Next Steps (Remaining Issues)

- [ ] Issue #2: Vocabulary/charset encoder
- [ ] Issue #3: CRNN backbone model
- [ ] Issue #4: CTC greedy decoder
- [ ] Issue #5: Training engine
- [ ] Issue #6: CER evaluation metric

## Requirements

- Python 3.8+
- PyTorch 1.9+
- Pillow
- NumPy

See `requirements.txt` for exact versions.
