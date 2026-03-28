"""
Example usage of the OCR DataLoader
Demonstrates how to prepare data and create batches for training
"""

from dataloader import create_dataloader
from PIL import Image
import numpy as np
import os


def prepare_sample_dataset(num_samples=10):
    """
    Create sample OCR images for demonstration
    In a real project, you would load your actual dataset here
    """
    print("Creating sample dataset...")
    
    # Create temporary directory for images
    os.makedirs("sample_data", exist_ok=True)
    
    image_paths = []
    labels = []
    
    # Sample texts of varying lengths
    sample_texts = [
        "hello", "world", "OCR", "recognition",
        "test", "sample", "data", "PyTorch",
        "deep", "learning"
    ]
    
    for i in range(min(num_samples, len(sample_texts))):
        text = sample_texts[i]
        
        # Create image with width proportional to text length
        # In real OCR, image width varies based on actual text content
        width = len(text) * 25 + np.random.randint(-10, 10)
        height = 32
        
        # Create random grayscale image (simulating text)
        img_array = np.random.randint(0, 255, (height, width), dtype=np.uint8)
        
        # Add some "text-like" patterns (horizontal strokes)
        for y in range(10, 22, 4):
            img_array[y:y+2, 5:-5] = np.random.randint(0, 100, width-10)
        
        img = Image.fromarray(img_array, mode='L')
        
        # Save image
        img_path = f"sample_data/img_{i:03d}.png"
        img.save(img_path)
        
        image_paths.append(img_path)
        labels.append(text)
    
    return image_paths, labels


def main():
    """Main demonstration"""
    
    # Step 1: Prepare dataset
    image_paths, labels = prepare_sample_dataset(num_samples=10)
    print(f"Created {len(image_paths)} sample images")
    print(f"Labels: {labels[:5]}...")
    
    # Step 2: Create DataLoader
    print("\n" + "="*60)
    print("Creating DataLoader with batch_size=3")
    print("="*60)
    
    dataloader = create_dataloader(
        image_paths=image_paths,
        labels=labels,
        batch_size=3,
        height=32,
        shuffle=False,  # Don't shuffle for demo clarity
        num_workers=0   # Use 0 for single-process (easier debugging)
    )
    
    # Step 3: Iterate through batches
    print("\nIterating through batches:")
    print("-" * 60)
    
    for batch_idx, batch in enumerate(dataloader):
        images = batch['images']
        labels_batch = batch['labels']
        widths = batch['lengths']
        
        print(f"\nBatch {batch_idx + 1}:")
        print(f"  ├─ Images shape: {images.shape}")
        print(f"  ├─ Labels: {labels_batch}")
        print(f"  ├─ Original widths: {widths.tolist()}")
        print(f"  └─ Max width (W_max): {images.shape[3]}")
        
        # Show per-image details
        for i, (label, width) in enumerate(zip(labels_batch, widths)):
            padding = images.shape[3] - width.item()
            print(f"      Image {i}: '{label}' | width={width.item()} | padding={padding}")
    
    # Step 4: Summary statistics
    print("\n" + "="*60)
    print("Summary:")
    print("="*60)
    print(f"Total images: {len(image_paths)}")
    print(f"Total batches: {len(dataloader)}")
    print(f"Batch size: 3")
    print(f"Fixed height: 32 pixels")
    print(f"Variable widths: {[len(text)*25 for text in labels[:5]]}... (approx)")
    print("\n✓ DataLoader ready for training!")
    
    # Cleanup
    print("\nCleaning up sample data...")
    import shutil
    shutil.rmtree("sample_data")
    print("Done!")


if __name__ == "__main__":
    main()
