"""
Dynamic-padding DataLoader for OCR line images
Handles variable-width images with fixed height (H=32)
"""

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from typing import List, Tuple, Optional
import os


class OCRDataset(Dataset):
    """Dataset for OCR line images with variable widths"""
    
    def __init__(
        self,
        image_paths: List[str],
        labels: List[str],
        height: int = 32,
        mean: float = 0.5,
        std: float = 0.5,
        augment: bool = False
    ):
        """
        Args:
            image_paths: List of paths to image files
            labels: List of corresponding text labels
            height: Fixed height for all images (default: 32)
            mean: Normalization mean (default: 0.5)
            std: Normalization std (default: 0.5)
            augment: Whether to apply data augmentation
        """
        assert len(image_paths) == len(labels), "Mismatch between images and labels"
        
        self.image_paths = image_paths
        self.labels = labels
        self.height = height
        self.mean = mean
        self.std = std
        self.augment = augment
        
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str, int]:
        """
        Returns:
            image: Tensor of shape (1, H, W) - grayscale image
            label: Ground truth text string
            width: Original width before padding
        """
        # Load image
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('L')  # Convert to grayscale
        
        # Resize height while maintaining aspect ratio
        original_width, original_height = img.size
        new_width = int(original_width * self.height / original_height)
        img = img.resize((new_width, self.height), Image.BILINEAR)
        
        # Apply augmentation if enabled
        if self.augment:
            img = self._apply_augmentation(img)
        
        # Convert to tensor and normalize
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_tensor = torch.from_numpy(img_array).unsqueeze(0)  # Add channel dim
        img_tensor = (img_tensor - self.mean) / self.std
        
        return img_tensor, self.labels[idx], new_width
    
    def _apply_augmentation(self, img: Image.Image) -> Image.Image:
        """Apply random augmentations to image"""
        # TODO: Add augmentations like brightness, contrast, blur, etc.
        return img


def collate_fn(batch: List[Tuple[torch.Tensor, str, int]]) -> dict:
    """
    Custom collate function for dynamic padding
    
    Args:
        batch: List of (image, label, width) tuples
        
    Returns:
        Dictionary containing:
            - images: Padded tensor of shape (B, 1, H, W_max)
            - labels: List of label strings
            - lengths: Tensor of original widths (B,)
    """
    images, labels, widths = zip(*batch)
    
    # Find max width in batch
    max_width = max(widths)
    batch_size = len(images)
    height = images[0].shape[1]
    
    # Create padded tensor
    padded_images = torch.zeros(batch_size, 1, height, max_width)
    
    # Fill in images (left-aligned, right-padded with zeros)
    for i, (img, width) in enumerate(zip(images, widths)):
        padded_images[i, :, :, :width] = img
    
    # Convert widths to tensor
    lengths = torch.tensor(widths, dtype=torch.long)
    
    return {
        'images': padded_images,
        'labels': labels,
        'lengths': lengths
    }


def create_dataloader(
    image_paths: List[str],
    labels: List[str],
    batch_size: int = 32,
    height: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    augment: bool = False
) -> DataLoader:
    """
    Create DataLoader for OCR training/validation
    
    Args:
        image_paths: List of image file paths
        labels: List of corresponding text labels
        batch_size: Batch size
        height: Fixed image height
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        augment: Whether to apply augmentation
        
    Returns:
        DataLoader with dynamic padding
    """
    dataset = OCRDataset(
        image_paths=image_paths,
        labels=labels,
        height=height,
        augment=augment
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    return dataloader


# Example usage and testing
if __name__ == "__main__":
    # Create dummy data for testing
    import tempfile
    
    print("Creating test data...")
    temp_dir = tempfile.mkdtemp()
    
    # Create some test images with different widths
    test_data = []
    for i, (width, text) in enumerate([
        (100, "hello"),
        (150, "world"),
        (80, "test"),
        (120, "OCR")
    ]):
        # Create random grayscale image
        img_array = np.random.randint(0, 255, (32, width), dtype=np.uint8)
        img = Image.fromarray(img_array, mode='L')
        
        # Save image
        img_path = os.path.join(temp_dir, f"test_{i}.png")
        img.save(img_path)
        test_data.append((img_path, text))
    
    image_paths, labels = zip(*test_data)
    
    # Create dataloader
    print("\nCreating DataLoader...")
    dataloader = create_dataloader(
        image_paths=list(image_paths),
        labels=list(labels),
        batch_size=2,
        shuffle=False,
        num_workers=0  # Use 0 for testing
    )
    
    # Test iteration
    print("\nTesting DataLoader iteration:")
    for batch_idx, batch in enumerate(dataloader):
        print(f"\nBatch {batch_idx + 1}:")
        print(f"  Images shape: {batch['images'].shape}")
        print(f"  Labels: {batch['labels']}")
        print(f"  Original widths: {batch['lengths'].tolist()}")
        
    print("\n✓ DataLoader implementation complete!")
    print(f"✓ Handles variable widths with dynamic padding")
    print(f"✓ Fixed height: 32 pixels")
    print(f"✓ Output shape: (B, 1, H=32, W_max)")
    
    # Cleanup
    import shutil
    shutil.rmtree(temp_dir)
