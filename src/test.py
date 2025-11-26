import torch
from torch.utils.data import DataLoader
from dataset import ShapePrimitiveDataset

def test_training_coverage():
    # Create dataset and dataloader
    dataset = ShapePrimitiveDataset()
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)
    
    print(f"Total samples in dataset: {len(dataset)}")
    
    # Track which samples are seen
    seen_samples = set()
    
    # Simulate a few training batches
    for batch_idx, batch in enumerate(dataloader):
        print(f"Batch {batch_idx}: shape {batch.shape}")
        
        if batch_idx >= 10:  # Just test a few batches
            break
    
    # Test individual sample access
    print("\nTesting individual sample access:")
    for i in range(len(dataset)):
        sample = dataset[i]
        shape_name = dataset.file_paths[i].split('/')[-2]
        print(f"Sample {i} ({shape_name}): shape {sample.shape}")
    
    # Check access statistics if available
    if hasattr(dataset, 'get_access_stats'):
        dataset.get_access_stats()

if __name__ == "__main__":
    test_training_coverage()