import os
import torch
from torch.utils.data import DataLoader, Dataset

class CustomDataset(Dataset):
    def __init__(self, image_tensors, target_tensors):
        self.images = image_tensors
        self.labels = target_tensors

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        x = self.images[idx].unsqueeze(0)
        y = self.labels[idx]
        return x, y
    
## Function for getting dataloader
def GetDataloader(batch_size=64, shuffle=True):

    base_path = os.path.join("data", "processed")

    image_tensors = torch.load(os.path.join(base_path, "train_images.pt"))
    target_tensors = torch.load(os.path.join(base_path, "train_targets.pt"))
    train_set = CustomDataset(image_tensors, target_tensors)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle)

    image_tensors = torch.load(os.path.join(base_path, "test_images.pt"))
    target_tensors = torch.load(os.path.join(base_path, "test_targets.pt"))
    test_set = CustomDataset(image_tensors, target_tensors)
    test_loader = DataLoader(test_set, batch_size=None, shuffle=False)
    

    return train_loader, test_loader

