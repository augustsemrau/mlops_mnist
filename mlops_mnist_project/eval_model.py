import os

import torch
from tqdm import tqdm

from data.dataloader import GetDataloader

def evaluate():
    """Evaluate a trained model."""
    print("Evaluating like my life dependends on it")
    


    model = torch.load(os.path.join(os.path.dirname(__file__), "..", "models", "model.pt"))
    train_loader, test_loader = GetDataloader(batch_size=64, shuffle=False)

    accuracy = 0
    with torch.no_grad():
        for images, labels in tqdm(test_loader, total=len(test_loader)):
            log_ps = model(images)
            ps = torch.exp(log_ps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor))

    print(f"Accuracy: {accuracy/len(test_loader)}")

if __name__ == "__main__":
    evaluate()