"""
Fill out the newly created <project_name>/models/predict_model.py file, such that it takes a pre-trained model file and creates prediction for some data. 
Recommended interface is that users can give this file either a folder with raw images that gets loaded in or a numpy or pickle file with already loaded images e.g. something like this

python <project_name>/models/predict_model.py \
    models/my_trained_model.pt \  # file containing a pretrained model
    data/example_images.npy  # file containing just 10 images for prediction
"""

import torch

def predict(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader
) -> None:
    """Run prediction for a given model and dataloader.
    
    Args:
        model: model to use for prediction
        dataloader: dataloader with batches
    
    Returns
        Tensor of shape [N, d] where N is the number of samples and d is the output dimension of the model

    """

    
    return torch.cat([model(batch) for batch in dataloader], 0)