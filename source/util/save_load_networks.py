import torch

# Todo: check if both models need to be saved and loaded during training and test
# Todo: depending on max epochs maybe remove older checkpoints to not run out of space

def save_models(model, optimizer, epoch, checkpointPath):
    checkpoint = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict()
    }
    torch.save(checkpoint, checkpointPath)

def load_models(model, optimizer, checkpointPath):
    loaded_checkpoint = torch.load(checkpointPath)
    model.loaded_checkpoint.load_state_dict(loaded_checkpoint["model_state"])
    optimizer.load_state_dict(loaded_checkpoint["optimizer_state"])
    return loaded_checkpoint["epoch"]