import torch

def get_device():
    """Get the available device: CUDA, MPS, or CPU."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA device")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS device")
    else:
        device = torch.device("cpu")
        print("Neither CUDA nor MPS available, using CPU")
    return device

def freeze_all_layers_except_last(model):
    """Freeze all layers of the model except the last one."""
    for name, param in model.named_parameters():
        if 'classifier' not in name:
            param.requires_grad = False
        else:
            param.requires_grad = True
