import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

tensor = torch.rand(3, device='cpu')
print(tensor, tensor.device)

tensor = tensor.to(device)
print(tensor, tensor.device)

# learning - device agnostic code
# cuda best practices