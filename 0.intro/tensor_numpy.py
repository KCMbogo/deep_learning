import torch
import numpy as np

# from numpy -> torch

# by default numpy uses float64 but torch p4ms mostly using float32 
# so when converting to torch make it float32
array = np.arange(1.0, 8.0)
tensor = torch.from_numpy(array).type(torch.float32)

print(f"The array is type: {array.dtype} \n and tensor is: {tensor.dtype} \n array is: {array} \n tensor is: {tensor} \n\n\n")


# tensor -> numpy

tensor = torch.ones(7)
numpy_tensor = tensor.numpy() # it will float32 unless changed

print(f"The tensor is type: {tensor.dtype} \n and numpy_tensor is type: {numpy_tensor.dtype} \n tensor is: {tensor} \n numpy_tensor is: {numpy_tensor}")

