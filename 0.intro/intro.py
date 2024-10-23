import torch
import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt


# scalar
scalar = torch.tensor(7) #single value with no dimensions
print(scalar.ndim) #no of dimension 0
print(scalar.item())
print(scalar, "\n")


# vector
vector = torch.tensor([5,6]) #list of scalar
print(vector.ndim) #no of dimension 1
print(vector.shape) #size[2] meaning a vector with 2 item
print(vector.size())
print(vector, "\n")

# MATRIX
MATRIX = torch.tensor([ [1,2],[3,4] ]) #list of vector
print(MATRIX.ndim) #no of dim 2
print(MATRIX.shape) #size[2,2] meaning there are 2 vector @ with 2 items
print(MATRIX, "\n")

# TENSOR
TENSOR = torch.tensor([ [ [1,2,3], [4,5,6], [7,8,9] ] ]) #list of MATRIX
print(TENSOR.ndim) #no of dim 3
print(TENSOR.shape) #size[1,3,3] meaning 1 matrix @ 3 vectors @ 3 items
print(TENSOR, "\n")

# random tensors
random_tensor = torch.rand(3) # vector: 1 vector @ 3 items
print(random_tensor, "\n")

random_tensor = torch.rand(3,4) # matirx: 1 matrix @ 3 vectors @ 4 items
print(random_tensor, "\n")

random_tensor = torch.rand(2,2,2) # a tensor: 2 matrix @ 2 vectors @ 2 items
print(random_tensor, "\n")

# random tensor with similar shape to image
random_image_size_tensor = torch.rand(size=(224, 224, 3)) #tensor: height, widht , colour
print(random_image_size_tensor, "\n")

# tensor of all zeros
zeros = torch.zeros(2,2)
print(zeros, "\n")

ones = torch.ones(2,2)
print(ones, ones.dtype, "\n")

# range of tensors
print(torch.arange(1, 11), "\n") # vector: 1 -> 10

one_to_ten = torch.arange(start=0, end=1000, step=10)
print(one_to_ten, "\n")

#tensor_like
ten_zeros = torch.zeros_like(input=one_to_ten)
print(ten_zeros, "\n")

# tensor dtype
float32_tensor = torch.tensor([3.0, 6.0, 9.0], 
                              dtype=torch.float16, #data type
                              device=None, # cpu or cuda
                              requires_grad=False # track gradient
                            )
                            # common dtypes: 
                            # torch.float32, single precision
                            # torch.float16, half precision
                            # torch.int32
print(float32_tensor.dtype)
print(f"Device is: {float32_tensor.device}")
            
# convert dtype
float32_tensor = float32_tensor.type(torch.float32)
print(float32_tensor.dtype, "\n")






