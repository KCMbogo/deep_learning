import torch
import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt

tensor = torch.tensor([1,2,3])
print(tensor, "\n")

# add 
print(tensor + 10, "\n")

# sub
print(tensor - 10, "\n")
 
# mul
print(tensor * 10, "\n")

# div
print(tensor / 10, "\n")

# matmul - matrix mul or using @ operator or torch.mm()
mat1 = torch.rand(2,3) #2 by 3 natrix (row by col)
print(mat1)
mat2 = torch.rand(3,4) # 3 by 4 matrix (row by col)
print(mat2)
print(torch.matmul(mat1, mat2), '\n') # 2 by 4 (row by col)
                                    # mat1 @ mat2
                                    # torch.mm(mat1, mat2)
                                    
                                
# Transpose: Switch the dimension rows to cols
mat_A = torch.rand(2,3)
print(mat_A, '\n')

mat_B = torch.rand(2,3)
print(mat_B, '\n')

# print(torch.mm(mat_A, mat_B)) # error because it is 2,3 * 2,3
print(torch.mm(mat_A, mat_B.T)) # transpose mat_B so it is 2,3 * 3,2
