import torch
import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt

x = torch.arange(0, 100, 10)
print(x, '\n')

# max
print(x.max(), torch.max(x))

# min
print(x.min(), torch.min(x))

# sum
print(x.sum(), torch.sum(x))

# mean - only deals with float32 and complex dtypes
print(x.type(torch.float32).mean(), torch.mean(x.type(torch.float32)))


# the position of min value and max
print(x.argmin()) # min 
print(x.argmax()) 