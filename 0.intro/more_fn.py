import torch

x = torch.arange(1., 11.)
print(x, '\n') # vector

# reshaping - reshapes an input tensor to a defined shape

print(x.reshape(10)) # vector of 9 items
print(x.reshape(1,10)) # matrix of 1 vector 9 items 
print(x.reshape(10,1)) # matrix of 9 vectors 1 item 
print(x.reshape(5,2)) # matrix of 5 vectors 2 item 
print(x.reshape(5,2).shape) # matrix of 5 vectors 2 item shape
print(x.reshape(1,1,10), '\n') # tensor of 1 matrix, 1 vector 9 items

#  view - just like reshape but shares memory with original tensor

x_view = x.view(1,10)
print(x_view, '\n') # matrix of 1 vector 9 items but x_view shares memory as x

# stack - combine multiple tensors on top of each other
# vstack - vertically combine dim=0
# hstack - horizontally combine dim=1

x_stacked = torch.stack([x,x,x,x], dim=1)
print(x_stacked, '\n')

# squeeze - removes a single dimensions from a tensor

print(x_view.squeeze(), '\n')

# unsqueeze - add a sinlgle dim to a target tensor

print(x_view.unsqueeze(dim=0))

# permute

x_image = torch.rand((224,240,3))
print(x_image.permute(2,0,1), '\n') # rearrange dim start with dim 2, 0, 1

# indexing

rand_x = torch.rand(3,3,2)
print(rand_x, '\n')
print(rand_x[0], '\n\n', rand_x[0][0], '\n\n', rand_x[:,0], '\n\n', rand_x[:,:,0], '\n\n', rand_x[0,0,:]) 
        # rand_x[:,:,0]:
        # : select all elements along the first dimension (i.e., all 3 layers).
        # : select all elements along the second dimension (i.e., all rows within each layer).
        # 0 select the first column in each 2D slice.
        
        # rand_x[0,0,:]: same as [0][0]
        # 0 select the first matrix in the first dimension.
        # 0 select the first row within the selected matrix.
        # : select all elements in the column of the selected row.
        


                            
