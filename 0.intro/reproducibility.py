import torch 

# reproducibility is concept that tends to reduce randomness of data when generating 
# random data for a tensor

rand_A = torch.rand(3,4)
rand_B = torch.rand(3,4)

print(f"Tensor A:\n {rand_A} \nTensor B:\n {rand_B} \n")
print(f"Do to the tensors match:\n {rand_A == rand_B}")


# random seed

RANDOM_SEED = 42
torch.manual_seed(seed=RANDOM_SEED)
random_tensor_c = torch.rand(3,4)
    # Have to reset the seed every time a new rand() is called 
    # Without this, tensor_D would be different to tensor_C 
torch.manual_seed(seed=RANDOM_SEED)
random_tensor_D = torch.rand(3, 4)

print(f"Tensor C:\n{random_tensor_c}\n")
print(f"Tensor D:\n{random_tensor_D}\n")
print(f"Does Tensor C equal Tensor D? (anywhere)")
print(f"Do to the tensors match:\n {random_tensor_c == random_tensor_D}")
