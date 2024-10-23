import torch
import torchvision
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

####################################
import requests
from pathlib import Path

if Path("helper_fn.py").is_file():
    # print("File exists, skipping download")
    pass
else:
    print("Downloading file...")
    request = requests.get("https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/helper_functions.py")
    with open("helper_fn.py", "wb") as f:
        f.write(request.content)
#####################################
        

# SET THE DATA

train_data = datasets.FashionMNIST(
    root='data', # where to download?
    train=True, # is it for training?
    download=True, # want to download?
    transform=ToTensor(), # want to transform data to tensor?
    target_transform=None # want to transform labels to tensor?
)

test_data = datasets.FashionMNIST(
    root='data',
    train=False,
    download=True,
    transform=ToTensor(),
    target_transform=None
) # print(len(train_data), len(test_data)) # 60000 10000

# the first image tensor and label
image, label = train_data[0] 

# class names 
class_names = train_data.classes
# print(class_names)

# dictionary of classnames and indecies
class_to_idx = train_data.class_to_idx
# print(class_to_idx)


# visualize the image

# plt.imshow(image.squeeze(), cmap='gray')
# plt.title(class_names[label])
# plt.axis(False)
# plt.show()

# visualize more images
torch.manual_seed(42)

fig = plt.figure(figsize=(9, 9))
rows, cols = 4, 4

# for i in range(1, rows*cols+1):
#     random_idx = torch.randint(0, len(train_data), size=[1]).item()
#     img, label = train_data[random_idx]
#     fig.add_subplot(rows, cols, i)
#     plt.imshow(img.squeeze(), cmap='gray')
#     plt.title(class_names[label])
#     plt.axis(False)
    

# USE DATALOADER TO CONVERT DATA TO ITERABLE
BATCH_SIZE = 32
train_dataloader = DataLoader(dataset=train_data,
                              batch_size=BATCH_SIZE,
                              shuffle=True)

test_dataloader = DataLoader(dataset=test_data,
                             batch_size=BATCH_SIZE,
                             shuffle=False)
# print(f"Dataloaders: {train_dataloader, test_dataloader}")
# print(f"Train dataloader length: {len(train_dataloader)} batches of {BATCH_SIZE}")
# print(f"Test dataloader length: {len(test_dataloader)} batches of {BATCH_SIZE}")


# Get the batches inside the training dataloader
train_features_batch, train_labels_batch = next(iter(train_dataloader))
# print(train_features_batch.shape, train_labels_batch.shape, len(train_features_batch))

torch.manual_seed(42)
random_idx = torch.randint(0, len(train_features_batch), size=[1]).item()
img, label = train_features_batch[random_idx], train_labels_batch[random_idx]

# plt.imshow(img.squeeze(), cmap='gray')
# plt.title(class_names[label])
# plt.axis(False)
# print(f"Image size: {img.shape}")
# print(f"Label: {label}, Label size: {label.shape}")
# plt.show()