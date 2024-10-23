import torch
import torch.nn as nn
import torchvision 
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from helper_fn import walk_through_dir

device = "cuda" if torch.cuda.is_available() else "cpu"

### DOWNLOADING THE FOOD DATA ######################
import requests
from pathlib import Path
import zipfile

data_path = Path("data")
image_path = Path(data_path / "pizza_steak_sushi")

# if image_path.is_dir():
#     print(f"{image_path} directory already exists...")
# else:
#     print(f"{image_path} directory doesn't exist, creating one...")
#     image_path.mkdir(parents=True, exist_ok=True)

# with open(data_path / "pizza_steak_sushi.zip", "wb") as f:
#     request = requests.get("https://github.com/mrdbourke/pytorch-deep-learning/raw/refs/heads/main/data/pizza_steak_sushi.zip")
#     print("Downloading pizza_steak_sushi.zip file...")
#     f.write(request.content)
    
# with zipfile.ZipFile(data_path / "pizza_steak_sushi.zip", "r") as zip_ref:
#     print("Extracting contents from pizza_steak_sushi.zip file...")
#     zip_ref.extractall(image_path)
######################################################

# walk_through_dir(image_path)

train_dir = image_path / "train"
test_dir = image_path / "test"

# visualize

import random
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

random.seed(42)

# get all image path: * means any combination
image_path_list = list(image_path.glob("*/*/*.jpg"))

# get random image path
random_image_path = random.choice(image_path_list)

# get class name of the image -> which is the parent folder of the image
image_class = random_image_path.parent.stem

# open image
img = Image.open(random_image_path)

# print meta data
# print(f"Random image path: {random_image_path}")
# print(f"Image class: {image_class}")
# print(f"Image height: {img.height}")
# print(f"Image width: {img.width}")
# print(img, img,show())

# img_as_array = np.asarray(img)
# plt.figure(figsize=(10, 5))
# plt.imshow(img_as_array)
# plt.title(f"Image class: {image_class} | Image shape: {img_as_array.shape} -> [height, width, color_channels]")
# plt.axis(False)
# plt.show()


# Transform the data to suit pytorch 

data_transform = transforms.Compose([
    transforms.Resize(size=(64, 64)), # resize the image
    transforms.RandomHorizontalFlip(p=0.5), # flip the images horizontally; p is the prob for flipping
    transforms.ToTensor()
])

# visualize transformed images

def plot_transformed_images(image_paths: list, transform: torchvision.transforms, n=3, seed=42):
    """Plots a series of random images from image_path

    Args:
        image_path (list): series of images
        transform (torchvision.transforms): transformation of images
        n (int, optional): number of images to plot. Defaults to 3.
        seed (int, optional): random generator. Defaults to 42.
    """
    random.seed(seed)
    random_image_path = random.sample(image_paths, k=n)
    
    for image_path in random_image_path:
        with Image.open(image_path) as f:
            fig, ax = plt.subplots(1, 2)
            ax[0].imshow(f)
            ax[0].set_title(f"Original \nSize: {f.size}")
            ax[0].axis("off")
            
            # Transform and plot image
            # transform.permute() changes the image to suit matplotlib's (H, W, C)
            
            transformed_image = transform(f).permute(1, 2, 0)
            ax[1].imshow(transformed_image)
            ax[1].set_title(f"Transformed \nSize: {transformed_image.shape}")
            ax[1].axis("off")
            
            fig.suptitle(f"Class: {image_path.parent.stem}", fontsize=16)
    plt.show()
            
# plot_transformed_images(image_paths=image_path_list, transform=data_transform)


# Turn our data into datasets

train_data = datasets.ImageFolder(root=train_dir,
                                  transform=data_transform,
                                  target_transform=None)

test_data = datasets.ImageFolder(root=test_dir,
                                 transform=data_transform)

class_names = train_data.classes

class_to_idx = train_data.class_to_idx

img, label = train_data[0][0], train_data[0][1]
# print(f"Image tensor:\n{img}")
# print(f"Image shape: {img.shape}")
# print(f"Image datatype: {img.dtype}")
# print(f"Image label: {label}")
# print(f"Label datatype: {type(label)}")

img_permute = img.permute(1, 2, 0)

# print(f"Original image shape: {img.shape} -> [C, H, W]")
# print(f"Permuted image shape: {img_permute.shape} -> [H, W, C]")

# plt.figure(figsize=(10, 7))
# plt.imshow(img_permute)
# plt.axis(False)
# plt.title(class_names[label], fontsize=14)
# plt.show()

import os
# print(os.cpu_count()) # return the number of cores in my pc to use for num_workers

train_dataloader = DataLoader(dataset=train_data,
                              batch_size=1,
                              num_workers=8,
                              shuffle=True)

test_dataloader = DataLoader(dataset=test_data,
                             batch_size=1,
                             num_workers=8,
                             shuffle=False)


# img, label = next(iter(train_dataloader))
# print(f"Image shape: {img.shape} \nLabel shape: {label.shape}")


# LET'S START SIMPLE

simple_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

train_data_simple = datasets.ImageFolder(root=train_dir,
                                         transform=simple_transform)

test_data_simple = datasets.ImageFolder(root=test_dir,
                                        transform=simple_transform)

BATCH_SIZE = 32
NUM_WORKERS = os.cpu_count()

train_dataloader_simple = DataLoader(train_data_simple,
                                     batch_size=BATCH_SIZE,
                                     num_workers=NUM_WORKERS,
                                     shuffle=True)

test_dataloader_simple = DataLoader(test_data_simple,
                                    batch_size=BATCH_SIZE,
                                    num_workers=NUM_WORKERS,
                                    shuffle=False)
