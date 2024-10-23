import os
import torch
import pathlib
import random
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from typing import Tuple, List, Dict
from helper_fn import plot_transformed_images

device = "cuda" if torch.cuda.is_available() else "cpu"

data_path = pathlib.Path("data")
image_path = pathlib.Path(data_path / "pizza_steak_sushi")

train_dir = image_path / "train"
test_dir = image_path / "test"

### SETTING A CUSTOM DATASET
# 1. getting class names and class to idx

def find_classes(directory: str) -> Tuple[List[str], Dict[str, int]]:
    """Finds the class folder names in target directory

    Args:
        directory (str): the directory for the classes

    Returns:
        Tuple[List[str], Dict[str, int]]: (list(str), {str: int})
    """
    classes = sorted(entry.name for entry in list(os.scandir(directory)) if entry.is_dir)
    
    if not classes:
        raise FileNotFoundError(f"Couldn't find any classes in {directory} directory")
    
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx

# print(find_classes(train_dir))

# 2. write custom dataset class by subclassing torch.utils.data.Dataset

class ImageFolderCustom(Dataset):
    
    # Initialize with a targ_dir and transform (optional) parameter
    def __init__(self, targ_dir: str, transform=None) -> None:
        self.paths = list(pathlib.Path(targ_dir).glob("*/*.jpg"))
        self.transform = transform
        self.classes, self.class_to_idx = find_classes(targ_dir)
    
    # Make a function to load images
    def load_images(self, index: int) -> Image.Image:
        image_path = self.paths[index]
        return Image.open(image_path)
    
    # Overwirte the __len__() method
    def __len__(self) -> int:
        return len(self.paths)
    
    # Overwrite __getitem__() method
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        img = self.load_images(index)
        class_name = self.paths[index].parent.name
        class_to_idx = self.class_to_idx[class_name]
        
        # Transform if necessary
        if self.transform:
            return self.transform(img), class_to_idx
        else:
            return img, class_to_idx # data, label
        

train_transforms = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.TrivialAugmentWide(num_magnitude_bins=31),
    transforms.ToTensor()
])

test_transforms = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

train_data_custom = ImageFolderCustom(targ_dir=train_dir,
                                      transform=train_transforms)

test_data_custom = ImageFolderCustom(targ_dir=test_dir,
                                     transform=test_transforms)


def display_random_images(dataset: torch.utils.data.Dataset,
                          classes: List[str] = None,
                          n: int = 10,
                          display_shape: bool = True,
                          seed: int = None):
    if n > 10:
        n = 10
        display_shape = False
        print(f"For display purposes, n shouldn't be larger than 10, setting to 10 and removing shape display.")
        
    if seed:
        random.seed(seed)
    
    random_samples_idx = random.sample(range(len(dataset)), k=n)
    
    plt.figure(figsize=(16, 8))
    for i, targ_sample in enumerate(random_samples_idx):
        targ_image, targ_label = dataset[targ_sample][0], dataset[targ_sample][1]

        # Adjust image tensor shape for plotting: 
        targ_image_adjust = targ_image.permute(1, 2, 0)

        # Plot adjusted samples
        plt.subplot(1, n, i+1)
        plt.imshow(targ_image_adjust)
        plt.axis("off")
        if classes:
            title = f"class: {classes[targ_label]}"
            if display_shape:
                title = title + f"\nshape: {targ_image_adjust.shape}"
        plt.title(title)
    plt.show()

class_names = train_data_custom.classes

# Display random images from ImageFolderCustom Dataset
# display_random_images(train_data_custom, 
#                       n=12, 
#                       classes=class_names,
#                       seed=None)

# Turn train and test custom Dataset's into DataLoader's
train_dataloader_custom = DataLoader(dataset=train_data_custom, # use custom created train Dataset
                                     batch_size=1, # how many samples per batch?
                                     num_workers=0, # how many subprocesses to use for data loading? (higher = more)
                                     shuffle=True) # shuffle the data?

test_dataloader_custom = DataLoader(dataset=test_data_custom, # use custom created test Dataset
                                    batch_size=1, 
                                    num_workers=0, 
                                    shuffle=False) # don't usually need to shuffle testing data

# print(train_dataloader_custom, test_dataloader_custom)

# Get image and label from custom DataLoader
img_custom, label_custom = next(iter(train_dataloader_custom))

# Batch size will now be 1, try changing the batch_size parameter above and see what happens
# print(f"Image shape: {img_custom.shape} -> [batch_size, color_channels, height, width]")
# print(f"Label shape: {label_custom.shape}")

image_path_list = list(image_path.glob("*/*/*.jpg"))

plot_transformed_images(image_paths=image_path_list, transform=train_transforms, n=3, seed=None)