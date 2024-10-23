import torch
import torchvision
import matplotlib.pyplot as plt
import data_setup
import engine
import random
from tqdm.auto import tqdm
from typing import List, Tuple
from PIL import Image
from torch import nn
from torchinfo import summary
from torchvision import transforms
from helper_fn import plot_loss_curves
from timeit import default_timer as timer
from pathlib import Path

import requests

if Path("data_setup.py").is_file() and Path("engine.py").is_file():
    print("data_setup.py and engine.py files already exist, skipping download...")
else:
    with open("data_setup.py", "wb") as f:
        request = requests.get("https://github.com/mrdbourke/pytorch-deep-learning/raw/refs/heads/main/going_modular/going_modular/data_setup.py")
        print("Downloading data_setup.py file...")
        f.write(request.content)
    with open("engine.py", "wb") as f:
        request = requests.get("https://github.com/mrdbourke/pytorch-deep-learning/raw/refs/heads/main/going_modular/going_modular/engine.py")
        print("Downloading engine.py file...")
        f.write(request.content)
        

# Setup device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"

data_path = Path("data")
image_path = Path(data_path / "pizza_steak_sushi")

train_dir = image_path / "train"
test_dir = image_path / "test"

# Get a set of pretrained model weights
weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT # DEFAULT = Best available options
model = torchvision.models.efficientnet_b0(weights=weights).to(device)

# Get the transforms used on the weights
auto_transforms = weights.transforms()
# print(auto_transforms)

train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(train_dir=train_dir,
                                                                               test_dir=test_dir,
                                                                               transform=auto_transforms,
                                                                               batch_size=32)
# print(train_dataloader, test_dataloader, class_names)
# print(model)

# summary(model=model,
#         input_size=(32, 3, 224, 224),
#         col_names=["input_size", "output_size", "num_params", "trainable"],
#         col_width=20,
#         row_settings=["var_names"]
#         )


# Let's freeze the features layer  -> Avoid updating gradient
for param in model.features.parameters():
    param.requires_grad = False

# Let's change the out_features on the classifier layer
torch.manual_seed(42)
torch.cuda.manual_seed(42)

out_shape = len(class_names)

model.classifier = nn.Sequential(
    nn.Dropout(p=0.2, inplace=True), # removes 20% of connections 
    nn.Linear(in_features=1280,
              out_features=out_shape,
              bias=True)
).to(device)

# summary *after* freezing the features and changing the output classifier layer (uncomment for actual output)
summary(model, 
        input_size=(32, 3, 224, 224),
        verbose=1, # 0 - no output, 1 - show output(default), 2 - show weights and bias
        col_names=["input_size", "output_size", "num_params", "trainable"],
        col_width=20,
        row_settings=["var_names"]
        )


optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
loss_fn = torch.nn.CrossEntropyLoss()

torch.manual_seed(42)
torch.cuda.manual_seed(42)

star_time = timer()

results = engine.train(model=model,
                       train_dataloader=train_dataloader,
                       test_dataloader=test_dataloader,
                       optimizer=optimizer,
                       loss_fn=loss_fn,
                       epochs=5,
                       device=device)

end_time = timer()
print(f"[INFO] Total training time: {end_time-star_time} seconds")

plot_loss_curves(results=results)

# 1. Take in a trained model, class names, image path, image size, a transform and target device
def pred_and_plot_image(model: torch.nn.Module,
                        image_path: str, 
                        class_names: List[str],
                        image_size: Tuple[int, int] = (224, 224),
                        transform: torchvision.transforms = None,
                        device: torch.device=device):
    # 2. Open image
    img = Image.open(image_path)

    # 3. Create transformation for image (if one doesn't exist)
    if transform is not None:
        image_transform = transform
    else:
        image_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    ### Predict on image ### 

    # 4. Make sure the model is on the target device
    model.to(device)

    # 5. Turn on model evaluation mode and inference mode
    model.eval()
    with torch.inference_mode():
      # 6. Transform and add an extra dimension to image (model requires samples in [batch_size, color_channels, height, width])
      transformed_image = image_transform(img).unsqueeze(dim=0)

      # 7. Make a prediction on image with an extra dimension and send it to the target device
      target_image_pred = model(transformed_image.to(device))

    # 8. Convert logits -> prediction probabilities (using torch.softmax() for multi-class classification)
    target_image_pred_probs = torch.softmax(target_image_pred, dim=1)

    # 9. Convert prediction probabilities -> prediction labels
    target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)

    # 10. Plot image with predicted label and probability 
    plt.figure()
    plt.imshow(img)
    plt.title(f"Pred: {class_names[target_image_pred_label]} | Prob: {target_image_pred_probs.max():.3f}")
    plt.axis(False)
    plt.show()
    
    return target_image_pred_label
    
# Get a random list of image paths from test set
import random
num_images_to_plot = 10
test_image_path_list = list(Path(test_dir).glob("*/*.jpg")) # get list all image paths from test data 
test_image_path_sample = random.sample(population=test_image_path_list, # go through all of the test image paths
                                       k=num_images_to_plot) # randomly select 'k' image paths to pred and plot

# Make predictions on and plot the images
for image_path in test_image_path_sample:
    pred_and_plot_image(model=model, 
                        image_path=image_path,
                        class_names=class_names,
                        # transform=weights.transforms(), # optionally pass in a specified transform from our pretrained model weights
                        image_size=(224, 224))    
    
    
    
    
    
# EXERCISE
# Confusion matrix

# y_preds = []
# model.eval()
# with torch.inference_mode():
#     for X, y in test_dataloader:
#         y_logit = model(X)
#         y_prob = torch.softmax(y_logit, dim=1)
#         y_pred = y_prob.argmax(dim=1)
#         y_preds.append(y_pred)
        
# y_pred_tensor = torch.cat(y_preds)

# image_transform = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                              std=[0.229, 0.224, 0.225]),
#     ])

# from torchmetrics import ConfusionMatrix
# from mlxtend.plotting import plot_confusion_matrix
# from torchvision import datasets

# test_data = datasets.ImageFolder(root=test_dir,
#                                  transform=image_transform)

# test_targets = torch.tensor(test_data.targets)

# confmat = ConfusionMatrix(num_classes=len(class_names), task="multiclass")
# confmat_tensor = confmat(preds=y_pred_tensor,
#                          target=test_targets)  

# fig, ax = plot_confusion_matrix(
#     conf_mat=confmat_tensor.numpy(),
#     class_names=class_names,
#     figsize=(10, 7)
# )      
# plt.show()

# from pathlib import Path
# test_data_paths = list(Path(test_dir).glob("*/*.jpg"))
# test_labels = [path.parent.stem for path in test_data_paths]

# # Create a function to return a list of dictionaries with sample, label, prediction, pred prob
# def pred_and_store(test_paths, model, transform, class_names, device):
#   test_pred_list = []
#   for path in tqdm(test_paths):
#     # Create empty dict to store info for each sample
#     pred_dict = {}

#     # Get sample path
#     pred_dict["image_path"] = path

#     # Get class name
#     class_name = path.parent.stem
#     pred_dict["class_name"] = class_name

#     # Get prediction and prediction probability
#     from PIL import Image
#     img = Image.open(path) # open image
#     transformed_image = transform(img).unsqueeze(0) # transform image and add batch dimension
#     model.eval()
#     with torch.inference_mode():
#       pred_logit = model(transformed_image.to(device))
#       pred_prob = torch.softmax(pred_logit, dim=1)
#       pred_label = torch.argmax(pred_prob, dim=1)
#       pred_class = class_names[pred_label.cpu()]

#       # Make sure things in the dictionary are back on the CPU 
#       pred_dict["pred_prob"] = pred_prob.unsqueeze(0).max().cpu().item()
#       pred_dict["pred_class"] = pred_class
  
#     # Does the pred match the true label?
#     pred_dict["correct"] = class_name == pred_class

#     # print(pred_dict)
#     # Add the dictionary to the list of preds
#     test_pred_list.append(pred_dict)

#   return test_pred_list

# test_pred_dicts = pred_and_store(test_paths=test_data_paths,
#                                  model=model,
#                                  transform=image_transform,
#                                  class_names=class_names,
#                                  device=device)

# print(test_pred_dicts[:5], 'end')

# # Turn the test_pred_dicts into a DataFrame
# import pandas as pd
# test_pred_df = pd.DataFrame(test_pred_dicts)
# # Sort DataFrame by correct then by pred_prob 
# top_5_most_wrong = test_pred_df.sort_values(by=["correct", "pred_prob"], ascending=[True, False]).head()
# print(top_5_most_wrong)

# # custom_image = data_path / "pizza.jpg"

# # prediction = pred_and_plot_image(model=model,
# #                     image_path=custom_image,
# #                     class_names=class_names,
# #                     image_size=(224, 224),
# #                     transform=image_transform)

# # print(prediction)