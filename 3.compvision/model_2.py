"""This module deals with a convolution neural network model
conv_block_x = conv_layer->relu->conv_layer->relu->maxpool_layer
classifier_layer = flatten->linear
"""
import torch
import torchvision
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from timeit import default_timer as timer
from tqdm.auto import tqdm
from helper_fn import accuracy_fn

# import os
# # See if torchmetrics exists, if not, install it
# try:
#     import torchmetrics, mlxtend
#     print(f"mlxtend version: {mlxtend.__version__}")
#     assert int(mlxtend.__version__.split(".")[1]) >= 19, "mlxtend verison should be 0.19.0 or higher"
# except:
#     os.system("pip install -q torchmetrics mlxtend") # <- Note: If you're using Google Colab, this may require restarting the runtime
#     import torchmetrics, mlxtend
#     print(f"mlxtend version: {mlxtend.__version__}")

device = "cuda" if torch.cuda.is_available() else "cpu"

train_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=False,
    transform=ToTensor(),
    target_transform=None
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=False,
    transform=ToTensor(),
    target_transform=None
)

train_dataloader = DataLoader(dataset=train_data,
                              batch_size=32,
                              shuffle=True,
                              )

test_dataloader = DataLoader(dataset=test_data,
                             batch_size=32,
                             shuffle=False)

class_names = train_data.classes

train_data_batch, train_label_batch = next(iter(train_dataloader))

torch.manual_seed(42)
random_idx = torch.randint(0, len(train_data_batch), size=[1]).item()
img, label = train_data_batch[random_idx], train_label_batch[random_idx]


class FashionMNISTModelV2(nn.Module):
    def __init__(self, in_shape: int, out_shape: int, hidden_units: int):
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=in_shape, # this is the number of color channels
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units*49, # out shape of flatten layer
                      out_features=out_shape)
        )
    
    def forward(self, x):
        x = self.conv_block_1(x)
        # print(f"Out_shape for conv block 1: {x.shape}")
        x = self.conv_block_2(x)
        # print(f"Out shape for conv block 2: {x.shape}")
        # print(f"Out shape for flatten: {nn.Flatten()(x).shape}")
        x = self.classifier(x)
        # print(f"Out shape for classifier: {x.shape}")
        
        return x
    
torch.manual_seed(42)

model_2 = FashionMNISTModelV2(in_shape=1, out_shape=len(class_names), hidden_units=10)


# images = torch.randn(size=(32, 1, 64, 64))
# test_image = images[0]

# Looking in conv2d layer
# torch.manual_seed(42)
# conv2d_layer = nn.Conv2d(in_channels=3,
#                          out_channels=10,
#                          kernel_size=3,
#                          padding=0,
#                          stride=1)

# print(f"Image: {test_image.shape}")

# conv2d_output = conv2d_layer(test_image)
# # print(conv2d_output)
# print(f"Image through convolution2d layer: {conv2d_output.shape}")


# # Looking in maxpool2d layer
# maxpool2d_layer = nn.MaxPool2d(kernel_size=2)
# maxpool2d_out = maxpool2d_layer(conv2d_output)
# print(f"Image through maxpool2d layer: {maxpool2d_out.shape}")


# # Looking in flatten
# flatten_layer = nn.Flatten()
# flatten_out = flatten_layer(maxpool2d_out)
# print(f"Image through flatten layer: {flatten_out.shape}")

# # Looking in linear layer
# linear_layer = nn.Linear(in_features=31*31,
#                          out_features=len(class_names))
# linear_out = linear_layer(flatten_out)
# print(f"Image through linear layer: {linear_out.shape}")

# dummy_model_2 = model_2(img)
# print(dummy_model_2.shape)
# print(img.shape)


# 3. Loss fn and Optimizer

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model_2.parameters(),
                            lr=0.1)

# 4. The timing function

def print_exec_time(start, end):
    total_time = end - start
    print(f"Time taken for execution is: {total_time: .2f} seconds")
    return total_time

def eval_model(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               accuracy_fn):
    """Returns dictionary containing the results of model predicting on data_loader"""

    loss, acc = 0, 0
    model.eval()
    
    with torch.inference_mode():
        for X, y in tqdm(data_loader):
            y_pred = model(X)
            loss += loss_fn(y_pred, y)
            acc += accuracy_fn(y, y_pred.argmax(dim=1))
          
        loss /= len(data_loader)    
        acc /= len(data_loader) 
    
    return {"model_name:": model.__class__.__name__,
            "model_loss:": loss.item(),
            "model_acc:": acc}  

# 5. The Training Loop

torch.manual_seed(42)


def train_step(model: torch.nn.Module,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim,
               dataloader: torch.utils.data.DataLoader,
               accuracy_fn
               ):
    
    train_loss, train_acc = 0, 0
    
    for batch, (X, y) in enumerate(dataloader):        
        model.train()
        y_pred = model(X)
        
        loss = loss_fn(y_pred, y)
        train_loss += loss
        train_acc += accuracy_fn(y, y_pred.argmax(dim=1))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss /= len(dataloader)
    train_acc /= len(dataloader)
    print(f"Train loss: {train_loss: .5f} | Train acc: {train_acc: .2f}%")
        

def test_step(model: torch.nn.Module,
              loss_fn: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              accuracy_fn
              ):
    test_loss, test_acc = 0, 0
    
    model.eval()
    with torch.inference_mode():
        for X_test, y_test in dataloader:
            test_pred = model(X_test)
            test_loss += loss_fn(test_pred, y_test)
            test_acc += accuracy_fn(y_true=y_test, y_pred=test_pred.argmax(dim=1))
        
        test_loss /= len(dataloader)
        test_acc /= len(dataloader)        
        print(f"Test loss: {test_loss: .5f} | Test acc: {test_acc: .2f}%")
        
epochs = 3

train_time_start = timer()

for epoch in tqdm(range(epochs)):
    print(f"Epoch: {epoch}\n ---------")
    train_step(model_2, loss_fn, optimizer, train_dataloader, accuracy_fn)
    test_step(model_2, loss_fn, test_dataloader, accuracy_fn)
       
train_time_end = timer()

total_train_time_model_2 = print_exec_time(train_time_start, train_time_end)     

model_2_results = eval_model(model=model_2,
                             loss_fn=loss_fn,
                             accuracy_fn=accuracy_fn,
                             data_loader=test_dataloader)

# print(model_2_results)


def make_predictions(model: torch.nn.Module,
                     data: list,
                     device: torch.device = device):
    pred_probs = []
    model.to(device)
    model.eval()
    with torch.inference_mode():
        for sample in data:
            # prepare the sample (add a batch dimension and pass to target device)
            sample = torch.unsqueeze(sample, dim=0).to(device)
            
            # forward pass
            pred_logit = model(sample)
            
            # prediction probability
            pred_prob = torch.softmax(pred_logit.squeeze(), dim=0)
    
            pred_probs.append(pred_prob.cpu())
            
    # stack pred probs to turn list to tensor
    return torch.stack(pred_probs, dim=0)
    

import random
# random.seed(42)

test_samples = []
test_labels = []

for sample, label in random.sample(list(test_data), k=9):
    test_samples.append(sample)
    test_labels.append(label)
   
pred_probs = make_predictions(model=model_2, data=test_samples)

pred_classes = pred_probs.argmax(dim=1)

plt.figure(figsize=(9, 9))

nrows = 3
ncols = 3

for i, sample in enumerate(test_samples):
    # create subplot
    plt.subplot(nrows, ncols, i+1)
    
    # plot the image
    plt.imshow(sample.squeeze(), cmap="gray")
    
    # find the predictions in text form eg "Sandal"
    pred_label = class_names[pred_classes[i]]
    
    # get the truth labels
    truth_label = class_names[test_labels[i]]
    
    # create a title for the plot
    title_text = f"Pred: {pred_label} | Truth: {truth_label}"
    
    # check for equality between pred and truth and change color of title text 
    if pred_label == truth_label:
        plt.title(title_text, fontsize=10, c="g") # green text if prediction same as truth
    else:
        plt.title(title_text, fontsize=10, c="r")
        
    plt.axis(False)
    
# plt.show()  


### Making a confusion matrix
# - Make predictions with our trained model
# - Make a confusion matrix `torchmetrics.ConfusionMatrix`
# - Plot the confusion matrix with `mlxtend.plotting.plot_confusion_matrix()`

# 1. Make predictions with trained model
y_preds = []
model_2.eval()
with torch.inference_mode():
  for X, y in tqdm(test_dataloader, desc="Making predictions"):
    # Send data and targets to target device
    X, y = X.to(device), y.to(device)
    
    # Do the forward pass
    y_logit = model_2(X)
    
    # Turn predictions from logits -> prediction probabilities -> predictions labels
    y_pred = torch.softmax(y_logit, dim=1).argmax(dim=1) # note: perform softmax on the "logits" dimension, not "batch" dimension (in this case we have a batch size of 32, so can perform on dim=1)
    
    # Put predictions on CPU for evaluation
    y_preds.append(y_pred.cpu())
    
# Concatenate list of predictions into a tensor
y_pred_tensor = torch.cat(y_preds)  


from torchmetrics import ConfusionMatrix
from mlxtend.plotting import plot_confusion_matrix

# 2. Setup confusion matrix instance and compare predictions to targets
confmat = ConfusionMatrix(num_classes=len(class_names), task='multiclass')
confmat_tensor = confmat(preds=y_pred_tensor,
                         target=test_data.targets)

# 3. Plot the confusion matrix
fig, ax = plot_confusion_matrix(
    conf_mat=confmat_tensor.numpy(), # matplotlib likes working with NumPy 
    class_names=class_names, # turn the row and column labels into class names
    figsize=(10, 7)
);

plt.show()


    