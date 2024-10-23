import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from timeit import default_timer as timer
from helper_fn import accuracy_fn
from tqdm.auto import tqdm
import torchmetrics, mlxtend

device = "cuda" if torch.cuda.is_available() else "cpu"

train_data = datasets.MNIST(
    root="data",
    download=True,
    transform=ToTensor(),
    train=True,
    target_transform=None
)

test_data = datasets.MNIST(
    root="data",
    download=True,
    train=False,
    transform=ToTensor(),
    target_transform=None
)

BATCH_SIZE = 32

train_dataloader = DataLoader(
    batch_size=BATCH_SIZE,
    dataset=train_data,
    shuffle=True
)

test_dataloader = DataLoader(
    batch_size=BATCH_SIZE,
    dataset=test_data,
    shuffle=True
)


class_names = train_data.classes
# num, label = next(iter(train_data))    
# print(f"The image shape is: {num.shape}")
# print(f"The label is: {label} | class name is: {class_names[label]}")
# print(f"Classes are: {class_names}")
# print(f"Amount of train data: {len(train_data)}")
# print(f"Amount of test data: {len(test_data)}")

# fig = plt.figure(figsize=(9, 9))
# row, col = 4, 4
# for i in range(1, row*col+1):
#     random_idx = torch.randint(0, len(train_data), size=[1]).item()
#     num, label = train_data[random_idx]
#     fig.add_subplot(row, col, i)
#     plt.imshow(num.squeeze(), cmap="gray")
#     plt.title(class_names[label])
#     plt.axis(False) 
# plt.show()


class MNISTModel(nn.Module):
    def __init__(self, in_shape: int,
                 out_shape: int,
                 hidden_units: int):
        super().__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(
                in_channels=in_shape,
                out_channels=hidden_units,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=hidden_units,
                out_channels=hidden_units,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(
                in_channels=hidden_units,
                out_channels=hidden_units,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=hidden_units,
                out_channels=hidden_units,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units*49, out_features=10)
        )
        
    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.classifier(x)
        return x

torch.manual_seed(42)
model = MNISTModel(in_shape=1, hidden_units=10, out_shape=len(class_names))

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model.parameters(),
                            lr=0.1)


def print_exec_time(start, end):
    total_time = end - start
    print(f"The time taken to execute is: {total_time} sec")
    return total_time


def eval_model(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               accuracy_fn: accuracy_fn):
    loss, acc = 0, 0
    
    model.eval()
    with torch.inference_mode():
        for X, y in tqdm(dataloader):
            y_pred = model(X)
            loss += loss_fn(y_pred, y)
            acc += accuracy_fn(y, y_pred.argmax(dim=1))
        loss /= len(dataloader)
        acc /= len(dataloader)
    
    return {
        "model_name": model.__class__.__name__,
        "model_acc": acc,
        "model_loss": loss
    }    

start = timer()
def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn,
               optimizer: torch.optim,
               accuracy_fn):
    train_acc, train_loss = 0, 0
    for batch, (X, y) in enumerate(dataloader):
        model.train()
        y_pred = model(X)
        
        loss = loss_fn(y_pred, y)
        train_loss += loss
        train_acc += accuracy_fn(y, y_pred.argmax(dim=1))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if batch % 400 == 0:
            print(f"Looked at: {batch * BATCH_SIZE} / {len(train_dataloader) * BATCH_SIZE}")
        
    train_loss /= len(dataloader)
    train_acc /= len(dataloader)
    print(f"Train loss: {train_loss: .4f} | Train acc: {train_acc: .2f}%")


def test_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn,
               accuracy_fn):
    test_acc, test_loss = 0, 0
    model.eval()
    with torch.inference_mode():
        for X_test, y_test in dataloader:
            y_pred = model(X_test)
            test_loss += loss_fn(y_pred, y_test)
            test_acc += accuracy_fn(y_test, y_pred.argmax(dim=1))
        
        test_loss /= len(dataloader)
        test_acc /= len(dataloader)
        print(f"Test loss: {test_loss: .4f} | Test acc: {test_acc: .2f}%")
        

epochs = 3

torch.manual_seed(42)
for epoch in range(epochs):
    print(f"\nEpoch {epoch}......\n")
    train_step(model=model, dataloader=train_dataloader, loss_fn=loss_fn, optimizer=optimizer, accuracy_fn=accuracy_fn)
    test_step(model=model, dataloader=train_dataloader, loss_fn=loss_fn, accuracy_fn=accuracy_fn)
    
end = timer()

total_exec_time = print_exec_time(start, end)

model_results = eval_model(model=model,
                           dataloader=test_dataloader,
                           loss_fn=loss_fn,
                           accuracy_fn=accuracy_fn)

print(model_results)
        
def make_predictions(model: torch.nn.Module,
                     data: list,
                     device: torch.device = "cpu"):
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
random.seed(42)

test_samples = []
test_labels = []

for sample, label in random.sample(list(test_data), k=9):
    test_samples.append(sample)
    test_labels.append(label)
    
pred_probs = make_predictions(model=model, data=test_samples)

pred_classes = pred_probs.argmax(dim=1)

plt.figure(figsize=(9, 9))

nrows = 3
ncols = 3

for i, sample in enumerate(test_samples):
    plt.subplot(nrows, ncols, i+1)
    
    plt.imshow(sample.squeeze(), cmap="gray")
    
    pred_label = class_names[pred_classes[i]]
    
    truth_label = class_names[test_labels[i]]
    
    title_text = f"Pred: {pred_label} | Truth: {truth_label}"
    
    if pred_label == truth_label:
        plt.title(title_text, fontsize=10, c="g") 
    else:
        plt.title(title_text, fontsize=10, c="r")
        
    plt.axis(False)
    
# plt.show()

# CONFUSION MATRIX

# Make predictions across all test data
from tqdm.auto import tqdm
model.eval()
y_preds = []
with torch.inference_mode():
  for batch, (X, y) in tqdm(enumerate(test_dataloader)):
    # Make sure data on right device
    X, y = X.to(device), y.to(device)
    # Forward pass
    y_pred_logits = model(X)
    # Logits -> Pred probs -> Pred label
    y_pred_labels = torch.argmax(torch.softmax(y_pred_logits, dim=1), dim=1)
    # Append the labels to the preds list
    y_preds.append(y_pred_labels)
  y_preds=torch.cat(y_preds).cpu()
len(y_preds) 

from torchmetrics import ConfusionMatrix
from mlxtend.plotting import plot_confusion_matrix

# Setup confusion matrix 
confmat = ConfusionMatrix(task="multiclass", num_classes=len(class_names))
confmat_tensor = confmat(preds=y_preds,
                         target=test_data.targets)

# Plot the confusion matrix
fix, ax = plot_confusion_matrix(
    conf_mat=confmat_tensor.numpy(),
    class_names=class_names,
    figsize=(10, 7)
)

# plt.show()