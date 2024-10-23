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

# 1. Prepare the data

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

# print(f" Test batches: {len(test_dataloader)} batches, Train batches: {len(train_dataloader)} batches")

class_names = train_data.classes

train_data_batch, train_label_batch = next(iter(train_dataloader))

torch.manual_seed(42)
random_idx = torch.randint(0, len(train_data_batch), size=[1]).item()
img, label = train_data_batch[random_idx], train_label_batch[random_idx]

# plt.imshow(img.squeeze(), cmap='gray')
# plt.title(class_names[label])
# plt.axis(False)
# plt.show()


# 2.The Model

class FashionMNISTModelV1(nn.Module):
    def __init__(self, in_shape: int, hidden_units: int, out_shape: int):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_shape, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.layer_stack(x)
    
model_1 = FashionMNISTModelV1(in_shape=28*28,
                              hidden_units=10,
                              out_shape=len(class_names))


# 3. Loss fn and Optimizer

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model_1.parameters(),
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

train_time_start = timer()

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

for epoch in tqdm(range(epochs)):
    print(f"Epoch: {epoch}\n ---------")
    train_step(model_1, loss_fn, optimizer, train_dataloader, accuracy_fn)
    test_step(model_1, loss_fn, test_dataloader, accuracy_fn)
       
train_time_end = timer()

total_train_time_model_1 = print_exec_time(train_time_start, train_time_end)     

model_1_results = eval_model(model=model_1,
                             loss_fn=loss_fn,
                             accuracy_fn=accuracy_fn,
                             data_loader=test_dataloader)

print(model_1_results)
        
        