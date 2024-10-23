import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np


device = "cuda" if torch.cuda.is_available() else "cpu"

## prepare the data to load for the model

# --linear regression - y = b + wx

# data required where x is the input 
w = 0.7
b = 0.3
x = torch.arange(0, 1, 0.02).unsqueeze(dim=1)

# y is the output
y = w * x + b

# print(x[:10], y[:10])

# set the training data and the test data -> 4:1(0.8:0.2 or 80% to 20%)
train_split = int(0.8 * len(x)) # input

x_train, y_train = x[:train_split], y[:train_split] 
x_test, y_test = x[train_split:], y[train_split:]

# visualizing the current data

def plot_fn(train_data=x_train,
            train_label=y_train,
            test_data=x_test,
            test_label=y_test,
            predictions=None):
    
    """Plots the training data, test data and prediction if available
    Args:
        train_data (_type_, tensor): _description_. training input.
        train_label (_type_, tensor): _description_. training expected output.
        test_data (_type_, tensor): _description_. test input.
        test_label (_type_, tensor): _description_. expected test output.
        predictions (_type_, tensor): _description_. model's output.
    """
    
    plt.scatter(train_data, train_label, c="b", s=4, label="Training Data")
    plt.scatter(test_data, test_label, c="g", s=4, label="Test Data")
    
    if predictions != None:
        plt.scatter(test_data, predictions, c="r", s=4, label="Predictions")
        
    plt.ylabel("Output(y)")
    plt.xlabel("Input(x)")

    plt.legend(prop={"size": 14})
    plt.show()


## building the model

class LinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        
        # define the parameters for the model, this will auto create weight and bias 
        self.linear_layer = nn.Linear(in_features=1,
                                      out_features=1)
    
    def forward(self, x: torch.tensor) -> torch.tensor:
        return self.linear_layer(x)
    

torch.manual_seed(42)

model = LinearRegression()
model.to(device) # this moves the model to gpu if available
print(next(model.parameters()).device)

## train the model 

# set the loss function
loss_fn = nn.L1Loss() # MAE

# set optimizer
optimizer = torch.optim.SGD(params=model.parameters(),
                            lr=0.01)

# training loop
torch.manual_seed(42)

epochs = 200

# track model predictions
epoch_count =[]
train_loss_values = []
test_loss_values = []

# move the varibles to the same device as model
x_test = x_test.to(device)
x_train = x_train.to(device)
y_test = y_test.to(device)
y_train = y_train.to(device)

for epoch in range(epochs):
    # initialize the trainig mode
    model.train()
    
    # forward pass - generating predictions\
    y_preds = model(x_train)
    
    # calculate loss
    loss = loss_fn(y_preds, y_train)
    
    # clear cached gradients
    optimizer.zero_grad()
    
    # backpropagation
    loss.backward()
    
    # update the parameters
    optimizer.step()
    
    
    # turn on evaluation mode
    model.eval()
    
    ## test the model and calc the loss
    with torch.inference_mode():
        test_preds = model(x_test)        
        test_loss =  loss_fn(test_preds, y_test)

    if epoch % 20 == 0:
        epoch_count.append(epoch)
        test_loss_values.append(test_loss)
        train_loss_values.append(loss)
        print(f"Train Loss: {loss}\n Test Loss: {test_loss}\n")
        
with torch.inference_mode():
    y_preds_new = model(x_test)
    
# many data science libs work on cpu eg matplotlib, numpy, pandas
plot_fn(predictions=y_preds_new.cpu())

## plot models performance

plt.plot(epoch_count, np.array(torch.tensor(train_loss_values).numpy()), label="Training Loss")
plt.plot(epoch_count, np.array(torch.tensor(test_loss_values).numpy()), label="Testing Loss")
plt.title("Traing & Testing Curve")
plt.ylabel("Loss")
plt.xlabel("Epochs")
plt.legend()
plt.show()    
 

