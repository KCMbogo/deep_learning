### improving the model by either:
# -adding more layers
# -adding more hidden units
# -changing the activation function
# -changing the loss function
# -change the learning rate 
# -fit for longer

from data import *
from helper_fn import plot_predictions, plot_decision_boundary

import sys
import os
# Add the directory containing data.py to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../1.workflow')))
import data

device = "cuda" if torch.cuda.is_available() else "cpu"

### MODEL V1
# model with more layers and more hidden units
class CircleModelV1(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(2, 10)
        self.layer_2 = nn.Linear(10, 10)
        self.layer_3 = nn.Linear(10, 1)
        
    def forward(self, x):
        return self.layer_3(self.layer_2(self.layer_1(x)))
    
model_1 = CircleModelV1().to(device)    

# loss fn and optimizer
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(params=model_1.parameters(),
                            lr=0.1)

def accuracy_fn(y_true, y_preds):
    """
    y_true: is the expected true output
    y_preds: is the predicted output by the model
    This function calculates the accuracy of the prediction:
    formula: sum of true prediction / y_preds * 100
    """
    correct = torch.eq(y_true, y_preds).sum().item()
    acc = (correct / len(y_preds)) * 100
    return acc

torch.manual_seed(42)
torch.cuda.manual_seed(42)

epochs = 1000

for epoch in range(epochs):
    model_1.train()
    
    # forward pass
    y_logits = model_1(X_train).squeeze()
    y_pred = torch.round(torch.sigmoid(y_logits))
    
    # calculate the loss
    loss = loss_fn(y_logits, y_train)
    acc = accuracy_fn(y_train, y_pred)
    
    # zero grad
    optimizer.zero_grad()
    
    # loss backward
    loss.backward()
    
    # step
    optimizer.step()
    
    # testing
    model_1.eval()
    
    with torch.inference_mode():
        # forward pass
        test_logits = model_1(data.X_test).squeeze()
        test_preds = torch.round(torch.sigmoid(test_logits))
        
        test_loss = loss_fn(test_logits, data.y_test)
        test_acc = accuracy_fn(y_true=y_test, y_preds=test_preds)

    # if epoch % 100 == 0:
    #     print(f"Epoch: {epoch} | Loss: {loss: .5f} | Acc: {acc: .2f}% | Test Loss: {test_loss: .5f} | Test Accuracy: {test_acc: .2f}%")

# Plot decision boundary of the model
# plt.figure(figsize=(12, 6))
# plt.subplot(1, 2, 1)
# plt.title("Train")
# plot_decision_boundary(model_1, X_train, y_train)
# plt.subplot(1, 2, 2)
# plt.title("Test")
# plot_decision_boundary(model_1, X_test, y_test) 
# plt.show()  


# preparing data to test the model with linear regression
w = 0.7
b = 0.3

X_reg = torch.arange(0, 1, 0.02).unsqueeze(dim=1) 
y_reg = w * X_reg + b

train_split = int(0.8 * len(X_reg))
X_reg_train, y_reg_train = X_reg[:train_split], y_reg[:train_split]
X_reg_test, y_reg_test = X_reg[train_split:], y_reg[train_split:]

def plot_prediction(train_data=X_reg_train,
                     train_labels=y_reg_train,
                     test_data=X_reg_test,
                     test_labels=y_reg_test,
                     predictions=None):
    """
    Plots training data, test data and compares predictions.
    """
    
    # plot training data in blue
    plt.scatter(train_data, train_labels, s=4, c="b", label="Training data")
    
    # plot test data in green
    plt.scatter(test_data, test_labels, s=4, c="g", label="Testing data")
    
    # are there predictions?
    if predictions is not None:
        plt.scatter(test_data, predictions, c="r", s=4, label="Predictions")
        
    # show the legend
    plt.legend(prop={"size":14})
    plt.show()
    
# plot_prediction()

model_2 = nn.Sequential(
    nn.Linear(1, 10),
    nn.Linear(10, 10),
    nn.Linear(10, 1)
).to(device)

loss_fn = nn.L1Loss()
optimizer = torch.optim.SGD(params=model_2.parameters(),
                            lr=0.01)

torch.manual_seed(42)

epochs = 1000

for epoch in range(epochs):
    # training
    model_2.train()
    
    y_reg_pred = model_2(X_reg_train)
    loss = loss_fn(y_reg_pred, y_reg_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # testing
    model_2.eval()
    with torch.inference_mode():
        test_pred = model_2(X_reg_test)
        test_loss = loss_fn(test_pred, y_reg_test)   
    # if epoch % 100 == 0:
        # print(f"Train Loss: {loss} | Test Loss: {test_loss}")

# plot_prediction(X_reg_train, y_reg_train, X_reg_test, y_reg_test, test_pred)
    
    
### MODEL V3 NON LINEAR

class CircleModelV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(2, 24)
        self.layer_2 = nn.Linear(24, 32)
        self.layer_3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        return (self.layer_3(self.relu(self.layer_2(self.relu(self.layer_1(x))))))
    
model_3 = CircleModelV2().to(device)

loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(params=model_3.parameters(),
                            lr=0.1)

for epoch in range(epochs):
    model_3.train()
    
    logits = model_3(X_train).squeeze()
    preds = torch.round(torch.sigmoid(logits))
    
    loss = loss_fn(logits, y_train)
    acc = accuracy_fn(y_train, preds)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    model_3.eval()
    with torch.inference_mode():
        test_logits = model_3(X_test).squeeze()
        test_preds = torch.round(torch.sigmoid(test_logits))
        
        test_loss = loss_fn(test_logits, y_test)
        test_acc = accuracy_fn(y_test, test_preds)
    
    if epoch % 100 == 0:
        print(f"Epoch: {epoch} | Train Loss: {loss: .4f} | Train Acc: {acc: .2f}% | Test Loss: {test_loss: .4f} | Test Acc: {test_acc: .2f}%")

# Plot decision boundary of the model
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Train")
plot_decision_boundary(model_3, X_train, y_train)
plt.subplot(1, 2, 2)
plt.title("Test")
plot_decision_boundary(model_3, X_test, y_test) 
plt.show()  







