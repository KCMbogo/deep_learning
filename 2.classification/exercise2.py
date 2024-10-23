import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split
from torchmetrics.classification import Accuracy
from helper_fn import plot_decision_boundary

### PREPARE THE DATA

N = 100 # number of points per class
D = 2 # dimensionality
K = 3 # number of classes
X = np.zeros((N*K, D)) # data matrix (each row = single example)
y = np.zeros(N*K, dtype='uint8') # class labels

for j in range(K):
  ix = range(N*j, N*(j+1))
  r = np.linspace(0.0,1,N) # radius
  t = np.linspace(j*4,(j+1)*4,N) + np.random.randn(N)*0.2 # theta
  X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
  y[ix] = j
  
X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.long) 

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=42)
  
# lets visualize the data:
# plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.viridis)
# plt.show()

### THE MODEL

class SpiralModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(),
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 3)
        ) 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer_stack(x)


model = SpiralModel()

### THE TRAINING

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters(),
                              lr=0.1) 
accuracy = Accuracy(task="multiclass",
                    num_classes=3)

torch.manual_seed(42)
torch.cuda.manual_seed(42)

epochs = 250

for epoch in range(epochs):
    model.train()
    
    logits = model(X_train)
    preds = torch.softmax(logits, dim=1).argmax(dim=1)
    
    loss = loss_fn(logits, y_train)
    acc = accuracy(preds, y_train)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    model.eval()
    with torch.inference_mode():
        test_logits = model(X_test)
        test_preds = torch.softmax(test_logits, dim=1).argmax(dim=1)
        test_loss = loss_fn(test_logits, y_test)
        test_acc = accuracy(test_preds, y_test)
    
    if epoch % 25 == 0:
        print(f"Epoch: {epoch} | Loss: {loss: .4f} | Acc: {acc * 100: .2f}% | Test loss: {test_loss: .4f} | Test acc: {test_acc * 100: .2f}%")

model.eval()

with torch.inference_mode():
    test_logits = model(X_test)
    test_preds = torch.softmax(test_logits, dim=1).argmax(dim=1)
    test_loss = loss_fn(test_logits, y_test)
    test_acc = accuracy(test_preds, y_test)   

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Train")
plot_decision_boundary(model, X_train, y_train)
plt.subplot(1, 2, 2)
plt.title("Test")
plot_decision_boundary(model, X_test, y_test)
plt.show()    
    