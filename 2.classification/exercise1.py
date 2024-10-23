import torch
import matplotlib.pyplot as plt
import torch.nn as nn
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from torchmetrics import Accuracy
from helper_fn import plot_decision_boundary

device = "cuda" if torch.cuda.is_available() else "cpu"

X_moon, y_moon = make_moons(n_samples=1000,
                  noise=0.01,
                  random_state=42)

X_moon = torch.from_numpy(X_moon).type(torch.float)
y_moon = torch.from_numpy(y_moon).type(torch.float)

X_moon_train, X_moon_test, y_moon_train, y_moon_test = train_test_split(X_moon, y_moon,
                                                                        test_size=0.2)

class MoonModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(),
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
        
    def forward(self, x):
        return self.layer_stack(x)
        
model = MoonModel().to(device)

loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(params=model.parameters(),
                            lr=0.1)

X_moon_train, y_moon_train = X_moon_train.to(device), y_moon_train.to(device)
X_moon_test, y_moon_test = X_moon_test.to(device), y_moon_test.to(device)

def accuracy_fn(y_true, y_preds):
    correct = torch.eq(y_true, y_preds).sum().item()
    acc = (correct / len(y_preds)) * 100
    return acc
    
# accuracy_fn = Accuracy(task='binary')

torch.manual_seed(42)
torch.cuda.manual_seed(42)

epochs = 1000

for epoch in range(epochs):
    model.train()
    
    logits = model(X_moon_train).squeeze()
    preds = torch.round(torch.sigmoid(logits))
    
    loss = loss_fn(logits, y_moon_train)
    acc = accuracy_fn(y_moon_train, preds)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    model.eval()
    
    with torch.inference_mode():
        test_logits = model(X_moon_test).squeeze()
        test_preds = torch.round(torch.sigmoid(test_logits))
        
        test_loss = loss_fn(test_logits, y_moon_test)
        test_acc = accuracy_fn(y_moon_test, test_preds)
        
    if epoch % 100 == 0:
        print(f"Epoch: {epoch} | Train loss: {loss: .4f} | Train acc: {acc: .2f}% | Test loss: {test_loss: .4f} | Test acc: {test_acc: .2f}%")
        
model.eval()

with torch.inference_mode():
    test_logits = model(X_moon_test).squeeze()
    test_preds = torch.round(torch.sigmoid(test_logits))

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title('Train')
plot_decision_boundary(model, X_moon_train, y_moon_train)
plt.subplot(1, 2, 2)
plt.title('Test')
plot_decision_boundary(model, X_moon_test, y_moon_test)
plt.show()











