import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from helper_fn import plot_decision_boundary

### PREPARE THE DATA
NUM_CLASS = 4
NUM_FEATURES = 2
RANDOM_SEED = 42

X_blob, y_blob = make_blobs(n_samples=1000,
                            n_features=NUM_FEATURES,
                            centers=NUM_CLASS,
                            random_state=RANDOM_SEED,
                            cluster_std=1.5)

# Turn to tensor because the sklearn is a numpy array
X_blob = torch.from_numpy(X_blob).type(torch.float)
y_blob = torch.from_numpy(y_blob).type(torch.LongTensor)

# set the training and testing data
X_blob_train, X_blob_test, y_blob_train, y_blob_test = train_test_split(X_blob,
                                                                        y_blob,
                                                                        test_size=0.2)
# Visualize the data

# plt.figure(figsize=(10, 7))
# plt.scatter(X_blob[:, 0], X_blob[:, 1], c=y_blob, cmap='RdYlBu')
# plt.show()

# accuracy function
def accuracy_fn(y_true, y_preds):
    correct = torch.eq(y_true, y_preds).sum().item()
    acc = (correct / len(y_preds)) * 100
    return acc


### CREATE THE MODEL

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class BlobModel(nn.Module):
    def __init__(self, in_features, out_features, hidden_units=8):
        """Initializes the in and out features of the layer and its hidden units

        Args:
            in_features (int): the number of input features or dimensions
            out_features (int): the number of expected output features or dimension
            hidden_units (int, optional): the number of hidden units betweeb the layers. Defaults to 8.
        """
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Linear(in_features, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, out_features)
        )
    
    def forward(self, x):
        return self.layer_stack(x)
    
# print(torch.unique(y_blob_train))   # this classes the unique outputs the are expected
    
model_4 = BlobModel(in_features=2,
                    out_features=4,
                    hidden_units=8).to(device)

### CREATE THE LOSS FN, OPTIMER AND TRAIN LOOP

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model_4.parameters(),
                            lr=0.1)

# see the current model_4 preds: logits -> probs -> preds

# model_4.eval()
# with torch.inference_mode():
#     logits = model_4(X_blob_test.to(device))

# probs = torch.softmax(logits, dim=1) 
# preds = torch.argmax(probs, dim=1) # returns the highest prob in each row of dim 1
# print(preds[:10], y_blob_test[:10])
# print(probs[:10], torch.sum(probs[0])) # -> the sum of these preds should alway be 1


# move the variables to the specific device
X_blob_train, X_blob_test = X_blob_train.to(device), X_blob_test.to(device)
y_blob_train, y_blob_test = y_blob_train.to(device), y_blob_test.to(device)

# the loop

torch.manual_seed(42)
torch.cuda.manual_seed(42)

epochs = 100

for epoch in range(epochs):
    # train mode
    model_4.train()
    
    # logits -> probs -> preds
    logits = model_4(X_blob_train)
    preds = torch.softmax(logits, dim=1).argmax(dim=1)
    
    # loss_fn
    loss = loss_fn(logits, y_blob_train)
    acc = accuracy_fn(y_blob_train, preds)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    model_4.eval()
    
    with torch.inference_mode():
        test_logits = model_4(X_blob_test)
        test_preds = torch.softmax(test_logits, dim=1).argmax(dim=1)
        
        test_loss = loss_fn(test_logits, y_blob_test)
        test_acc = accuracy_fn(y_blob_test, test_preds)
        
    if epoch % 10 == 0:
        print(f"Epoch: {epoch} | Train loss: {loss: .4f} | Train acc: {acc: .2f}% | Test loss: {test_loss: .4f} | Test acc: {test_acc: .2f}%")


### PLOT PREDICTIONS

with torch.inference_mode():
    y_logits = model_4(X_blob_test)
    y_preds = torch.softmax(y_logits, dim=1).argmax(dim=1)
    
# Plot decision boundary of the model
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Train")
plot_decision_boundary(model_4, X_blob_train, y_blob_train)
plt.subplot(1, 2, 2)
plt.title("Test")
plot_decision_boundary(model_4, X_blob_test, y_blob_test)
plt.show()     
