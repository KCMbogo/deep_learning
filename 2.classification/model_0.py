from data import *
from helper_fn import plot_predictions, plot_decision_boundary

device = "cuda" if torch.cuda.is_available() else "cpu"

# create the model
class CircleModelV0(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(in_features=2, out_features=5)
        self.layer_2 = nn.Linear(in_features=5, out_features=1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer_2(self.layer_1(x))
    
# create the object
model_0 = CircleModelV0().to(device)

### alternative nn.Sequential()
# class CircleModelV0(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.layers = nn.Sequential(
#             nn.Linear(in_features=2, out_features=5),
#             nn.Linear(in_features=5, out_features=1)
#         )
        # no need for a forward method

# print(model_0.state_dict())

### Make untrained predictions
# with torch.inference_mode():
#     untrained_preds = model_0(X_test.to(device))
# print(f"Untrained preds: \n{untrained_preds[:10]}\n Untrained shape: {untrained_preds.shape}")
# print(f"Expected preds: \n{y_test[:10]}\n Expected shape: {y_test.shape}")


### Choose a loss fn and optimizer and create an accuracy fn to measure the accuracy of the model
loss_fn = nn.BCEWithLogitsLoss() # has sigmoid activation function built in

optimizer = torch.optim.SGD(params=model_0.parameters(),
                            lr=0.01)

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

### using activation function; raw logits -> prediction probabilities -> prediction labels 
with torch.inference_mode():
    y_test_logits = model_0(X_test.to(device))

# the sigmoid activation fn
y_preds_prob = torch.sigmoid(y_test_logits)
y_preds_prob = torch.round(y_preds_prob)

torch.manual_seed(42)
torch.cuda.manual_seed(42) # for cuda 

epochs = 1000

X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)

# note BCELoss expects probabilities but BCELossWithLogits expects logits

for epoch in range(epochs):
    model_0.train()
    
    # forward pass
    y_logits = model_0(X_train).squeeze()
    y_pred = torch.round(torch.sigmoid(y_logits))
    
    # calculate loss and accuracy
    # note: use logits b'cause BCEWithLogitsLoss() expects logits
    loss = loss_fn(y_logits, y_train)
    acc = accuracy_fn(y_true=y_train, y_preds=y_pred)
    
    # zero grad
    optimizer.zero_grad()
    
    # loss backward
    loss.backward()
    
    # step
    optimizer.step()
    
    # test the model
    model_0.eval()
    
    with torch.inference_mode():
        test_logits = model_0(X_test).squeeze()
        test_preds = torch.round(torch.sigmoid(test_logits))
        
        test_loss = loss_fn(test_logits, y_test)
        test_acc = accuracy_fn(y_true=y_test, y_preds=test_preds)

    if epoch % 100 == 0:
        print(f"Epoch: {epoch} | Loss: {loss: .5f} | Acc: {acc: .2f}% | Test Loss: {test_loss: .5f} | Test Accuracy: {test_acc: .2f}%")
    

# Plot decision boundary of the model
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Train")
plot_decision_boundary(model_0, X_train, y_train)
plt.subplot(1, 2, 2)
plt.title("Test")
plot_decision_boundary(model_0, X_test, y_test)
# plt.show()        


### improving the model by either:
# -adding more layers
# -adding more hidden units
# -changing the activation function
# -changing the loss function
# -change the learning rate 
# -fit for longer

