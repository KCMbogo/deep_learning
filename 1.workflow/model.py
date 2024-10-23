from data import *
import numpy as np
from pathlib import Path # for saving the model

## create a model

class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.weight = nn.Parameter(torch.randn(1, # start with random weight then adjust to ideal weight
                                               requires_grad=True, # it is default and used for gradient descent
                                               dtype=torch.float))
        
        self.bias = nn.Parameter(torch.randn(1, # start with random bias then adjust to ideal bias
                                             requires_grad=True,
                                             dtype=torch.float))
        
    # forward method to define computation performed at each call in the model
    def forward(self, x: torch.Tensor) -> torch.Tensor: # x is the input data 
        return self.weight * x + self.bias # linear regression formula  
    

## generate predictions

torch.manual_seed(42)
model_0 = LinearRegressionModel()

print(list(model_0.parameters()), '\n') # give parameter output
print(model_0.state_dict(), '\n') # give a dictionary output

# creating predictions using torch.inference_mode()

# with torch.inference_mode():
#     y_preds = model_0(X_test)

# print(f"Model output: \n{y_preds} \nExpected output: \n{y_test} \n")

# plot_predictions(predictions=y_preds)


## train model

# set the loss function
loss_fn = nn.L1Loss() # MAE - Mean Absolute Error

# set up the optimizer SGD(Stochastic(random) Gradient Descent)
optimizer = torch.optim.SGD(params=model_0.parameters(), # adjusts the parameters in our model
                            lr=0.01) # the level of adjusting the values


## build the training loop

torch.manual_seed(42) # the optimizer.step() will be random so this makes it reproducible

# epoch is one loop thru data
epochs = 1000

# Track model performance
epoch_count = []
loss_values = []
test_loss_values = []

# 0. loop the data
for epoch in range(epochs):
    # set model to train mode
    model_0.train() # sets all params that require grad to require grad
    
    # 1. forward pass - move thru mode front to back to make predictions
    y_pred = model_0(X_train)
    
    # 2. calculate loss - compare forward pass to actual data
    loss = loss_fn(y_pred, y_train) # input, output
    
    # 3. optimizer zero grad
    optimizer.zero_grad()
    
    # 4. loss backward - move from back to front to calc the gradient => backpropagation
    loss.backward()
    
    # 5. optimizer step - use optim to adjust the models params => gradient descent
    optimizer.step()
   
    model_0.eval() # turns off settings not needed for evaluation
    
    ### Testing
    
    with torch.inference_mode(): # turns off grad tracking
        test_pred = model_0(X_test) # test data
        
        test_loss = loss_fn(test_pred, y_test) # calc the loss of the test
    
    if epoch % 100 == 0:
        epoch_count.append(epoch)
        loss_values.append(loss)
        test_loss_values.append(test_loss)
        
        print(f"Epoch: {epoch} | Test Loss: {test_loss} | Train Loss: {loss}\n")
        print(f"Current model parameters: {model_0.state_dict()} \nExpected model parameters: {weight, bias}\n")
        
with torch.inference_mode():
    y_preds_new = model_0(X_test)

# plot_predictions(predictions=y_preds_new)

# plot the models performance
plt.plot(epoch_count, np.array(torch.tensor(loss_values).numpy()), label="Traing Loss")
plt.plot(epoch_count, np.array(torch.tensor(test_loss_values).numpy()), label="Testing Loss")
plt.title("Traing & Testing Curve")
plt.ylabel("Loss")
plt.xlabel("Epochs")
plt.legend()
plt.show()


### Saving the model

# create model directory
# MODEL_PATH = Path("../.models")
# MODEL_PATH.mkdir(parents=True, exist_ok=True)

# create model save path
# MODEL_NAME = "01_workflow_linear_regression_model.pt"
# MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

# save the model state dict -  recommended but also save model
# torch.save(obj=model_0.state_dict(),
#            f=MODEL_SAVE_PATH)




## load model

# load_model_0 = LinearRegressionModel()

# load_model_0.load_state_dict(torch.load(f=MODEL_SAVE_PATH))