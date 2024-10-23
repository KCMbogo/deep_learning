from data import *
from helper_fn import accuracy_fn
from timeit import default_timer as timer
from tqdm.auto import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"

class FashionMNISTModelV0(nn.Module):
    def __init__(self, in_shape: int, 
                 hidden_units: int, 
                 out_shape: int):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Flatten(), # turn 3D or 2D tensor to a vector
            nn.Linear(in_features=in_shape, out_features=hidden_units),
            nn.Linear(in_features=hidden_units, out_features=out_shape)
        )
        
    def forward(self, x):
        return self.layer_stack(x)    
    
model_0 = FashionMNISTModelV0(in_shape=28*28, # because nn.Flatten() changes [1, 28, 28] -> [1, 28*28]
                              hidden_units=10,
                              out_shape=len(class_names)).to(device)    


# SETUP LOSS FN AND OPTIMIZER

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model_0.parameters(),
                            lr=0.1)

# function to calc the exec time

def print_train_time(start: float,
                     end: float,
                     device: torch.device = None):
    """Print the total time taken to execute certain code
    Args:
        start (float): start time
        end (float): end time
        device (torch.device, optional): device running on. Defaults to None.
    """
    total_time = end - start
    print(f"Train time on: {device}: {total_time: .3f} seconds")
    return total_time

# example
# start = timer()
# #... some code ...
# end = timer()
# print_train_time(start, end, 'cpu')
    

### SET THE TRAINING LOOP 
# - Loop thru the epochs
# - Loop thru training batches, p4m train steps, calc train loss per batch
# - Loop thru testing batches, p4m test steps, calc test loss per batch
# - Print what's happenin'
# - Time it all
 
torch.manual_seed(42)

train_time_start_on_cpu = timer()

epochs = 3
 
for epoch in tqdm(range(epochs)):
    print(f"Epoch: {epoch}\n------")
    
    ### Training
    train_loss = 0
    
    # loop thru train batches
    for batch, (X_train, y_train) in enumerate(train_dataloader):
        model_0.train()
        
        # forward pass
        y_pred = model_0(X_train)
        
        # calc the loss per batch then add to overall train_loss
        loss = loss_fn(y_pred, y_train)
        train_loss += loss
        
        # zero grad
        optimizer.zero_grad()
        
        # loss backward()
        loss.backward()
        
        # step
        optimizer.step()
        
        # print what's happenin'
        if batch % 400 == 0:
            print(f"Looked at {batch * len(X_train)} / {len(train_dataloader.dataset)} samples.")
            
    # train loss average 
    train_loss /= len(train_dataloader)
    
    ### Testing
    test_loss = 0
    test_acc = 0
    
    model_0.eval()
    with torch.inference_mode():
        for X_test, y_test in test_dataloader:
            # forward pass
            test_pred = model_0(X_test)
            
            # calc loss
            test_loss += loss_fn(test_pred, y_test)
            
            # calc acc  
            test_acc += accuracy_fn(y_true=y_test, y_pred=test_pred.argmax(dim=1))
         
        # average test loss
        test_loss /= len(test_dataloader)
        
        # average test acc
        test_acc /= len(test_dataloader)
        
    # print what's happenin'
    print(f"Train loss: {test_loss: .4f} | Test loss: {test_loss: .4f} | Test acc: {test_acc: .4f} | Test pred shape: {test_pred.shape}") 
    
# calc train time
train_time_end_on_cpu = timer()
total_train_time_model_0 = print_train_time(start=train_time_start_on_cpu,
                 end=train_time_end_on_cpu,
                 device=str(next(model_0.parameters()).device))


# EVALUATE THE MODEL

torch.manual_seed(42)

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

model_0_results = eval_model(model=model_0,
                             data_loader=test_dataloader,
                             loss_fn=loss_fn,
                             accuracy_fn=accuracy_fn)

print(model_0_results)










