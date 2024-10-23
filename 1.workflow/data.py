import torch
from torch import nn # nn - neural networks
import matplotlib.pyplot as plt

# linear regression: y = ax +c or y = a + bx
# for y = a + bx: a = bias/intercept and b = weight/slope
weight = 0.7 
bias = 0.3

start = 0
end = 1
step = 0.02

X = torch.arange(start, end, step).unsqueeze(dim=1)
y = weight * X + bias # bx + a

print(X[:10], len(X), '\n\n')
print(y[:10], len(y), '\n')

# Two important sets of data: training set & test set

train_split = int(0.8 * len(X)) # training set 80% of X
X_train, y_train = X[:train_split], y[:train_split] # the first 80% are for training
X_test, y_test = X[train_split:], y[train_split:] # the 20% for test

print(f"Train sets: \n{X_train}, \n{y_train}\n")
print(f"Test sets: \n{X_test}, \n{y_test}\n\n")

# plotting a graph for the data
# note: data and label in DL represent features and targets
# features(data): these are inputs that are used to generate predictions
# targets(label): these are outputs that the models tries to predict

def plot_predictions(train_data=X_train,
                     train_labels=y_train,
                     test_data=X_test,
                     test_labels=y_test,
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
    
# plot_predictions()