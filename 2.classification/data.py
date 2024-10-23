import torch
import torch.nn as nn
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split

n_samples = 1000

X, y = make_circles(n_samples=n_samples, # X and y are numpy arrays
                    noise=0.03,
                    random_state=42)
# print(f"X samples: \n{X[:10]}\n y samples: \n{y[:10]}")
# print(type(X), type(y))

# tabular visualization of data
data = {"X1": X[:, 0],
        "X2": X[:, 1],
        "labels": y}

circles = pd.DataFrame(data)
# print(circles.head(10))

### graphical presentation
# plt.scatter(x=X[:, 0],
#             y=X[:, 1],
#             c=y,
#             cmap=plt.cm.RdBu)
# plt.show()

# split to training and testing data
# first turn data to tensor
X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.float)

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2, # 20% test data
                                                    random_state=42 # same to manual_seed
                                                    )
