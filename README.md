# Gradient Descent: The Ultimate Optimizer

![gdtuo_turtles](https://user-images.githubusercontent.com/31300675/193727211-bff82331-998c-4d44-b675-03d1fd222e0e.png)
# Abstract
Working with any gradient-based machine learning algorithm involves the tedious task of tuning the optimizer's hyperparameters, such as the step size. Recent work has shown how the step size can itself be "learned" on-line by gradient descent, by manually deriving expressions for "hypergradients" ahead of time.

We show how to *automatically* compute hypergradients with a simple and elegant modification to backpropagation. This allows us to apply the method to other hyperparameters besides the step size, such as the momentum coefficient. We can even recursively apply the method to its own *hyper*-hyperparameters, and so on *ad infinitum*. As these towers of optimizers grow taller, they become less sensitive to the initial choice of hyperparameters. We present experiments validating this for MLPs, CNNs, and RNNs.

*This repository contains an implementation of the algorithm in our paper.*

# Citation
```
@article{chandra2022gradient,
    title = {Gradient Descent: The Ultimate Optimizer},
    author = {Chandra, Kartik and Xie, Audrey and Ragan-Kelley, Jonathan and Meijer, Erik},
    journal = {NeurIPS},
    year = {2022},
    url = {https://arxiv.org/abs/1909.13371}
}
```

# Install
```
# install pytorch for your specific machine
...

# install our package
pip install -i https://test.pypi.org/simple/ gradient-descent-the-ultimate-optimizer
```
# Example
First, build the MLP and initialize data loaders as you would normally in PyTorch.
```python
import math
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

class MNIST_FullyConnected(nn.Module):
    """
    A fully-connected NN for the MNIST task. This is Optimizable but not itself
    an optimizer.
    """
    def __init__(self, num_inp, num_hid, num_out):
        super(MNIST_FullyConnected, self).__init__()
        self.layer1 = nn.Linear(num_inp, num_hid)
        self.layer2 = nn.Linear(num_hid, num_out)

    def initialize(self):
        nn.init.kaiming_uniform_(self.layer1.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.layer2.weight, a=math.sqrt(5))

    def forward(self, x):
        """Compute a prediction."""
        x = self.layer1(x)
        x = torch.tanh(x)
        x = self.layer2(x)
        x = torch.tanh(x)
        x = F.log_softmax(x, dim=1)
        return x

BATCH_SIZE = 256
EPOCHS = 5
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

mnist_train = torchvision.datasets.MNIST('./data', train=True, download=True, transform=torchvision.transforms.ToTensor())
mnist_test = torchvision.datasets.MNIST('./data', train=False, download=True, transform=torchvision.transforms.ToTensor())
dl_train = torch.utils.data.DataLoader(mnist_train, batch_size=BATCH_SIZE, shuffle=True)
dl_test = torch.utils.data.DataLoader(mnist_test, batch_size=10000, shuffle=False)

model = MNIST_FullyConnected(28 * 28, 128, 10).to(DEVICE)
```
Next, import our package and initialize a stack of hyperoptimizers. This example uses the stack `Adam/SGD`.
```python
from gradient_descent_the_ultimate_optimizer import gdtuo

optim = gdtuo.Adam(optimizer=gdtuo.SGD(1e-5))
```
`gdtuo.ModuleWrapper` allows any `nn.Module` to be optimized by hyperoptimizers.
```python
mw = gdtuo.ModuleWrapper(model, optimizer=optim)
mw.initialize()
```
Lastly, use `mw` instead of a PyTorch optimizer to optimize the model. The train loop is nearly identical to what you would typically implement in PyTorch (differences are marked by comments).
```python
for i in range(1, EPOCHS+1):
    running_loss = 0.0
    for j, (features_, labels_) in enumerate(dl_train):
        mw.begin() # call this before each step, enables gradient tracking on desired params
        features, labels = torch.reshape(features_, (-1, 28 * 28)).to(DEVICE), labels_.to(DEVICE)
        pred = mw.forward(features)
        loss = F.nll_loss(pred, labels)
        mw.zero_grad()
        loss.backward(create_graph=True) # important! use create_graph=True
        mw.step()
        running_loss += loss.item() * features_.size(0)
    train_loss = running_loss / len(dl_train.dataset)
    print("EPOCH: {}, TRAIN LOSS: {}".format(i, train_loss))
```
Note that on the first step of the train loop PyTorch will return the following warning:
```
UserWarning: Using backward() with create_graph=True will create a reference cycle between the parameter and its gradient which can cause a memory leak. We recommend using autograd.grad when creating the graph to avoid this. If you have to use this function, make sure to reset the .grad fields of your parameters to None after use to break the cycle and avoid the leak.
```
This is normal and to be expected.