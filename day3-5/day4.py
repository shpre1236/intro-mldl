import torch
import torch.nn as nn
import torch.optim as optim

x0 = 5.0
num_epoch = 10
learning_rate = 0.1

x = torch.tensor(x0, requires_grad=True)

loss_func = nn.MSELoss()
optimizer = optim.SGD([x], lr=learning_rate)

for i in range(num_epoch):
    # initialize
    optimizer.zero_grad()

    # (optional) calculate output
    y = (x-3)**2 + 4

    # calculate gradient
    y.backward()

    # update x variable
    optimizer.step()

    # print states
    print(y.item(), x.grad.item(), x.item())
