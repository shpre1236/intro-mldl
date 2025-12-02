import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init

num_data = 10000
num_epoch = 1000
learning_rate = 0.01

x = init.uniform_(torch.Tensor(num_data,1),0,6)
y = (x - 3)**2 + 4
noise = init.normal_(torch.FloatTensor(num_data,1), std=0.5)
y_noise = y + noise

model = nn.Sequential(
    nn.Linear(1,10),
    nn.ReLU(),
    nn.Linear(10,30),
    nn.ReLU(),
    nn.Linear(30,10),
    nn.ReLU(),
    nn.Linear(10,1)
)

loss_func = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

loss_array = []
for i in range(num_epoch):
    #initialize
    optimizer.zero_grad()

    # calculate output and loss
    output = model(x)
    loss = loss_func(output, y_noise)

    # calculate gradient
    loss.backward()

    # update
    optimizer.step()

    # (optional) save loss value
    loss_array.append(loss.item())

print(model(torch.FloatTensor([3.0])).item())
print(model(torch.FloatTensor([4.0])).item())
print(model(torch.FloatTensor([2.0])).item())

import numpy as np
import matplotlib.pyplot as plt

t = np.arange(0, 6, 0.01)
y_pred = np.array([])
y_real = np.array([])
for i in range(t.size):
    y_pred = np.append(y_pred, model(torch.FloatTensor([t[i]])).item())
    y_real = np.append(y_real, (t[i] - 3) ** 2 + 4)

plt.figure()
plt.plot(x, y_noise, '.k')
plt.plot(t, y_real, 'b-')
plt.plot(t, y_pred, 'r-')
