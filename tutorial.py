x0 = 5.0
num_epoch = 0
learning_rate = 0.1

x = x0
for i in range(num_epoch):
    # initialize
    grad = 0

    # (optional) calculate output
    y = (x - 3) ** 2 + 4

    # calculate gradient
    grad = 2 * (x - 3)

    # update x variable
    x = x - grad * learning_rate

    # print states
    print(y, grad, x)
