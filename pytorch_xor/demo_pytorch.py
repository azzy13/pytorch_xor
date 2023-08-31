# import libraries
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

import rclpy
from rclpy.node import Node

from std_msgs.msg import String

# create data
Xs = torch.Tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
y = torch.Tensor([0.0, 1.0, 1.0, 0.0]).reshape(Xs.shape[0], 1)


class XOR(nn.Module):
    def __init__(self):
        super(XOR, self).__init__()
        self.linear = nn.Linear(2, 2)
        self.Sigmoid = nn.Sigmoid()
        self.linear2 = nn.Linear(2, 1)

    def forward(self, input):
        x = self.linear(input)
        sig = self.Sigmoid(x)
        yh = self.linear2(sig)
        return yh


xor_network = XOR()

epochs = 1000
mseloss = nn.MSELoss()
optimizer = torch.optim.Adam(xor_network.parameters(), lr=0.03)
all_losses = []
current_loss = 0
plot_every = 50

for epoch in range(epochs):
    # input training example and return the prediction
    yhat = xor_network.forward(Xs)
    # calculate MSE loss
    loss = mseloss(yhat, y)
    # backpropogate through the loss gradiants
    loss.backward()
    # update model weights
    optimizer.step()
    # remove current gradients for next iteration
    optimizer.zero_grad()
    # append to loss
    current_loss += loss
    if epoch % plot_every == 0:
        all_losses.append(current_loss / plot_every)
        current_loss = 0
    # print progress
    if epoch % 500 == 0:
        print(f"Epoch: {epoch} completed")

# show weights and bias
for name, param in xor_network.named_parameters():
    if param.requires_grad:
        print(name, param.data)

# test input
input = torch.tensor([0.0, 1.0])
print("For inputs", input)
out = xor_network(input)
# convert to string
output = out.detach().numpy()
string = np.array2string(output, precision=0)
print(string)


def main(args=None):
    # rclpy.init(args=args)

    xor = XOR()

    # rclpy.spin(XOR)

    # XOR.destroy_node()
    # rclpy.shutdown()


if __name__ == "__main__":
    main()
