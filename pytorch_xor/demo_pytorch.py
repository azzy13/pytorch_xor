import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

import rclpy
from rclpy.node import Node

from std_msgs.msg import String


class Publisher(Node):
    def __init__(self):
        super().__init__("node")
        self.publisher_ = self.create_publisher(String, "topic", 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        msg = String()
        msg.data = self.string_data
        # Pass the string data from train_xor_network to timer_callback
        self.publisher_.publish(msg)
        self.get_logger().info('Publishing: "%s"' % msg.data)
        self.i += 1


def train_xor_network():
    Xs = torch.Tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    y = torch.Tensor([0.0, 1.0, 1.0, 0.0]).reshape(Xs.shape[0], 1)

    # Define the XOR neural network
    xor_network = nn.Sequential(nn.Linear(2, 2), nn.Sigmoid(), nn.Linear(2, 1))

    epochs = 1000
    mseloss = nn.MSELoss()
    optimizer = torch.optim.Adam(xor_network.parameters(), lr=0.03)
    all_losses = []
    current_loss = 0
    plot_every = 50

    for epoch in range(epochs):
        # Input training example and return the prediction
        yhat = xor_network(Xs)
        # Calculate MSE loss
        loss = mseloss(yhat, y)
        # Backpropagate through the loss gradients
        loss.backward()
        # Update model weights
        optimizer.step()
        # Remove current gradients for the next iteration
        optimizer.zero_grad()
        # Append to the loss
        current_loss += loss
        if epoch % plot_every == 0:
            all_losses.append(current_loss / plot_every)
            current_loss = 0
        # Print progress
        if epoch % 500 == 0:
            print(f"Epoch: {epoch} completed")

    # Show weights and bias
    for name, param in xor_network.named_parameters():
        if param.requires_grad:
            print(name, param.data)

    # Test input
    all_out = ""
    for p in [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]:
        inp = torch.tensor(p)
        print("For inputs", inp)
        out = xor_network(inp)
        # Convert to string
        out1 = out.detach().numpy()
        str = np.array2string(out1, precision=0)
        all_out += str
        print(str)

    # Return the string data to be passed to timer_callback
    return all_out


def main(args=None):
    rclpy.init(args=args)
    node = Publisher()

    # Call the function to train the XOR network and get the string data
    string_data = train_xor_network()

    # Pass the string data to the timer_callback
    node.string_data = string_data

    rclpy.spin(node)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
