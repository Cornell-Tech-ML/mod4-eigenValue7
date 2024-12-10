"""
Be sure you have minitorch installed in you Virtual Env.
>>> pip install -Ue .
"""

from numpy import matmul
import minitorch

def RParam(*shape):
    r = 2 * (minitorch.rand(shape) - 0.5)
    return minitorch.Parameter(r)

class Network(minitorch.Module):
    """
    This is a simple neural network class that inherits from the minitorch Module.
    It initializes the network with a specified number of hidden layers and sets up the network architecture.
    The network consists of a Linear layer from the input to the hidden layer, another Linear layer from the hidden layer to the output layer,
    and an optional third Linear layer if needed.
    The layers are then added to the network.
    """

    def __init__(self, hidden_layers):
        """
        This is the constructor of the Network class.
        It initializes the network with a specified number of hidden layers and sets up the network architecture.
        The network consists of a Linear layer from the input to the hidden layer, another Linear layer from the hidden layer to the output layer,
        and an optional third Linear layer if needed.
        The layers are then added to the network.
        """
        super().__init__()
        self.layer1 = Linear(2, hidden_layers)
        self.layer2 = Linear(hidden_layers,hidden_layers)
        self.layer3 = Linear(hidden_layers, 1)

    def forward(self, x):
        """
        This function performs the forward pass of the neural network.
        It first passes the input through the first linear layer and applies the ReLU activation function.
        Then it passes the output of the first layer through the second linear layer and applies the ReLU activation function.
        Finally, it passes the output of the second layer through the third linear layer and applies the sigmoid activation function.
        The output of the third layer is the final output of the network.
        """
        middle = self.layer1.forward(x).relu()
        end = self.layer2.forward(middle).relu()
        return self.layer3.forward(end).sigmoid()

class Linear(minitorch.Module):
    """
    A Linear layer for neural networks.

    Attributes:
        weights (Tensor): The weights of the layer.
        bias (Tensor): The bias terms for the layer.

    Methods:
        forward(x): Computes the output of the layer for a given input x.
    """
    def __init__(self, in_size, out_size):
        """
        Initializes a Linear layer with specified input and output sizes.

        Args:
            in_size (int): The number of neurons in the input layer.
            out_size (int): The number of neurons in the output layer.

        This method sets up the linear layer by initializing the weights and bias terms. The weights are initialized using the Xavier initialization method, and the bias terms are initialized randomly. The layer is then added to the network.
        """
        super().__init__()
        self.weights = RParam(in_size, out_size)
        self.bias = RParam(out_size)
        self.out_size = out_size

    def forward(self, x):
        """
        Computes the output of the Linear layer for a given input x.

        Args:
            x (Tensor): The input tensor to the layer.

        Returns:
            Tensor: The output tensor of the layer.
        """
        batch, in_size = x.shape
        return (
            self.weights.value.view(1, in_size, self.out_size)
            * x.view(batch, in_size, 1)
        ).sum(1).view(batch, self.out_size) + self.bias.value.view(self.out_size)

def default_log_fn(epoch, total_loss, correct, losses):
    print("Epoch ", epoch, " loss ", total_loss, "correct", correct)


class TensorTrain:
    def __init__(self, hidden_layers):
        self.hidden_layers = hidden_layers
        self.model = Network(hidden_layers)

    def run_one(self, x):
        return self.model.forward(minitorch.tensor([x]))

    def run_many(self, X):
        return self.model.forward(minitorch.tensor(X))

    def train(self, data, learning_rate, max_epochs=500, log_fn=default_log_fn):
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.model = Network(self.hidden_layers)
        optim = minitorch.SGD(self.model.parameters(), learning_rate)

        X = minitorch.tensor(data.X)
        y = minitorch.tensor(data.y)

        losses = []
        for epoch in range(1, self.max_epochs + 1):
            total_loss = 0.0
            correct = 0
            optim.zero_grad()

            # Forward
            out = self.model.forward(X).view(data.N)
            prob = (out * y) + (out - 1.0) * (y - 1.0)

            loss = -prob.log()
            (loss / data.N).sum().view(1).backward()
            total_loss = loss.sum().view(1)[0]
            losses.append(total_loss)

            # Update
            optim.step()

            # Logging
            if epoch % 10 == 0 or epoch == max_epochs:
                y2 = minitorch.tensor(data.y)
                correct = int(((out.detach() > 0.5) == y2).sum()[0])
                log_fn(epoch, total_loss, correct, losses)


if __name__ == "__main__":
    PTS = 50
    HIDDEN = 2
    RATE = 0.5
    data = minitorch.datasets["Simple"](PTS)
    TensorTrain(HIDDEN).train(data, RATE)
