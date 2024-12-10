"""
Be sure you have minitorch installed in you Virtual Env.
>>> pip install -Ue .
"""

import random
import numpy as np

from turtle import hideturtle

import minitorch


class Network(minitorch.Module):
    """
    A simple neural network with two hidden layers for binary classification.

    This network consists of three layers: an input layer, a hidden layer, and an output layer.
    The input layer takes in two features, the hidden layer has a variable number of neurons,
    and the output layer has one neuron for binary classification. The network uses the ReLU
    activation function for the hidden layer and the sigmoid function for the output layer.

    Attributes:
        layer1 (Linear): The first layer of the network, mapping input to hidden layer.
        layer2 (Linear): The second layer of the network, mapping hidden layer to output layer.
        layer3 (Linear): The third layer of the network, mapping hidden layer to output layer (optional).

    Methods:
        forward(x): Computes the output of the network for a given input x.
    """
    def __init__(self, hidden_layers):
        """
        Initializes the Network with a specified number of hidden layers.

        Args:
            hidden_layers (int): The number of neurons in the hidden layer.

        This method sets up the network architecture by initializing the layers with the specified number of hidden layers.
        It creates a Linear layer from the input to the hidden layer, another Linear layer from the hidden layer to the output layer,
        and an optional third Linear layer if needed. The layers are then added to the network.
        """
        super().__init__()
        self.layer1 = Linear(2, hidden_layers)  # Input layer to first hidden layer
        # self.layer2 = Linear(hidden_layers, 1)  # First hidden layer to output layer
        self.layer2 = Linear(hidden_layers, hidden_layers)  # First hidden layer to output layer

        self.layer3 = Linear(2, 1)  # Optional: another layer if needed

    def forward(self, x):
        """
        Computes the output of the network for a given input x.

        Args:
            x (list of Scalars): The input to the network.

        Returns:
            Scalar: The output of the network for the given input x.

        This method propagates the input through the network, applying the ReLU activation function to the hidden layers and the sigmoid function to the output layer. It returns the final output of the network.
        """
        middle = [h.relu() for h in self.layer1.forward(x)]
        end = [h.relu() for h in self.layer2.forward(middle)]
        return self.layer3.forward(end)[0].sigmoid()
        # layer1 = self.layer1.forward(x)[0].relu()
        # layer2 = self.layer2.forward(x)[0].relu()
        # return self.layer3.forward((layer1, layer2))[0].sigmoid()




class Linear(minitorch.Module):
    """
    A Linear layer for neural networks.

    Attributes:
        weights (list of lists of Scalars): The weights of the layer, organized as a list of lists where each sublist corresponds to a neuron in the output layer.
        bias (list of Scalars): The bias terms for each neuron in the output layer.

    Methods:
        forward(inputs): Computes the output of the layer for a given input.
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
        self.weights = []
        self.bias = []
        #For Xor, it's better to used xavier weights
        xavier_weights = Linear.get_xavier_weights(in_size, out_size)

        for i in range(in_size):
            self.weights.append([])
            for j in range(out_size):
                self.weights[i].append(
                    self.add_parameter(
                        # f"weight_{i}_{j}", minitorch.Scalar(2 * (random.random() - 0.5))

                        #For Xor, it's better to used xavier weights
                        f"weight_{i}_{j}", minitorch.Scalar(xavier_weights[i * out_size + j])
                    )
                )
        for j in range(out_size):
            self.bias.append(
                self.add_parameter(
                    f"bias_{j}", minitorch.Scalar(2 * (random.random() - 0.5))
                )
            )

    def forward(self, inputs):
        """
        Computes the output of the Linear layer for a given input.

        Args:
            inputs (list of Scalars): The input values to the layer.

        Returns:
            list of Scalars: The output values of the layer.
        """
        outputs = []
        for j in range(len(self.bias)):
            # Compute the dot product of inputs and weights, plus bias
            weighted_sum = self.bias[j].value
            for i in range(len(inputs)):
                weighted_sum += inputs[i] * self.weights[i][j].value
            outputs.append(weighted_sum)
        return outputs

    @staticmethod
    def get_xavier_weights(fan_in: int, fan_out: int):
        """
        Generates weights for a linear layer using the Xavier initialization method.

        Args:
            fan_in (int): The number of neurons in the input layer.
            fan_out (int): The number of neurons in the output layer.

        Returns:
            list of floats: The weights initialized using the Xavier method.
        """
        n = fan_in * fan_out
        random_weights = np.random.uniform(low=-1.0, high=1.0, size=n)

        # Adjust the mean to be exactly 0
        actual_mean = np.mean(random_weights)
        xavier_weights = random_weights - actual_mean

        # Calculate desired variance
        desired_variance = 2/ (fan_in + fan_out)

        # Adjust the variance to be the desired variance
        actual_variance = np.var(xavier_weights)
        scaling_factor = np.sqrt(desired_variance / actual_variance)
        xavier_weights = xavier_weights * scaling_factor

        return xavier_weights
    # Ensure inputs are in the correct format

def default_log_fn(epoch, total_loss, correct, losses):
    """
    Logs the training progress at each epoch.

    Args:
        epoch (int): The current epoch number.
        total_loss (float): The total loss accumulated over the epoch.
        correct (int): The number of correct predictions made during the epoch.
        losses (list of floats): A list of total losses at each epoch.
    """
    print("- Epoch ", epoch, " loss ", total_loss, "correct", correct)


class ScalarTrain:
    """
    A class for training scalar models.

    This class is designed to facilitate the training process of scalar models. It provides methods for running the model on a single input, training the model on a dataset, and logging the training progress.

    Attributes:
        hidden_layers (list): A list of integers representing the number of neurons in each hidden layer of the model.
        model (Network): The neural network model to be trained.
        learning_rate (float): The learning rate used for training the model.
        max_epochs (int): The maximum number of epochs to train the model.
    """
    def __init__(self, hidden_layers):
        """
        Initializes a ScalarTrain object with the specified hidden layers.

        Args:
            hidden_layers (list): A list of integers representing the number of neurons in each hidden layer of the model.
        """
        self.hidden_layers = hidden_layers
        self.model = Network(self.hidden_layers)

    def run_one(self, x):
        """
        Runs the model on a single input.

        Args:
            x (list): A list of two values representing the input to the model.

        Returns:
            The output of the model after running on the input.
        """
        return self.model.forward(
            (minitorch.Scalar(x[0], name="x_1"), minitorch.Scalar(x[1], name="x_2"))
        )

    def train(self, data, learning_rate, max_epochs=500, log_fn=default_log_fn):
        """
        Trains the scalar model on the provided dataset.

        This method trains the scalar model on the given dataset for a specified number of epochs with a given learning rate. It logs the training progress at each epoch using a provided logging function.

        Args:
            data (Dataset): The dataset to train the model on.
            learning_rate (float): The learning rate to use for training.
            max_epochs (int, optional): The maximum number of epochs to train the model. Defaults to 500.
            log_fn (function, optional): A function to log the training progress. Defaults to default_log_fn.
        """
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.model = Network(self.hidden_layers)
        optim = minitorch.SGD(self.model.parameters(), learning_rate)

        losses = []
        for epoch in range(1, self.max_epochs + 1):
            total_loss = 0.0
            correct = 0
            optim.zero_grad()

            # Forward
            loss = 0
            for i in range(data.N):
                x_1, x_2 = data.X[i]
                y = data.y[i]
                x_1 = minitorch.Scalar(x_1)
                x_2 = minitorch.Scalar(x_2)
                out = self.model.forward((x_1, x_2))

                if y == 1:
                    prob = out
                    correct += 1 if out.data > 0.5 else 0
                else:
                    prob = -out + 1.0
                    correct += 1 if out.data < 0.5 else 0
                loss = -prob.log()
                (loss / data.N).backward()
                total_loss += loss.data

            losses.append(total_loss)

            # Update
            optim.step()

            # Logging
            if epoch % 10 == 0 or epoch == max_epochs:
                log_fn(epoch, total_loss, correct, losses)


if __name__ == "__main__":
    PTS = 50
    HIDDEN = 5
    RATE = 0.1
    data = minitorch.datasets["Xor"](PTS)
    ScalarTrain(HIDDEN).train(data, RATE)
