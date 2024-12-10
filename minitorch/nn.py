from math import inf
from typing import Tuple
from xmlrpc.client import Boolean


from . import operators
from .autodiff import Context
from .fast_ops import FastOps
from .tensor import Tensor
from .tensor_functions import Function, rand, tensor


# List of functions in this file:
# - avgpool2d: Tiled average pooling 2D
# - argmax: Compute the argmax as a 1-hot tensor
# - Max: New Function for max operator
# - max: Apply max reduction
# - softmax: Compute the softmax as a tensor
# - logsoftmax: Compute the log of the softmax as a tensor - See https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations
# - maxpool2d: Tiled max pooling 2D
# - dropout: Dropout positions based on random noise, include an argument to turn off


def tile(input: Tensor, kernel: Tuple[int, int]) -> Tuple[Tensor, int, int]:
    """Reshape an image tensor for 2D pooling

    Args:
    ----
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Tensor of size batch x channel x new_height x new_width x (kernel_height * kernel_width) as well as the new_height and new_width value.

    """
    batch, channel, height, width = input.shape
    kh, kw = kernel
    assert height % kh == 0
    assert width % kw == 0
    # TODO: Implement for Task 4.3.
    # raise NotImplementedError("Need to implement for Task 4.3")
    new_height = height // kh
    new_width = width // kw
    reshaped = input.contiguous().view(batch, channel, new_height, kh, new_width, kw)
    permuted = reshaped.permute(0, 1, 2, 4, 3, 5).contiguous()
    tiled = permuted.view(batch, channel, new_height, new_width, kh * kw)
    return tiled, new_height, new_width


# TODO: Implement for Task 4.3.
def avgpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Applies average pooling over a 2D input signal composed of several input planes.

    Args:
    ----
        input: Tensor of size batch x channel x height x width
        kernel: Tuple of two integers, specifying the size of the window for each dimension of the input signal.

    Returns:
    -------
        Tensor of size batch x channel x new_height x new_width, where new_height and new_width are calculated based on the kernel size and the stride (default is 1).

    """
    kh = kernel[0]
    kw = kernel[1]
    tiled, new_height, new_width = tile(input, (kh, kw))
    result = tiled.mean(dim=4)
    result = result.view(
        input.shape[0], input.shape[1], int(new_height), int(new_width)
    )
    return result


# - max: Apply max reduction
def max(inputs: Tensor, dims: int) -> Tensor:
    """Applies max reduction over the specified dimension of the input tensor.

    Args:
    ----
        inputs: Tensor of shape batch x channel x height x width
        dims: Integer specifying the dimension over which to apply max reduction

    Returns:
    -------
        Tensor of shape batch x channel x height x width, where the specified dimension has been reduced to a single value.

    """
    return Max.apply(inputs, tensor(dims))


max_reduce = FastOps.reduce(operators.max, -inf)
add_reduce = FastOps.reduce(operators.add, 0.0)


class Max(Function):
    """Applies max reduction over the specified dimension of the input tensor.

    This function computes the maximum value along the specified dimension of the input tensor. It is used to implement the max reduction operation in the forward pass of the max function.

    Args:
    ----
        ctx (Context): The context in which the operation is performed.
        input (Tensor): The input tensor over which the max reduction is applied.
        dims (Tensor): A tensor specifying the dimension over which to apply max reduction.

    Returns:
    -------
        Tensor: The result of applying max reduction over the specified dimension of the input tensor.

    """

    @staticmethod
    def forward(ctx: Context, input: Tensor, dims: Tensor) -> Tensor:
        """Computes the maximum value along the specified dimension of the input tensor.

        This function computes the maximum value along the specified dimension of the input tensor. It is used to implement the max reduction operation in the forward pass of the max function.

        Args:
        ----
            ctx (Context): The context in which the operation is performed.
            input (Tensor): The input tensor over which the max reduction is applied.
            dims (Tensor): A tensor specifying the dimension over which to apply max reduction.

        Returns:
        -------
            Tensor: The result of applying max reduction over the specified dimension of the input tensor.

        """
        dim_i = int(dims.item())
        result = max_reduce(input, dim_i)
        ctx.save_for_backward(input, dim_i)
        return result

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """Computes the gradient of the max function.

        This function computes the gradient of the max function with respect to the input tensor. It is used to implement the backward pass of the max function.

        Args:
        ----
            ctx (Context): The context in which the operation is performed.
            grad_output (Tensor): The gradient of the output tensor.

        Returns:
        -------
            Tuple[Tensor, float]: The gradient of the input tensor and a constant value of 0.0.

        """
        input, dim = ctx.saved_values
        output = argmax(input, dim) * grad_output
        return (output, 0.0)


# argmax: Compute the argmax as a 1-hot tensor
def argmax(input: Tensor, dims: int) -> Tensor:
    """Computes the argmax as a 1-hot tensor.

    This function computes the argmax of the input tensor along the specified dimension and returns a 1-hot tensor where the maximum value is set to 1 and all other values are set to 0.

    Args:
    ----
        input (Tensor): The input tensor over which the argmax is computed.
        dims (int): The dimension along which to compute the argmax.

    Returns:
    -------
        Tensor: A 1-hot tensor where the maximum value is set to 1 and all other values are set to 0.

    """
    result = max_reduce(input, dims)
    output = input == result
    return output


# - maxpool2d: Tiled max pooling 2D
def maxpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Computes the max pooling operation over the input tensor.

    This function applies max pooling over the input tensor with the specified kernel size. It is used to downsample the input data to reduce the spatial dimensions and the number of parameters and computation in the network.

    Args:
    ----
        input (Tensor): The input tensor over which max pooling is applied.
        kernel (Tuple[int, int]): The size of the kernel to use for max pooling.

    Returns:
    -------
        Tensor: The result of applying max pooling over the input tensor.

    """
    kh = kernel[0]
    kw = kernel[1]
    tiled, new_height, new_width = tile(input, (kh, kw))
    result = max(tiled, -1)
    return result.view(input.shape[0], input.shape[1], int(new_height), int(new_width))


# - softmax: Compute the softmax as a tensor
def softmax(input: Tensor, dim: int) -> Tensor:
    """Computes the softmax of the input tensor along the specified dimension.

    This function computes the softmax of the input tensor along the specified dimension. The softmax function is often used in the output layer of a neural network to predict probabilities.

    Args:
    ----
        input (Tensor): The input tensor over which the softmax is computed.
        dim (int): The dimension along which to compute the softmax.

    Returns:
    -------
        Tensor: The softmax of the input tensor along the specified dimension.

    """
    exp_values = input.exp()
    exp_values_sum = exp_values.sum(dim)
    output = exp_values / exp_values_sum
    return output


# - logsoftmax: Compute the log of the softmax as a tensor - See https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations
def logsoftmax(input: Tensor, dim: int) -> Tensor:
    """Computes the log of the softmax of the input tensor along the specified dimension.

    This function computes the log of the softmax of the input tensor along the specified dimension. The log of the softmax function is often used in the output layer of a neural network to predict probabilities.

    Args:
    ----
        input (Tensor): The input tensor over which the log of the softmax is computed.
        dim (int): The dimension along which to compute the log of the softmax.

    Returns:
    -------
        Tensor: The log of the softmax of the input tensor along the specified dimension.

    """
    exp_values = input.exp()
    exp_values_sum = exp_values.sum(dim)
    output = exp_values / exp_values_sum
    output = output.log()
    return output


def dropout(input: Tensor, rate: float, ignore: Boolean = False) -> Tensor:
    """Applies dropout to the input tensor.

    This function randomly sets elements of the input tensor to zero with a probability equal to the dropout rate. This is a regularization technique used to prevent overfitting in neural networks.

    Args:
    ----
        input (Tensor): The input tensor to which dropout is applied.
        rate (float): The dropout rate, which is the probability of an element being set to zero.
        ignore (Boolean, optional): If True, dropout is not applied. Defaults to False.

    Returns:
    -------
        Tensor: The input tensor with dropout applied.

    """
    if ignore:
        return input
    else:
        random_input = rand(input.shape) > rate
        output = input * random_input
        return output
