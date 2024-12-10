# type: ignore
# Currently pyright doesn't support numba.cuda

from typing import Callable, Optional, TypeVar, Any

import numba
from numba import cuda
from numba.cuda import jit as _jit
from .tensor import Tensor
from .tensor_data import (
    MAX_DIMS,
    Shape,
    Storage,
    Strides,
    TensorData,
    broadcast_index,
    index_to_position,
    shape_broadcast,
    to_index,
)
from .tensor_ops import MapProto, TensorOps

FakeCUDAKernel = Any

# This code will CUDA compile fast versions your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.

Fn = TypeVar("Fn")


def device_jit(fn: Fn, **kwargs: Any) -> Fn:
    """Compiles a function for execution on a CUDA device.

    This function takes a Python function as input and compiles it for execution on a CUDA device using Numba's CUDA support. The compiled function can then be executed on the GPU.

    Args:
    ----
        fn (Callable): The Python function to be compiled for CUDA execution.
        **kwargs: Additional keyword arguments to be passed to Numba's `cuda.jit` function.

    Returns:
    -------
        Callable: The compiled CUDA kernel function.

    """
    return _jit(device=True, **kwargs)(fn)  # type: ignore


def jit(fn: Fn, **kwargs: Any) -> FakeCUDAKernel:
    """Compiles a function for execution on a CUDA device with additional options.

    This function takes a Python function as input and compiles it for execution on a CUDA device using Numba's CUDA support. The compiled function can then be executed on the GPU. This function provides additional options for compilation compared to `device_jit`.

    Args:
    ----
        fn (Callable): The Python function to be compiled for CUDA execution.
        **kwargs: Additional keyword arguments to be passed to Numba's `cuda.jit` function.

    Returns:
    -------
        Callable: The compiled CUDA kernel function.

    """
    return _jit(**kwargs)(fn)  # type: ignore


to_index = device_jit(to_index)
index_to_position = device_jit(index_to_position)
broadcast_index = device_jit(broadcast_index)

THREADS_PER_BLOCK = 32


class CudaOps(TensorOps):
    cuda = True

    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        """Applies a given function to each element of a tensor, mapping it to a new tensor.

        This function takes a Python function as input and applies it to each element of a tensor, creating a new tensor with the results. The function is compiled for execution on a CUDA device using Numba's CUDA support, allowing for parallel execution on the GPU.

        Args:
        ----
            fn (Callable[[float], float]): The Python function to be applied to each element of the tensor. This function should take a single float argument and return a float value.

        Returns:
        -------
            MapProto: A function that can be called on a tensor to apply the given function to each element, returning a new tensor with the results.

        """
        cufn: Callable[[float], float] = device_jit(fn)
        f = tensor_map(cufn)

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)

            # Instantiate and run the cuda kernel.
            threadsperblock = THREADS_PER_BLOCK
            blockspergrid = (out.size + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
            f[blockspergrid, threadsperblock](*out.tuple(), out.size, *a.tuple())  # type: ignore
            return out

        return ret

    @staticmethod
    def zip(fn: Callable[[float, float], float]) -> Callable[[Tensor, Tensor], Tensor]:
        """Applies a given binary function to each pair of elements from two tensors, mapping them to a new tensor.

        This function takes a Python function as input and applies it to each pair of elements from two tensors, creating a new tensor with the results. The function is compiled for execution on a CUDA device using Numba's CUDA support, allowing for parallel execution on the GPU.

        Args:
        ----
            fn (Callable[[float, float], float]): The Python function to be applied to each pair of elements from the two tensors. This function should take two float arguments and return a float value.

        Returns:
        -------
            Callable[[Tensor, Tensor], Tensor]: A function that can be called on two tensors to apply the given function to each pair of elements, returning a new tensor with the results.

        """
        cufn: Callable[[float, float], float] = device_jit(fn)
        f = tensor_zip(cufn)

        def ret(a: Tensor, b: Tensor) -> Tensor:
            c_shape = shape_broadcast(a.shape, b.shape)
            out = a.zeros(c_shape)
            threadsperblock = THREADS_PER_BLOCK
            blockspergrid = (out.size + (threadsperblock - 1)) // threadsperblock
            f[blockspergrid, threadsperblock](  # type: ignore
                *out.tuple(), out.size, *a.tuple(), *b.tuple()
            )
            return out

        return ret

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[[Tensor, int], Tensor]:
        """Applies a given binary reduction function to each element along a specified dimension of a tensor, reducing it to a smaller tensor.

        This function takes a Python function as input and applies it to each element along a specified dimension of the input tensor, reducing it to a smaller tensor. The function is compiled for execution on a CUDA device using Numba's CUDA support, allowing for parallel execution on the GPU.

        Args:
        ----
            fn (Callable[[float, float], float]): The Python function to be applied to each element along the specified dimension of the tensor. This function should take two float arguments and return a float value.
            start (float, optional): The starting value for the reduction operation. Defaults to 0.0.

        Returns:
        -------
            Callable[[Tensor, int], Tensor]: A function that can be called on a tensor and a dimension to apply the given reduction function, returning a new tensor with the results.

        """
        cufn: Callable[[float, float], float] = device_jit(fn)
        f = tensor_reduce(cufn)

        def ret(a: Tensor, dim: int) -> Tensor:
            out_shape = list(a.shape)
            out_shape[dim] = (a.shape[dim] - 1) // 1024 + 1
            out_a = a.zeros(tuple(out_shape))

            threadsperblock = 1024
            blockspergrid = out_a.size
            f[blockspergrid, threadsperblock](  # type: ignore
                *out_a.tuple(), out_a.size, *a.tuple(), dim, start
            )

            return out_a

        return ret

    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
        """Performs a matrix multiplication between two tensors on a CUDA device.

        This function takes two tensors as input and performs a matrix multiplication operation on them. The operation is executed on a CUDA device, allowing for parallel execution and improved performance. The function ensures that the input tensors are broadcasted to a common shape if necessary, and the result is stored in a new tensor.

        Args:
        ----
            a (Tensor): The first tensor to be multiplied.
            b (Tensor): The second tensor to be multiplied.

        Returns:
        -------
            Tensor: A new tensor containing the result of the matrix multiplication.

        """
        # Make these always be a 3 dimensional multiply
        both_2d = 0
        if len(a.shape) == 2:
            a = a.contiguous().view(1, a.shape[0], a.shape[1])
            both_2d += 1
        if len(b.shape) == 2:
            b = b.contiguous().view(1, b.shape[0], b.shape[1])
            both_2d += 1
        both_2d = both_2d == 2

        ls = list(shape_broadcast(a.shape[:-2], b.shape[:-2]))
        ls.append(a.shape[-2])
        ls.append(b.shape[-1])
        assert a.shape[-1] == b.shape[-2]
        out = a.zeros(tuple(ls))

        # One block per batch, extra rows, extra col
        blockspergrid = (
            (out.shape[1] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
            (out.shape[2] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
            out.shape[0],
        )
        threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1)

        tensor_matrix_multiply[blockspergrid, threadsperblock](
            *out.tuple(), out.size, *a.tuple(), *b.tuple()
        )

        # Undo 3d if we added it.
        if both_2d:
            out = out.view(out.shape[1], out.shape[2])
        return out


# Implement


def tensor_map(
    fn: Callable[[float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides], None]:
    """CUDA higher-order tensor map function. ::

      fn_map = tensor_map(fn)
      fn_map(out, ... )

    Args:
    ----
        fn: function mappings floats-to-floats to apply.

    Returns:
    -------
        Tensor map function.

    """

    def _map(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        in_storage: Storage,
        in_shape: Shape,
        in_strides: Strides,
    ) -> None:
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        in_index = cuda.local.array(MAX_DIMS, numba.int32)
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        # TODO: Implement for Task 3.3.
        # raise NotImplementedError("Need to implement for Task 3.3")
        if i < out_size:
            to_index(i, out_shape, out_index)
            broadcast_index(out_index, out_shape, in_shape, in_index)
            o = index_to_position(out_index, out_strides)
            j = index_to_position(in_index, in_strides)
            out[o] = fn(in_storage[j])

    return cuda.jit()(_map)  # type: ignore


def tensor_zip(
    fn: Callable[[float, float], float],
) -> Callable[
    [Storage, Shape, Strides, Storage, Shape, Strides, Storage, Shape, Strides], None
]:
    """CUDA higher-order tensor zipWith (or map2) function ::

      fn_zip = tensor_zip(fn)
      fn_zip(out, ...)

    Args:
    ----
        fn: function mappings two floats to float to apply.

    Returns:
    -------
        Tensor zip function.

    """

    def _zip(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        b_storage: Storage,
        b_shape: Shape,
        b_strides: Strides,
    ) -> None:
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        a_index = cuda.local.array(MAX_DIMS, numba.int32)
        b_index = cuda.local.array(MAX_DIMS, numba.int32)
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

        # TODO: Implement for Task 3.3.
        # raise NotImplementedError("Need to implement for Task 3.3")
        if i < out_size:
            to_index(i, out_shape, out_index)
            o = index_to_position(out_index, out_strides)
            broadcast_index(out_index, out_shape, a_shape, a_index)
            j = index_to_position(a_index, a_strides)
            broadcast_index(out_index, out_shape, b_shape, b_index)
            k = index_to_position(b_index, b_strides)
            out[o] = fn(a_storage[j], b_storage[k])

    return cuda.jit()(_zip)  # type: ignore


def _sum_practice(out: Storage, a: Storage, size: int) -> None:
    r"""Practice sum kernel to prepare for reduce.

    Given an array of length $n$ and out of size $n // \text{blockDIM}$
    it should sum up each blockDim values into an out cell.

    $[a_1, a_2, ..., a_{100}]$

    |

    $[a_1 +...+ a_{31}, a_{32} + ... + a_{64}, ... ,]$

    Note: Each block must do the sum using shared memory!

    Args:
    ----
        out (Storage): storage for `out` tensor.
        a (Storage): storage for `a` tensor.
        size (int):  length of a.

    """
    BLOCK_DIM = 32

    cache = cuda.shared.array(BLOCK_DIM, numba.float64)
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    pos = cuda.threadIdx.x

    # TODO: Implement for Task 3.3.
    # raise NotImplementedError("Need to implement for Task 3.3")

    if i < size:
        val = float(a[i])
        cache[pos] = val
        cuda.syncthreads()
    else:
        cache[pos] = 0

    if i < size:
        for j in [1, 2, 3, 8, 16]:
            if pos % (j * 2) == 0:
                cache[pos] += cache[pos + j]
                cuda.syncthreads()
        if pos == 0:
            out[cuda.blockIdx.x] = cache[0]


jit_sum_practice = cuda.jit()(_sum_practice)


def sum_practice(a: Tensor) -> TensorData:
    r"""Performs a sum operation on a tensor using CUDA.

    Args:
    ----
        a (Tensor): The input tensor to be summed.

    Returns:
    -------
        TensorData: The result of the sum operation.

    """
    (size,) = a.shape
    threadsperblock = THREADS_PER_BLOCK
    blockspergrid = (size // THREADS_PER_BLOCK) + 1
    out = TensorData([0.0 for i in range(2)], (2,))
    out.to_cuda_()
    jit_sum_practice[blockspergrid, threadsperblock](
        out.tuple()[0], a._tensor._storage, size
    )
    return out


def tensor_reduce(
    fn: Callable[[float, float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides, int], None]:
    """CUDA higher-order tensor reduce function.

    Args:
    ----
        fn: reduction function maps two floats to float.

    Returns:
    -------
        Tensor reduce function.

    """

    def _reduce(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        reduce_dim: int,
        reduce_value: float,
    ) -> None:
        # *out_a.tuple(), out_a.size, *a.tuple(), dim, start
        BLOCK_DIM = 1024
        cache = cuda.shared.array(BLOCK_DIM, numba.float64)
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        out_pos = cuda.blockIdx.x
        pos = cuda.threadIdx.x
        cache[pos] = reduce_value

        # TODO: Implement for Task 3.3.
        # raise NotImplementedError("Need to implement for Task 3.3")
        if out_pos < out_size:
            to_index(out_pos, out_shape, out_index)
            o = index_to_position(out_index, out_strides)

            out_index[reduce_dim] = out_index[reduce_dim] * BLOCK_DIM + pos
            if out_index[reduce_dim] < a_shape[reduce_dim]:
                in_a = index_to_position(out_index, a_strides)
                cache[pos] = a_storage[in_a]
                cuda.syncthreads()
                x = 0
                while 2**x < BLOCK_DIM:
                    j = 2**x
                    if pos % (j * 2) == 0:
                        cache[pos] = fn(cache[pos], cache[pos + j])
                        cuda.syncthreads()
                    x += 1
                if pos == 0:
                    out[o] = cache[0]

    return jit(_reduce)  # type: ignore


def _mm_practice(out: Storage, a: Storage, b: Storage, size: int) -> None:
    """Practice square MM kernel to prepare for matmul.

    Given a storage `out` and two storage `a` and `b`. Where we know
    both are shape [size, size] with strides [size, 1].

    Size is always < 32.

    Requirements:

    * All data must be first moved to shared memory.
    * Only read each cell in `a` and `b` once.
    * Only write to global memory once per kernel.

    Compute

    ```
     for i:
         for j:
              for k:
                  out[i, j] += a[i, k] * b[k, j]
    ```

    Args:
    ----
        out (Storage): storage for `out` tensor.
        a (Storage): storage for `a` tensor.
        b (Storage): storage for `b` tensor.
        size (int): size of the square

    """
    BLOCK_DIM = 32

    # TODO: Implement for Task 3.3.
    # raise NotImplementedError("Need to implement for Task 3.3")
    """
    Each thread reads a value from `a` at its position and writes it to `sharedA` at the corresponding position.
    Similarly, each thread reads a value from `b` at its position and writes it to `sharedB` at the corresponding position.
    """
    sharedA = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)
    sharedB = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)
    local_i = cuda.threadIdx.x
    local_j = cuda.threadIdx.y
    position = local_i * size + local_j

    if local_i < size and local_j < size:
        sharedA[local_i, local_j] = a[position]
        sharedB[local_i, local_j] = b[position]
    else:
        sharedA[local_i, local_j] = 0
        sharedB[local_i, local_j] = 0
    """
    After writing, all threads synchronize to ensure the shared arrays are up-to-date for all threads.
    Synchronization is crucial because threads run simultaneously, and without it, some values in `sharedA`
    and/or `sharedB` might be missing or incomplete.
    """
    cuda.syncthreads()
    """
    Using the synchronized `sharedA` and `sharedB`, which now contain all relevant values from the global `a` and `b`,
    each thread calculates its part of the matrix multiplication.

    Specifically, each thread (t) accumulates the product of row `x` of `a` (using `sharedA`) and column `y` of `b`
    (using `sharedB`).
    """
    t = 0
    for k in range(size):
        t += sharedA[local_i, k] * sharedB[k, local_j]
    """
    Once the accumulation is complete, the thread writes its result to the global `out` array
    at its corresponding position.
    """
    out[position] = t


jit_mm_practice = jit(_mm_practice)


def mm_practice(a: Tensor, b: Tensor) -> TensorData:
    """Performs a matrix multiplication between two tensors on a CUDA device.

    This function takes two tensors as input and performs a matrix multiplication operation on them. The operation is executed on a CUDA device, allowing for parallel execution and improved performance. The function ensures that the input tensors are broadcasted to a common shape if necessary, and the result is stored in a new tensor.

    Args:
    ----
        a (Tensor): The first tensor to be multiplied.
        b (Tensor): The second tensor to be multiplied.

    Returns:
    -------
        TensorData: A new tensor containing the result of the matrix multiplication.

    """
    (size, _) = a.shape
    threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK)
    blockspergrid = 1
    out = TensorData([0.0 for i in range(size * size)], (size, size))
    out.to_cuda_()
    jit_mm_practice[blockspergrid, threadsperblock](
        out.tuple()[0], a._tensor._storage, b._tensor._storage, size
    )
    return out


def _tensor_matrix_multiply(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    out_size: int,
    a_storage: Storage,
    a_shape: Shape,
    a_strides: Strides,
    b_storage: Storage,
    b_shape: Shape,
    b_strides: Strides,
) -> None:
    """CUDA tensor matrix multiply function.

    Requirements:

    * All data must be first moved to shared memory.
    * Only read each cell in `a` and `b` once.
    * Only write to global memory once per kernel.

    Should work for any tensor shapes that broadcast as long as ::

    ```python
    assert a_shape[-1] == b_shape[-2]
    ```
    Returns:
        None : Fills in `out`
    """
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0
    # Batch dimension - fixed
    batch = cuda.blockIdx.z

    BLOCK_DIM = 32
    a_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)
    b_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)

    # The final position c[i, j]
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    j = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

    # The local position in the block.
    pi = cuda.threadIdx.x
    pj = cuda.threadIdx.y

    # Code Plan:
    # 1) Move across shared dimension by block dim.
    #    a) Copy into shared memory for a matrix.
    #    b) Copy into shared memory for b matrix
    #    c) Compute the dot produce for position c[i, j]
    # TODO: Implement for Task 3.4.
    # raise NotImplementedError("Need to implement for Task 3.4")
    """
    When the last dimension of `a_shape` (a_shape[-1]) is larger than the block dimension (BLOCK_DIM),
    each thread computes partial results of the matrix multiplication within the block and accumulates
    them into the final output.

    `t` is a local variable used to store the accumulated result of the matrix multiplication for the thread's position.
    """
    t = 0
    size = a_shape[-1]
    for s in range(0, size, BLOCK_DIM):
        """
        Each thread reads the relative position of `a` and `b` corresponding to its coordinates
        within the block and writes the values to the shared memory (`a_shared` and `b_shared`).

        For example, for a thread at position (0, 0) in the second block, it reads `a(0:32, 32:64)`
        (as the second block corresponds to that section of `a`) and stores the value at (0, 0) in `a_shared`.
        """
        if i < a_shape[1] and (pj + s) < a_shape[2]:
            a_position = (
                batch * a_batch_stride + i * a_strides[-2] + (pj + s) * a_strides[-1]
            )
            a_shared[pi, pj] = a_storage[a_position]

        if (pi + s) < b_shape[1] and j < b_shape[2]:
            b_position = (
                batch * b_batch_stride + (pi + s) * b_strides[-2] + j * b_strides[-1]
            )
            b_shared[pi, pj] = b_storage[b_position]

        "Synchronize shared memory to ensure all threads have updated `a_shared` and `b_shared`."
        cuda.syncthreads()
        "Compute the partial matrix multiplication for the current block and accumulate it into `t`."
        for k in range(min(BLOCK_DIM, size - s)):
            t += a_shared[pi, k] * b_shared[k, pj]

    "After computing the accumulated result, write `t` to the global `out` storage at the thread's position."
    if i < out_shape[1] and j < out_shape[2]:
        out_position = batch * out_strides[0] + i * out_strides[1] + j * out_strides[2]
        out[out_position] = t


tensor_matrix_multiply = jit(_tensor_matrix_multiply)
