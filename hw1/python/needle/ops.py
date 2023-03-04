"""Operator implementations."""

from numbers import Number
from typing import Optional, List
from .autograd import NDArray
from .autograd import Op, Tensor, Value, TensorOp
from .autograd import TensorTuple, TensorTupleOp
import numpy

# NOTE: we will import numpy as the array_api
# as the backend for our computations, this line will change in later homeworks
import numpy as array_api


class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        return a ** self.scalar

    def gradient(self, out_grad, node):
        a = node.inputs[0]
        return (out_grad * self.scalar * (a ** (self.scalar - 1)),)


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        return a / b

    def gradient(self, out_grad, node):
        lhs, rhs = node.inputs
        return (out_grad / rhs, -out_grad * lhs / rhs**2)


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        return a / self.scalar

    def gradient(self, out_grad, node):
        return (out_grad / self.scalar,)


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ndim = len(a.shape)
        axes = array_api.arange(ndim)
        if self.axes is not None:
            axes[self.axes[0]], axes[self.axes[1]] = axes[self.axes[1]], axes[self.axes[0]]
        elif ndim >= 2:
            axes[-2], axes[-1] = axes[-1], axes[-2]
        return array_api.transpose(a, axes=axes)

    def gradient(self, out_grad, node):
        return out_grad.transpose(axes=self.axes)


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return array_api.reshape(a, self.shape)

    def gradient(self, out_grad, node):
        input_shape = node.inputs[0].shape
        return out_grad.reshape(input_shape)


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return array_api.broadcast_to(a, self.shape)

    def gradient(self, out_grad, node):
        input_shape = node.inputs[0].shape
        output_dim = len(self.shape)
        sum_axes = [i for i in range(output_dim - len(input_shape))]
        for i in range(len(input_shape)):
            if input_shape[i] == 1:
                sum_axes.append(i + output_dim - len(input_shape))
        return out_grad.sum(axes=tuple(sum_axes)).reshape(input_shape)


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        return array_api.sum(a, axis=self.axes)

    def gradient(self, out_grad, node):
        broadcast_shape = list(node.inputs[0].shape)
        flatten_axes = self.axes
        reshape_shape = [
            1 if flatten_axes is None or i in flatten_axes else broadcast_shape[i]
            for i in range(len(broadcast_shape))
        ]
        return out_grad.reshape(reshape_shape).broadcast_to(broadcast_shape)


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        return array_api.matmul(a, b)

    def gradient(self, out_grad, node):
        lhs, rhs = node.inputs
        # out_grad - (a, c), lhs - (a, b), rhs - (b, c)
        left_grad, right_grad = out_grad.matmul(rhs.transpose()), lhs.transpose().matmul(out_grad)
        lhs_broadcast = tuple(i for i in range(0, len(left_grad.shape) - len(lhs.shape)))
        rhs_broadcast = tuple(i for i in range(0, len(right_grad.shape) - len(rhs.shape)))
        if lhs_broadcast:
            left_grad = left_grad.sum(axes=lhs_broadcast)
        if rhs_broadcast:
            right_grad = right_grad.sum(axes=rhs_broadcast)
        return (left_grad, right_grad)


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        return -a

    def gradient(self, out_grad, node):
        return -out_grad


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        return array_api.log(a)

    def gradient(self, out_grad, node):
        return out_grad / node.inputs[0]


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        return array_api.exp(a)

    def gradient(self, out_grad, node):
        a = node.inputs[0]
        return out_grad * node


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        return array_api.maximum(a, 0)

    def gradient(self, out_grad, node):
        relu_grad = (node.realize_cached_data() > 0).astype(array_api.float32)
        return Tensor.make_const(relu_grad) * out_grad


def relu(a):
    return ReLU()(a)

