import einx._src.adapter as adapter
import numpy as np
from einx._src.util.functools import use_name_of
from .._util import _associative_binary_to_nary, _to_tensor


class ops:
    def __init__(self, jax):
        jnp = jax.numpy

        def to_tensor_all(*args):
            to_tensor_one = _to_tensor(jnp.asarray, forward=[jnp.ndarray, "numpy", "scalar"], convert=[])
            return [to_tensor_one(arg) for arg in args]

        self.reshape = adapter.classical_from_numpy.reshape(jnp.reshape, to_tensor=to_tensor_all)
        self.transpose = adapter.classical_from_numpy.transpose(jnp.transpose, to_tensor=to_tensor_all)
        self.broadcast_to = adapter.classical_from_numpy.broadcast_to(jnp.broadcast_to, to_tensor=to_tensor_all)
        self.diagonal = adapter.classical_from_numpy.diagonal(jnp.diagonal, self.transpose, to_tensor=to_tensor_all)
        self.stop_gradient = jax.lax.stop_gradient

        self.add = adapter.classical_from_numpy.elementwise(_associative_binary_to_nary(jnp.add), to_tensor=to_tensor_all)
        self.subtract = adapter.classical_from_numpy.elementwise(jnp.subtract, to_tensor=to_tensor_all)
        self.multiply = adapter.classical_from_numpy.elementwise(_associative_binary_to_nary(jnp.multiply), to_tensor=to_tensor_all)
        self.true_divide = adapter.classical_from_numpy.elementwise(jnp.true_divide, to_tensor=to_tensor_all)
        self.floor_divide = adapter.classical_from_numpy.elementwise(jnp.floor_divide, to_tensor=to_tensor_all)
        self.divide = adapter.classical_from_numpy.elementwise(jnp.divide, to_tensor=to_tensor_all)
        self.logaddexp = adapter.classical_from_numpy.elementwise(_associative_binary_to_nary(jnp.logaddexp), to_tensor=to_tensor_all)
        self.logical_and = adapter.classical_from_numpy.elementwise(_associative_binary_to_nary(jnp.logical_and), to_tensor=to_tensor_all)
        self.logical_or = adapter.classical_from_numpy.elementwise(_associative_binary_to_nary(jnp.logical_or), to_tensor=to_tensor_all)
        self.where = adapter.classical_from_numpy.elementwise(jnp.where, to_tensor=to_tensor_all)
        self.maximum = adapter.classical_from_numpy.elementwise(_associative_binary_to_nary(jnp.maximum), to_tensor=to_tensor_all)
        self.minimum = adapter.classical_from_numpy.elementwise(_associative_binary_to_nary(jnp.minimum), to_tensor=to_tensor_all)
        self.less = adapter.classical_from_numpy.elementwise(jnp.less, to_tensor=to_tensor_all)
        self.less_equal = adapter.classical_from_numpy.elementwise(jnp.less_equal, to_tensor=to_tensor_all)
        self.greater = adapter.classical_from_numpy.elementwise(jnp.greater, to_tensor=to_tensor_all)
        self.greater_equal = adapter.classical_from_numpy.elementwise(jnp.greater_equal, to_tensor=to_tensor_all)
        self.equal = adapter.classical_from_numpy.elementwise(jnp.equal, to_tensor=to_tensor_all)
        self.not_equal = adapter.classical_from_numpy.elementwise(jnp.not_equal, to_tensor=to_tensor_all)
        self.exp = adapter.classical_from_numpy.elementwise(jnp.exp, to_tensor=to_tensor_all)
        self.log = adapter.classical_from_numpy.elementwise(jnp.log, to_tensor=to_tensor_all)
        self.negative = adapter.classical_from_numpy.elementwise(jnp.negative, to_tensor=to_tensor_all)
        self.divmod = adapter.classical_from_numpy.elementwise(jnp.divmod, to_tensor=to_tensor_all)

        self.sum = adapter.classical_from_numpy.reduce(jnp.sum, to_tensor=to_tensor_all)
        self.mean = adapter.classical_from_numpy.reduce(jnp.mean, to_tensor=to_tensor_all)
        self.var = adapter.classical_from_numpy.reduce(jnp.var, to_tensor=to_tensor_all)
        self.std = adapter.classical_from_numpy.reduce(jnp.std, to_tensor=to_tensor_all)
        self.prod = adapter.classical_from_numpy.reduce(jnp.prod, to_tensor=to_tensor_all)
        self.count_nonzero = adapter.classical_from_numpy.reduce(jnp.count_nonzero, to_tensor=to_tensor_all)
        self.any = adapter.classical_from_numpy.reduce(jnp.any, to_tensor=to_tensor_all)
        self.all = adapter.classical_from_numpy.reduce(jnp.all, to_tensor=to_tensor_all)
        self.max = adapter.classical_from_numpy.reduce(jnp.max, to_tensor=to_tensor_all)
        self.min = adapter.classical_from_numpy.reduce(jnp.min, to_tensor=to_tensor_all)
        self.logsumexp = adapter.classical_from_numpy.reduce(jax.nn.logsumexp, to_tensor=to_tensor_all)
        self.argmax = adapter.classical_from_numpy.reduce(jnp.argmax, to_tensor=to_tensor_all)
        self.argmin = adapter.classical_from_numpy.reduce(jnp.argmin, to_tensor=to_tensor_all)

        self.sort = adapter.classical_from_numpy.sort(jnp.sort, to_tensor=to_tensor_all)
        self.argsort = adapter.classical_from_numpy.sort(jnp.argsort, to_tensor=to_tensor_all)
        self.roll = adapter.classical_from_numpy.roll(jnp.roll, to_tensor=to_tensor_all)
        self.flip = adapter.classical_from_numpy.preserve_shape(jnp.flip, to_tensor=to_tensor_all)
        self.softmax = adapter.classical_from_numpy.preserve_shape(jax.nn.softmax, to_tensor=to_tensor_all)
        self.log_softmax = adapter.classical_from_numpy.preserve_shape(jax.nn.log_softmax, to_tensor=to_tensor_all)

        def to_tensor_index(x, *args):
            x = _to_tensor(jnp.asarray, forward=[jnp.ndarray], convert=["numpy", "scalar"])(x)
            args = [_to_tensor(jnp.asarray, forward=[jnp.ndarray, "numpy", "scalar"], convert=[])(a) for a in args]
            return x, *args

        self.get_at = adapter.classical_from_numpy.get_at(jnp.ndarray.__getitem__, jnp.take, to_tensor=to_tensor_index)
        self.set_at = adapter.classical_from_numpy.update_at(
            lambda x, indices, updates: jnp.ndarray.at(x, indices).set(updates), to_tensor=to_tensor_index, broadcast=self.broadcast_to
        )
        self.add_at = adapter.classical_from_numpy.update_at(
            lambda x, indices, updates: jnp.ndarray.at(x, indices).add(updates), to_tensor=to_tensor_index, broadcast=self.broadcast_to
        )
        self.subtract_at = adapter.classical_from_numpy.update_at(
            lambda x, indices, updates: jnp.ndarray.at(x, indices).subtract(updates), to_tensor=to_tensor_index, broadcast=self.broadcast_to
        )

        self.arange = adapter.classical_from_numpy.arange(jnp.arange)
        self.split = adapter.classical_from_numpy.split(jnp.split, to_tensor=to_tensor_all)
        self.concatenate = adapter.classical_from_numpy.concatenate(jnp.concatenate, to_tensor=to_tensor_all)
        self.dot = adapter.classical_from_numpy.dot(jnp.dot, to_tensor=to_tensor_all)
        self.matmul = adapter.classical_from_numpy.matmul(jnp.matmul, to_tensor=to_tensor_all)
