import einx._src.adapter as adapter
import numpy as np
from functools import partial
from einx._src.util.functools import use_name_of
from .._util import _associative_binary_to_nary, _to_tensor
import einx._src.tracer as tracer


class ops:
    def __init__(self, tf):
        def to_tensor_all(*args):
            to_tensor_one = _to_tensor(tf.convert_to_tensor, forward=[tf.Tensor], convert=["scalar", "numpy"])
            return [to_tensor_one(arg) for arg in args]

        def dtype(x):
            if isinstance(x, tracer.signature.classical.Tensor):
                return tracer.signature.python.getattr(x, "dtype")
            elif isinstance(tf.Tensor, type) and isinstance(x, tf.Tensor):
                return x.dtype
            elif isinstance(x, int | float | bool | np.integer | np.floating | np.bool_):
                return tf.as_dtype(type(x))
            elif isinstance(x, tracer.signature.classical.ConvertibleTensor) and issubclass(
                x.concrete.type, int | float | bool | np.integer | np.floating | np.bool_
            ):
                return tf.as_dtype(x.concrete.type)
            elif isinstance(x, np.ndarray):
                return tf.as_dtype(x.dtype)
            elif isinstance(x, tracer.signature.classical.ConvertibleTensor) and issubclass(x.concrete.type, np.ndarray):
                return tf.as_dtype(x.concrete.type.dtype)
            else:
                raise ValueError(f"Cannot determine dtype of {type(x)}.")

        self.dtype = dtype
        self.tensor = tf.Tensor

        self.reshape = adapter.classical_from_numpy.reshape(tf.reshape, to_tensor=to_tensor_all)
        self.transpose = adapter.classical_from_numpy.transpose(tf.transpose, to_tensor=to_tensor_all)
        self.broadcast_to = adapter.classical_from_numpy.broadcast_to(tf.broadcast_to, to_tensor=to_tensor_all)
        self.diagonal = adapter.classical_from_numpy.diagonal(tf.linalg.diag_part, self.transpose, to_tensor=to_tensor_all, axis_always_last=True)
        self.stop_gradient = tf.stop_gradient

        def to_tensor_elementwise(*args):
            to_tensor_one = _to_tensor(tf.convert_to_tensor, forward=[tf.Tensor, "scalar"], convert=["numpy"])
            return [to_tensor_one(arg) for arg in args]

        self.add = adapter.classical_from_numpy.elementwise(_associative_binary_to_nary(tf.math.add), to_tensor=to_tensor_elementwise)
        self.subtract = adapter.classical_from_numpy.elementwise(tf.math.subtract, to_tensor=to_tensor_elementwise)
        self.multiply = adapter.classical_from_numpy.elementwise(_associative_binary_to_nary(tf.math.multiply), to_tensor=to_tensor_elementwise)
        self.true_divide = adapter.classical_from_numpy.elementwise(tf.math.truediv, to_tensor=to_tensor_elementwise)
        self.floor_divide = adapter.classical_from_numpy.elementwise(tf.math.floordiv, to_tensor=to_tensor_elementwise)
        self.divide = adapter.classical_from_numpy.elementwise(tf.math.divide, to_tensor=to_tensor_elementwise)
        self.logaddexp = adapter.classical_from_classical.logaddexp(self)
        self.logical_and = adapter.classical_from_numpy.elementwise(_associative_binary_to_nary(tf.math.logical_and), to_tensor=to_tensor_elementwise)
        self.logical_or = adapter.classical_from_numpy.elementwise(_associative_binary_to_nary(tf.math.logical_or), to_tensor=to_tensor_elementwise)
        self.where = adapter.classical_from_numpy.elementwise(tf.where, to_tensor=to_tensor_elementwise)
        self.maximum = adapter.classical_from_numpy.elementwise(_associative_binary_to_nary(tf.math.maximum), to_tensor=to_tensor_elementwise)
        self.minimum = adapter.classical_from_numpy.elementwise(_associative_binary_to_nary(tf.math.minimum), to_tensor=to_tensor_elementwise)
        self.less = adapter.classical_from_numpy.elementwise(tf.math.less, to_tensor=to_tensor_elementwise)
        self.less_equal = adapter.classical_from_numpy.elementwise(tf.math.less_equal, to_tensor=to_tensor_elementwise)
        self.greater = adapter.classical_from_numpy.elementwise(tf.math.greater, to_tensor=to_tensor_elementwise)
        self.greater_equal = adapter.classical_from_numpy.elementwise(tf.math.greater_equal, to_tensor=to_tensor_elementwise)
        self.equal = adapter.classical_from_numpy.elementwise(tf.math.equal, to_tensor=to_tensor_elementwise)
        self.not_equal = adapter.classical_from_numpy.elementwise(tf.math.not_equal, to_tensor=to_tensor_elementwise)
        self.exp = adapter.classical_from_numpy.elementwise(tf.math.exp, to_tensor=to_tensor_elementwise)
        self.log = adapter.classical_from_numpy.elementwise(tf.math.log, to_tensor=to_tensor_elementwise)
        self.negative = adapter.classical_from_numpy.elementwise(tf.math.negative, to_tensor=to_tensor_elementwise)
        self.divmod = adapter.classical_from_numpy.elementwise(lambda x, y: (tf.math.floordiv(x, y), tf.math.floormod(x, y)), to_tensor=to_tensor_elementwise)

        self.sum = adapter.classical_from_numpy.reduce(tf.math.reduce_sum, to_tensor=to_tensor_all)
        self.mean = adapter.classical_from_numpy.reduce(tf.math.reduce_mean, to_tensor=to_tensor_all)
        self.var = adapter.classical_from_numpy.reduce(tf.math.reduce_variance, to_tensor=to_tensor_all)
        self.std = adapter.classical_from_numpy.reduce(tf.math.reduce_std, to_tensor=to_tensor_all)
        self.prod = adapter.classical_from_numpy.reduce(tf.math.reduce_prod, to_tensor=to_tensor_all)
        self.count_nonzero = adapter.classical_from_numpy.reduce(tf.math.count_nonzero, to_tensor=to_tensor_all)
        self.any = adapter.classical_from_numpy.reduce(tf.math.reduce_any, to_tensor=to_tensor_all)
        self.all = adapter.classical_from_numpy.reduce(tf.math.reduce_all, to_tensor=to_tensor_all)
        self.max = adapter.classical_from_numpy.reduce(tf.math.reduce_max, to_tensor=to_tensor_all)
        self.min = adapter.classical_from_numpy.reduce(tf.math.reduce_min, to_tensor=to_tensor_all)
        self.logsumexp = adapter.classical_from_numpy.reduce(tf.math.reduce_logsumexp, to_tensor=to_tensor_all)
        self.argmax = adapter.classical_from_numpy.reduce(tf.math.argmax, to_tensor=to_tensor_all)
        self.argmin = adapter.classical_from_numpy.reduce(tf.math.argmin, to_tensor=to_tensor_all)

        self.sort = adapter.classical_from_numpy.sort(tf.sort, to_tensor=to_tensor_all)
        self.argsort = adapter.classical_from_numpy.sort(tf.argsort, to_tensor=to_tensor_all)
        self.roll = adapter.classical_from_numpy.roll(tf.roll, to_tensor=to_tensor_all)
        self.flip = adapter.classical_from_numpy.preserve_shape(tf.reverse, to_tensor=to_tensor_all)
        self.softmax = adapter.classical_from_classical.softmax(self)
        self.log_softmax = adapter.classical_from_classical.log_softmax(self)

        def to_tensor_index(x, y, *args):
            x = _to_tensor(tf.convert_to_tensor, forward=[], convert=["numpy", "scalar"])(x)
            y = _to_tensor(tf.convert_to_tensor, forward=["scalar"], convert=["numpy"])(y)
            args = [_to_tensor(tf.convert_to_tensor, forward=[], convert=["numpy", "scalar"])(a) for a in args]
            return x, y, *args

        self.get_at = adapter.classical_from_numpy.get_at(tf.Tensor.__getitem__, tf.gather, to_tensor=to_tensor_index)
        self.set_at = adapter.classical_from_numpy.update_at(
            tf.tensor_scatter_nd_update, to_tensor=to_tensor_all, broadcast=self.broadcast_to, reshape=self.reshape
        )
        self.add_at = adapter.classical_from_numpy.update_at(
            tf.tensor_scatter_nd_add, to_tensor=to_tensor_all, broadcast=self.broadcast_to, reshape=self.reshape
        )
        self.subtract_at = adapter.classical_from_numpy.update_at(
            tf.tensor_scatter_nd_sub, to_tensor=to_tensor_all, broadcast=self.broadcast_to, reshape=self.reshape
        )

        self.arange = adapter.classical_from_numpy.arange(tf.range)
        self.split = adapter.classical_from_numpy.split(tf.split, to_tensor=to_tensor_all, cumulative=False)
        self.concatenate = adapter.classical_from_numpy.concatenate(tf.concat, to_tensor=to_tensor_all)
        self.dot = adapter.classical_from_mlx.dot(tf.tensordot, to_tensor=to_tensor_all)
        self.matmul = adapter.classical_from_numpy.matmul(tf.matmul, to_tensor=to_tensor_all)
