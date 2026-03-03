import einx._src.tracer as tracer
import einx._src.tracer.signature as signature
import types


class math:
    def __init__(self, math):
        self.add = signature.classical.elementwise(math.add, num_outputs=1)
        self.subtract = signature.classical.elementwise(math.subtract, num_outputs=1)
        self.multiply = signature.classical.elementwise(math.multiply, num_outputs=1)
        self.truediv = signature.classical.elementwise(math.truediv, num_outputs=1)
        self.floordiv = signature.classical.elementwise(math.floordiv, num_outputs=1)
        self.floormod = signature.classical.elementwise(math.floormod, num_outputs=1)
        self.divide = signature.classical.elementwise(math.divide, num_outputs=1)
        self.logical_and = signature.classical.elementwise(math.logical_and, num_outputs=1)
        self.logical_or = signature.classical.elementwise(math.logical_or, num_outputs=1)

        self.maximum = signature.classical.elementwise(math.maximum, num_outputs=1)
        self.minimum = signature.classical.elementwise(math.minimum, num_outputs=1)
        self.less = signature.classical.elementwise(math.less, num_outputs=1)
        self.less_equal = signature.classical.elementwise(math.less_equal, num_outputs=1)
        self.greater = signature.classical.elementwise(math.greater, num_outputs=1)
        self.greater_equal = signature.classical.elementwise(math.greater_equal, num_outputs=1)
        self.equal = signature.classical.elementwise(math.equal, num_outputs=1)
        self.not_equal = signature.classical.elementwise(math.not_equal, num_outputs=1)
        self.exp = signature.classical.elementwise(math.exp, num_outputs=1)
        self.log = signature.classical.elementwise(math.log, num_outputs=1)
        self.negative = signature.classical.elementwise(math.negative, num_outputs=1)

        self.reduce_sum = signature.classical.reduce(math.reduce_sum)
        self.reduce_mean = signature.classical.reduce(math.reduce_mean)
        self.reduce_variance = signature.classical.reduce(math.reduce_variance)
        self.reduce_std = signature.classical.reduce(math.reduce_std)
        self.reduce_prod = signature.classical.reduce(math.reduce_prod)
        self.count_nonzero = signature.classical.reduce(math.count_nonzero)
        self.reduce_all = signature.classical.reduce(math.reduce_all)
        self.reduce_any = signature.classical.reduce(math.reduce_any)
        self.reduce_min = signature.classical.reduce(math.reduce_min)
        self.reduce_max = signature.classical.reduce(math.reduce_max)
        self.reduce_logsumexp = signature.classical.reduce(math.reduce_logsumexp)
        self.argmax = signature.classical.reduce(math.argmax)
        self.argmin = signature.classical.reduce(math.argmin)


class tensorflow:
    def __init__(self, tf=None):
        if tf is None:
            tf = tracer.signature.python.import_("tensorflow", as_="tf")

        self.Tensor = tf.Tensor
        self.Tensor.__getitem__ = signature.classical.getitem()

        self.as_dtype = tf.as_dtype

        self.convert_to_tensor = signature.classical.preserve_shape(tf.convert_to_tensor)
        self.reshape = signature.classical.set_shape(tf.reshape)
        self.transpose = signature.classical.transpose(tf.transpose)
        self.broadcast_to = signature.classical.set_shape(tf.broadcast_to)
        self.range = signature.classical.arange(tf.range)
        self.concat = signature.classical.concatenate(tf.concat)
        self.split = signature.classical.split(tf.split, cumulative=False)

        self.where = signature.classical.elementwise(tf.where, num_outputs=1)

        self.tensordot = signature.classical.tensordot(tf.tensordot)
        self.matmul = signature.classical.matmul(tf.matmul)
        self.einsum = signature.classical.einsum(tf.einsum)

        self.roll = signature.classical.preserve_shape(tf.roll)
        self.reverse = signature.classical.preserve_shape(tf.reverse)
        self.sort = signature.classical.preserve_shape(tf.sort)
        self.argsort = signature.classical.preserve_shape(tf.argsort)

        self.argmin = signature.classical.reduce(tf.argmin)
        self.argmax = signature.classical.reduce(tf.argmax)

        self.stop_gradient = signature.classical.preserve_shape(tf.stop_gradient)

        self.gather = signature.classical.take(tf.gather)
        self.tensor_scatter_nd_update = signature.classical.setitem(tf.tensor_scatter_nd_update)
        self.tensor_scatter_nd_add = signature.classical.setitem(tf.tensor_scatter_nd_add)
        self.tensor_scatter_nd_sub = signature.classical.setitem(tf.tensor_scatter_nd_sub)

        self.math = math(tf.math)

        # self.experimental = types.SimpleNamespace(
        #     numpy=signature.numpy(tf.experimental.numpy),
        # )

        self.linalg = types.SimpleNamespace(diag_part=signature.classical.diagonal(tf.linalg.diag_part, axis_always_last=True))

        self.nn = types.SimpleNamespace(
            softmax=signature.classical.preserve_shape(tf.nn.softmax), log_softmax=signature.classical.preserve_shape(tf.nn.log_softmax)
        )
