from functools import partial
from .base import Backend, associative_binary_to_nary

def make_tensorflow_backend():
    import tensorflow as tf
    import tensorflow.experimental.numpy as tnp

    def prepare_coordinates_and_update(coordinates, updates):
        if isinstance(coordinates, tuple):
            shape = coordinates[0].shape
            for c in coordinates[1:]:
                shape = tf.broadcast_static_shape(shape, c.shape)
            coordinates = [tf.broadcast_to(c, shape) for c in coordinates]
            coordinates = tf.stack(coordinates, axis=-1)
        else:
            coordinates = coordinates[..., tf.newaxis]
        assert updates.ndim + 1 == coordinates.ndim

        # Broadcast to common shape
        shape = tf.broadcast_static_shape(updates.shape, coordinates.shape[:-1])
        coordinates = tf.broadcast_to(coordinates, shape + coordinates.shape[-1:])
        updates = tf.broadcast_to(updates, shape)

        return coordinates, updates

    class tensorflow(Backend):
        @staticmethod
        def to_tensor(tensor):
            tensor = tf.convert_to_tensor(tensor)
            if any(s is None for s in tensor.shape):
                raise ValueError("Tensorflow tensors with dynamic shape are not supported")
            return tensor

        tensor = tf.Tensor
        name = "tensorflow"

        cast = tf.cast
        reshape = tf.reshape
        transpose = tf.transpose
        broadcast_to = tf.broadcast_to
        einsum = partial(tnp.einsum, optimize="optimal")
        dot = tnp.dot
        swapaxes = tnp.swapaxes
        arange = tnp.arange

        stack = tnp.stack
        concatenate = tnp.concatenate

        zeros = lambda shape, dtype="float32": tf.zeros(shape, dtype=dtype)
        ones = lambda shape, dtype="float32": tf.ones(shape, dtype=dtype)

        add = associative_binary_to_nary(tnp.add)
        subtract = tnp.subtract
        multiply = associative_binary_to_nary(tnp.multiply)
        true_divide = tnp.true_divide
        floor_divide = tnp.floor_divide
        divide = tnp.divide
        logical_and = associative_binary_to_nary(tnp.logical_and)
        logical_or = associative_binary_to_nary(tnp.logical_or)
        where = tnp.where
        less = tnp.less
        less_equal = tnp.less_equal
        greater = tnp.greater
        greater_equal = tnp.greater_equal
        equal = tnp.equal
        not_equal = tnp.not_equal
        maximum = associative_binary_to_nary(tnp.maximum)
        minimum = associative_binary_to_nary(tnp.minimum)

        sum = tnp.sum
        mean = tnp.mean
        var = tnp.var
        var = tnp.std
        prod = tnp.prod
        count_nonzero = tnp.count_nonzero
        any = tnp.any
        all = tnp.all
        min = tnp.min
        max = tnp.max
        logsumexp = tf.math.reduce_logsumexp

        def get_at(tensor, coordinates):
            return tensor[coordinates]
        def set_at(tensor, coordinates, updates):
            coordinates, updates = prepare_coordinates_and_update(coordinates, updates)
            return tf.tensor_scatter_nd_update(tensor, coordinates, updates)
        def add_at(tensor, coordinates, updates):
            coordinates, updates = prepare_coordinates_and_update(coordinates, updates)
            return tf.tensor_scatter_nd_add(tensor, coordinates, updates)
        def subtract_at(tensor, coordinates, updates):
            coordinates, updates = prepare_coordinates_and_update(coordinates, updates)
            return tf.tensor_scatter_nd_sub(tensor, coordinates, updates)

        def flip(x, axis):
            if isinstance(axis, int):
                axis = [axis]
            return tf.reverse(x, axis)
        def roll(x, axis, shift):
            if isinstance(axis, int):
                axis = [axis]
            if isinstance(shift, int):
                shift = [shift]
            return tf.roll(x, tuple(shift), axis=tuple(axis))
        def softmax(x, axis):
            if isinstance(axis, (list, tuple)):
                if len(axis) != 1:
                    raise ValueError(f"Tensorflow only supports softmax along a single axis, got {len(axis)} axes")
                axis = axis[0]
            return tf.nn.softmax(x, axis=axis)
        def log_softmax(x, axis):
            if isinstance(axis, (list, tuple)):
                if len(axis) != 1:
                    raise ValueError(f"Tensorflow only supports log_softmax along a single axis, got {len(axis)} axes")
                axis = axis[0]
            return tf.nn.log_softmax(x, axis=axis)

        sqrt = tf.math.sqrt
        rsqrt = tf.math.rsqrt
        square = tnp.square

        allclose = tnp.allclose

        def vmap(op, in_axes, out_axes, input_shapes=None, output_shapes=None):
            def inner(*args):
                # TODO: suboptimal (?) implementation of vmap in tensorflow that transposes the vmapped axis to the front and calls tf.vectorized_map
                if len(args) != len(in_axes):
                    raise ValueError(f"Expected {len(in_axes)} arguments, got {len(args)}")
                value = set(arg.shape[axis] for arg, axis in zip(args, in_axes) if not axis is None)
                if len(value) != 1:
                    raise ValueError(f"Expected all arguments to have same size along vmap axis, got {value}")
                value = value.pop()

                # Move vmapped axes to front
                xs = []
                for arg, axis in zip(args, in_axes):
                    if not axis is None:
                        if axis != 0:
                            perm = [axis] + [a for a in range(len(arg.shape)) if a != axis]
                            arg = tf.transpose(arg, perm=perm)
                    else:
                        arg = arg[tf.newaxis]
                    xs.append(arg)

                xs = tf.vectorized_map(lambda xs: op(*xs), xs)
                if len(xs) != len(out_axes):
                    raise ValueError(f"Expected {len(out_axes)} arguments from vmapped function, got {len(xs)}")

                # Move vmapped axis to out_axis
                xs = [tf.transpose(x, perm=[(a + 1 if a < out_axis else (0 if a == out_axis else a)) for a in range(len(x.shape))]) for x, out_axis in zip(xs, out_axes)]

                return tuple(xs)
            inner.__name__ = f"vmap({op.__name__ if '__name__' in dir(op) else str(op)}, in_axes={in_axes}, out_axes={out_axes})"
            return inner

        class random:
            def bernoulli(rng, p, shape):
                return tf.random.uniform(shape, minval=0.0, maxval=1.0, dtype="float32", seed=rng) <= p

    return tensorflow
