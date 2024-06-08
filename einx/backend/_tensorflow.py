from .base import *
import einx.tracer as tracer
from einx.tracer.tensor import op
import einx, types
from functools import partial


def create():
    import tensorflow as tf
    import tensorflow.experimental.numpy as tnp

    ttf = tracer.import_("tensorflow", "tf")
    ttnp = tracer.import_("tensorflow.experimental.numpy", "tnp")

    def _broadcast_static_shape(shape1, shape2):
        assert len(shape1) == len(shape2) and all(
            s1 == s2 or s1 == 1 or s2 == 1 for s1, s2 in zip(shape1, shape2)
        )
        return tuple(max(s1, s2) for s1, s2 in zip(shape1, shape2))

    class tensorflow(Backend):
        name = "tensorflow"
        tensor_types = [tf.Tensor]

        @staticmethod
        @einx.trace
        def to_tensor(tensor, shape):
            return einx.tracer.apply(
                ttf.convert_to_tensor,
                args=[tensor],
                output=einx.tracer.Tensor(shape),
            )

        reshape = op.reshape(ttnp.reshape)
        transpose = op.transpose(ttnp.transpose)
        broadcast_to = op.broadcast_to(ttnp.broadcast_to)

        @staticmethod
        @einx.trace
        def einsum(equation, *tensors):
            return op.einsum(ttnp.einsum)(equation, *tensors, optimize="optimal")

        arange = op.arange(ttnp.arange)

        stack = op.stack(ttnp.stack)
        concatenate = op.concatenate(ttnp.concatenate)

        add = associative_binary_to_nary(op.elementwise(ttnp.add))
        subtract = op.elementwise(ttnp.subtract)
        multiply = associative_binary_to_nary(op.elementwise(ttnp.multiply))
        true_divide = op.elementwise(ttnp.true_divide)
        floor_divide = op.elementwise(ttnp.floor_divide)
        divide = op.elementwise(ttnp.divide)
        logical_and = associative_binary_to_nary(op.elementwise(ttnp.logical_and))
        logical_or = associative_binary_to_nary(op.elementwise(ttnp.logical_or))
        where = op.elementwise(ttnp.where)
        less = op.elementwise(ttnp.less)
        less_equal = op.elementwise(ttnp.less_equal)
        greater = op.elementwise(ttnp.greater)
        greater_equal = op.elementwise(ttnp.greater_equal)
        equal = op.elementwise(ttnp.equal)
        not_equal = op.elementwise(ttnp.not_equal)
        maximum = associative_binary_to_nary(op.elementwise(ttnp.maximum))
        minimum = associative_binary_to_nary(op.elementwise(ttnp.minimum))

        sum = op.reduce(ttnp.sum)
        mean = op.reduce(ttnp.mean)
        var = op.reduce(ttnp.var)
        std = op.reduce(ttnp.std)
        prod = op.reduce(ttnp.prod)
        count_nonzero = op.reduce(ttnp.count_nonzero)
        any = op.reduce(ttnp.any)
        all = op.reduce(ttnp.all)
        min = op.reduce(ttnp.min)
        max = op.reduce(ttnp.max)
        logsumexp = op.reduce(ttf.math.reduce_logsumexp)

        log = op.elementwise(ttnp.log)
        exp = op.elementwise(ttnp.exp)
        sqrt = op.elementwise(ttnp.sqrt)
        rsqrt = op.elementwise(ttf.math.rsqrt)
        square = op.elementwise(ttnp.square)

        @classmethod
        @einx.trace
        def get_at(backend, tensor, coordinates):
            coordinates, _ = backend._prepare_coordinates_and_update(coordinates, None)
            if isinstance(coordinates, tuple):
                out_shape = coordinates[0].shape
                coordinates = ttf.stack(coordinates, axis=-1)
            else:
                out_shape = coordinates.shape[:-1]
            return einx.tracer.apply(
                ttf.gather_nd,
                args=[tensor, coordinates],
                output=einx.tracer.Tensor(out_shape),
            )

        @classmethod
        @einx.trace
        def _prepare_coordinates_and_update(backend, coordinates, updates):
            assert updates is None or isinstance(updates, einx.tracer.Tensor)
            if isinstance(coordinates, tuple):
                assert all(isinstance(c, einx.tracer.Tensor) for c in coordinates)
                shape = coordinates[0].shape
                for c in coordinates[1:]:
                    shape = _broadcast_static_shape(shape, c.shape)
                coordinates = [backend.broadcast_to(c, shape) for c in coordinates]
                coordinates = backend.stack(coordinates, axis=-1)
            else:
                assert isinstance(coordinates, einx.tracer.Tensor)
                coordinates = coordinates[(slice(None),) * (coordinates.ndim - 1) + (None,)]
                coordinates = coordinates[..., None]

            assert updates is None or updates.ndim + 1 == coordinates.ndim

            # Broadcast to common shape
            if updates is None:
                shape = coordinates.shape[:-1]
            else:
                shape = _broadcast_static_shape(updates.shape, coordinates.shape[:-1])
            coordinates = backend.broadcast_to(coordinates, shape + coordinates.shape[-1:])
            if updates is not None:
                updates = backend.broadcast_to(updates, shape)

            return coordinates, updates

        @classmethod
        @einx.trace
        def set_at(backend, tensor, coordinates, updates):
            coordinates, updates = backend._prepare_coordinates_and_update(coordinates, updates)
            return einx.tracer.apply(
                ttf.tensor_scatter_nd_update,
                args=[tensor, coordinates, updates],
                output=einx.tracer.Tensor(tensor.shape),
            )

        @classmethod
        @einx.trace
        def add_at(backend, tensor, coordinates, updates):
            coordinates, updates = backend._prepare_coordinates_and_update(coordinates, updates)
            return einx.tracer.apply(
                ttf.tensor_scatter_nd_add,
                args=[tensor, coordinates, updates],
                output=einx.tracer.Tensor(tensor.shape),
            )

        @classmethod
        @einx.trace
        def subtract_at(backend, tensor, coordinates, updates):
            coordinates, updates = backend._prepare_coordinates_and_update(coordinates, updates)
            return einx.tracer.apply(
                ttf.tensor_scatter_nd_sub,
                args=[tensor, coordinates, updates],
                output=einx.tracer.Tensor(tensor.shape),
            )

        @staticmethod
        @einx.trace
        def flip(x, axis):
            if isinstance(axis, int):
                axis = [axis]
            return op.keep_shape(ttf.reverse)(x, axis)

        @staticmethod
        @einx.trace
        def roll(x, axis, shift):
            if isinstance(axis, int):
                axis = [axis]
            if isinstance(shift, int):
                shift = [shift]
            return op.keep_shape(ttf.roll)(x, tuple(shift), axis=tuple(axis))

        @staticmethod
        @einx.trace
        def softmax(x, axis):
            if isinstance(axis, (list, tuple)):
                if len(axis) != 1:
                    raise ValueError(
                        "Tensorflow only supports softmax along a single axis, "
                        f"got {len(axis)} axes"
                    )
                axis = axis[0]
            return op.keep_shape(ttf.nn.softmax)(x, axis=axis)

        @staticmethod
        @einx.trace
        def log_softmax(x, axis):
            if isinstance(axis, (list, tuple)):
                if len(axis) != 1:
                    raise ValueError(
                        "Tensorflow only supports log_softmax along a single axis, "
                        f"got {len(axis)} axes"
                    )
                axis = axis[0]
            return op.keep_shape(ttf.nn.log_softmax)(x, axis=axis)

        sqrt = op.keep_shape(ttf.math.sqrt)
        rsqrt = op.keep_shape(ttf.math.rsqrt)
        square = op.keep_shape(ttnp.square)

        stop_gradient = op.keep_shape(ttf.stop_gradient)

        @staticmethod
        def vmap(op, in_axes, out_axes, input_shapes, output_shapes):
            @einx.trace
            def inner(*args):
                # TODO: suboptimal (?) implementation of vmap in tensorflow that transposes the
                # vmapped axis to the front and calls tf.vectorized_map. Possible optimization:
                # Transpose only once for multiple vmaps?
                if len(args) != len(in_axes):
                    raise ValueError(f"Expected {len(in_axes)} arguments, got {len(args)}")
                value = {arg.shape[axis] for arg, axis in zip(args, in_axes) if axis is not None}
                if len(value) != 1:
                    raise ValueError(
                        f"Expected all arguments to have same size along vmap axis, got {value}"
                    )
                value = value.pop()

                # Move vmapped axes to front
                xs = []
                for arg, axis in zip(args, in_axes):
                    if axis is not None:
                        if axis != 0:
                            perm = [axis] + [a for a in range(len(arg.shape)) if a != axis]
                            arg = einx.tracer.op.transpose(ttnp.transpose)(arg, perm)
                    else:
                        arg = arg[tf.newaxis]
                    xs.append(arg)

                op2 = einx.trace(
                    lambda xs: op(*xs), args=[[einx.tracer.Tensor(x.shape[1:]) for x in xs]]
                )

                xs = einx.tracer.apply(
                    ttf.vectorized_map,
                    args=[op2, xs],
                    output=[einx.tracer.Tensor(shape) for shape in output_shapes],
                )

                if len(xs) != len(out_axes):
                    raise ValueError(
                        f"Expected {len(out_axes)} arguments from vmapped function, got {len(xs)}"
                    )

                # Move vmapped axis to out_axis
                xs = [
                    einx.tracer.op.transpose(ttnp.transpose)(
                        x,
                        [
                            (a + 1 if a < out_axis else (0 if a == out_axis else a))
                            for a in range(len(x.shape))
                        ],
                    )
                    for x, out_axis in zip(xs, out_axes)
                ]

                return tuple(xs)

            return inner

        class random:
            @einx.trace
            def bernoulli(rng, p, shape):
                return (
                    einx.tracer.apply(
                        ttf.random.uniform,
                        args=[shape],
                        kwargs={"minval": 0.0, "maxval": 1.0, "dtype": "float32", "seed": rng},
                        output=einx.tracer.Tensor(shape),
                    )
                    <= p
                )

    return tensorflow()
