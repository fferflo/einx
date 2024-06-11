from .base import *
import einx.tracer as tracer
from einx.tracer.tensor import op
import einx, types
from functools import partial
import functools


def create():
    tTensor = tracer.import_("Tensor", from_="tinygrad")
    tdtypes = tracer.import_("dtypes", from_="tinygrad")
    from tinygrad import Tensor, dtypes

    def scalar_to_tensor(x):
        if isinstance(x, (einx.tracer.Scalar, float, int)):
            return einx.tracer.apply(
                tTensor,
                args=[x],
                output=einx.tracer.Tensor([]),
            )
        else:
            return x

    def elementwise(func, convert_all_to_tensor=False):
        @einx.trace
        @functools.wraps(func)
        def outer(*args):
            if convert_all_to_tensor:
                args = [scalar_to_tensor(a) for a in args]
            else:
                args = [a for a in args]
                args[0] = scalar_to_tensor(args[0])
            return op.elementwise(func)(*args)

        return outer

    def reduce(func):
        @einx.trace
        @functools.wraps(func)
        def reduce(tensor, axis=None, **kwargs):
            keepdims = kwargs.get("keepdims", False)
            if axis is None:
                shape = ()
            else:
                axes = [axis] if isinstance(axis, int) else axis
                shape = list(tensor.shape)
                if keepdims:
                    for a in axes:
                        shape[a] = 1
                else:
                    for a in sorted(axes, reverse=True):
                        del shape[a]
                kwargs = {**kwargs, **{"axis": axis}}
            if "keepdims" in kwargs:
                kwargs["keepdim"] = kwargs.pop("keepdims")
            return tracer.apply(func, args=[tensor], kwargs=kwargs, output=tracer.Tensor(shape))

        return reduce

    def to_dtype(x):
        if isinstance(x, str):
            return getattr(dtypes, x)
        else:
            return x

    to_dtype2 = to_dtype

    class tinygrad(Backend):
        name = "tinygrad"
        tensor_types = [Tensor]

        to_dtype = staticmethod(to_dtype2)

        @staticmethod
        @einx.trace
        def to_tensor(tensor, shape):
            return einx.tracer.apply(
                tTensor,
                args=[tensor],
                output=einx.tracer.Tensor(shape),
            )

        reshape = op.reshape(tTensor.reshape)
        transpose = op.transpose(tTensor.permute)
        broadcast_to = op.broadcast_to(tTensor.expand)

        @classmethod
        @einx.trace
        def einsum(backend, equation, *tensors):
            x = equation.split("->")
            if len(x) != 2:
                raise ValueError("Invalid equation")
            inputs, output = x
            inputs = inputs.split(",")
            if len(inputs) != len(tensors):
                raise ValueError("Invalid equation")
            inputs = [x.strip().replace(" ", "") for x in inputs]
            tensors = [t for t in tensors]

            scalars = []
            for i in list(range(len(inputs)))[::-1]:
                if (len(inputs[i]) > 0) != (len(tensors[i].shape) > 0):
                    raise ValueError("Invalid equation")
                if len(inputs[i]) == 0:
                    scalars.append(tensors[i])
                    inputs.pop(i)
                    tensors.pop(i)

            if len(tensors) > 1:
                equation = ",".join(inputs) + "->" + output
                x = op.einsum(tTensor.einsum)(equation, *tensors)
            elif len(tensors) == 1:
                x = tensors[0]
            else:
                x = scalars[0]
                scalars = scalars[1:]
            for scalar in scalars:
                x = backend.multiply(x, scalar)

            return x

        @staticmethod
        @einx.trace
        def arange(n, dtype="int32"):
            if isinstance(dtype, str):
                dtype = getattr(tdtypes, dtype)
            return op.arange(tTensor.arange)(n, dtype=dtype)

        @staticmethod
        @einx.trace
        def concatenate(tensors, axis=0):
            shape = list(tensors[0].shape)
            shape[axis] = sum(tensor.shape[axis] for tensor in tensors)
            return tracer.apply(
                tTensor.cat, args=[*tensors], kwargs={"dim": axis}, output=tracer.Tensor(shape)
            )

        add = associative_binary_to_nary(elementwise(tTensor.add))
        subtract = elementwise(tTensor.sub)
        multiply = associative_binary_to_nary(elementwise(tTensor.mul))
        true_divide = elementwise(tTensor.div)
        floor_divide = elementwise(partial(tTensor.div, upcast=False))
        divide = elementwise(tTensor.div)
        logical_and = associative_binary_to_nary(elementwise(tTensor.mul))
        logical_or = associative_binary_to_nary(elementwise(tTensor.add))
        where = elementwise(tTensor.where)
        less = elementwise(tracer.Operator("<"))
        less_equal = elementwise(tracer.Operator("<="))
        greater = elementwise(tracer.Operator(">"))
        greater_equal = elementwise(tracer.Operator(">="))
        equal = elementwise(tracer.Operator("=="))
        not_equal = elementwise(tracer.Operator("!="))
        maximum = associative_binary_to_nary(elementwise(tTensor.maximum))
        minimum = associative_binary_to_nary(elementwise(tTensor.minimum))

        sum = reduce(tTensor.sum)
        mean = reduce(tTensor.mean)
        var = reduce(tTensor.var)
        std = reduce(tTensor.std)

        count_nonzero = reduce(tTensor.sum)
        min = reduce(tTensor.min)
        max = reduce(tTensor.max)
        # tinygrad's logsumexp currently does not support multiple axes, so
        # we use our custom implementation instead:
        # logsumexp = reduce(tTensor.logsumexp)

        log = op.elementwise(tTensor.log)
        exp = op.elementwise(tTensor.exp)
        sqrt = op.elementwise(tTensor.sqrt)
        rsqrt = op.elementwise(tTensor.rsqrt)
        square = op.elementwise(tTensor.square)

        @staticmethod
        @einx.trace
        def get_at(tensor, coordinates):
            raise NotImplementedError()

        @staticmethod
        @einx.trace
        def set_at(tensor, coordinates, updates):
            raise NotImplementedError()

        @staticmethod
        @einx.trace
        def add_at(tensor, coordinates, updates):
            raise NotImplementedError()

        @staticmethod
        @einx.trace
        def subtract_at(tensor, coordinates, updates):
            raise NotImplementedError()

        flip = op.keep_shape(tTensor.flip)
        softmax = op.keep_shape(tTensor.softmax)
        log_softmax = op.keep_shape(tTensor.log_softmax)

        @staticmethod
        @einx.trace
        def stop_gradient(tensor):
            return tensor  # TODO: set requires_grad to False?

        @staticmethod
        @einx.trace
        def vmap(op, in_axes, out_axes, input_shapes, output_shapes):
            raise NotImplementedError(
                "Functions relying on vmap are not supported for the tinygrad backend"
            )

        class random:
            @einx.trace
            def bernoulli(rng, p, shape):
                return (
                    einx.tracer.apply(
                        tTensor.rand,
                        args=[*shape],
                        output=einx.tracer.Tensor(shape),
                    )
                    <= p
                )

    return tinygrad()
