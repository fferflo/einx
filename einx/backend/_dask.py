from .base import Backend, associative_binary_to_nary
import einx.tracer as tracer
from einx.tracer.tensor import op
import einx
import types
from functools import partial


def create():
    import dask.array as da

    tda = tracer.import_("dask.array", "da")

    class dask(Backend):
        name = "dask"
        tensor_types = [da.Array]

        @staticmethod
        @einx.trace
        def to_tensor(tensor, shape):
            return einx.tracer.apply(
                tda.asarray,
                args=[tensor],
                output=einx.tracer.Tensor(shape),
            )

        @staticmethod
        @einx.trace
        def reshape(tensor, shape):
            if einx.tracer.is_scalar(tensor):
                tensor = tda.asarray(tensor)
            return op.reshape(tda.reshape)(tensor, shape)

        transpose = op.transpose(tda.transpose)
        broadcast_to = op.broadcast_to(tda.broadcast_to)

        @classmethod
        @einx.trace
        def einsum(backend, equation, *operands):
            exprs = equation.split("->")
            if len(exprs) != 2:
                raise ValueError("Invalid einsum equation")
            in_exprs = exprs[0].split(",")
            out_expr = exprs[1]

            # Remove scalars
            scalars = []
            for in_expr, operand in zip(in_exprs, operands):
                if (len(in_expr) == 0) != (operand.shape == ()):
                    raise ValueError(
                        f"Tensor and einsum expression do not match: {in_expr} and {operand.shape}"
                    )
                if operand.shape == ():
                    scalars.append(operand)
            operands = [operand for operand in operands if operand.shape != ()]
            in_exprs = [in_expr for in_expr in in_exprs if len(in_expr) > 0]
            assert len(in_exprs) == len(operands)
            equation = ",".join(in_exprs) + "->" + out_expr

            # Call without scalars
            if len(operands) == 1:
                if in_exprs[0] != out_expr:
                    output = op.einsum(tda.einsum)(equation, *operands)
                else:
                    output = operands[0]
            elif len(operands) > 1:
                output = op.einsum(tda.einsum)(equation, *operands)
            else:
                output = None

            # Multiply scalars
            if len(scalars) > 0:
                if output is None:
                    output = backend.multiply(*scalars)
                else:
                    output = backend.multiply(output, *scalars)

            return output

        arange = op.arange(tda.arange)

        @staticmethod
        @einx.trace
        def stack(tensors, axis=0):
            tensors = [tda.asarray(t) if einx.tracer.is_scalar(t) else t for t in tensors]
            return op.stack(tda.stack)(tensors, axis=axis)

        @staticmethod
        @einx.trace
        def concatenate(tensors, axis=0):
            tensors = [tda.asarray(t) if einx.tracer.is_scalar(t) else t for t in tensors]
            return op.concatenate(tda.concatenate)(tensors, axis=axis)

        add = associative_binary_to_nary(op.elementwise(tda.add))
        subtract = op.elementwise(tda.subtract)
        multiply = associative_binary_to_nary(op.elementwise(tda.multiply))
        true_divide = op.elementwise(tda.true_divide)
        floor_divide = op.elementwise(tda.floor_divide)
        divide = op.elementwise(tda.divide)
        logical_and = associative_binary_to_nary(op.elementwise(tda.logical_and))
        logical_or = associative_binary_to_nary(op.elementwise(tda.logical_or))
        where = op.elementwise(tda.where)
        less = op.elementwise(tda.less)
        less_equal = op.elementwise(tda.less_equal)
        greater = op.elementwise(tda.greater)
        greater_equal = op.elementwise(tda.greater_equal)
        equal = op.elementwise(tda.equal)
        not_equal = op.elementwise(tda.not_equal)
        maximum = associative_binary_to_nary(op.elementwise(tda.maximum))
        minimum = associative_binary_to_nary(op.elementwise(tda.minimum))

        sum = op.reduce(tda.sum)
        mean = op.reduce(tda.mean)
        var = op.reduce(tda.var)
        std = op.reduce(tda.std)
        prod = op.reduce(tda.prod)
        count_nonzero = op.reduce(tda.count_nonzero)
        any = op.reduce(tda.any)
        all = op.reduce(tda.all)
        min = op.reduce(tda.min)
        max = op.reduce(tda.max)

        log = op.elementwise(tda.log)
        exp = op.elementwise(tda.exp)
        sqrt = op.elementwise(tda.sqrt)
        square = op.elementwise(tda.square)

        @staticmethod
        @einx.trace
        def get_at(tensor, coordinates):
            return tensor[coordinates]

        @staticmethod
        @einx.trace
        def set_at(tensor, coordinates, updates):
            return tensor.__setitem__(coordinates, updates)

        @staticmethod
        @einx.trace
        def add_at(tensor, coordinates, updates):
            return tensor.__setitem__(
                coordinates, tensor.__getitem__(coordinates).__iadd__(updates)
            )

        @staticmethod
        @einx.trace
        def subtract_at(tensor, coordinates, updates):
            return tensor.__setitem__(
                coordinates, tensor.__getitem__(coordinates).__isub__(updates)
            )

        flip = op.keep_shape(tda.flip)
        roll = op.keep_shape(tda.roll)

        @staticmethod
        @einx.trace
        def vmap(*args, **kwargs):
            raise NotImplementedError(
                "Functions relying on vmap are not supported for the dask backend"
            )

    return dask()
