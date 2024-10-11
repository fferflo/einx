from .base import Backend, associative_binary_to_nary
import einx.tracer as tracer
from einx.tracer.tensor import op
import einx
import types
from functools import partial


def create():
    import mlx.core as mx

    version = tuple(int(i) for i in mx.__version__.split(".")[:3])
    if version < (0, 16, 1):
        return InvalidBackend(
            "mlx",
            "einx with mlx requires mlx version >= 0.16.1, but found "
            f"{mx.__version__}. einx functions are disabled for mlx.",
            tensor_types=[mx.array],
        )

    tmx = tracer.import_("mlx.core", "mx")

    def to_tuple(x):
        if isinstance(x, tuple):
            return x
        elif isinstance(x, list):
            return tuple(x)
        elif isinstance(x, np.ndarray):
            return tuple(x.tolist())
        else:
            raise ValueError(f"Cannot convert {type(x)} to tuple")

    def to_dtype(x):
        if isinstance(x, str):
            if x == "bool":
                return mx.bool_
            else:
                return vars(mx)[x]
        else:
            return x

    to_dtype2 = to_dtype

    class mlx(Backend):
        name = "mlx"
        tensor_types = [mx.array]

        to_dtype = staticmethod(to_dtype2)

        @staticmethod
        @einx.trace
        def to_tensor(tensor, shape):
            return einx.tracer.apply(
                tmx.array,
                args=[tensor],
                output=einx.tracer.Tensor(shape),
            )

        @staticmethod
        @einx.trace
        def reshape(tensor, shape):
            if einx.tracer.is_scalar(tensor):
                tensor = tmx.array(tensor)
            return op.reshape(tmx.reshape)(tensor, to_tuple(shape))

        transpose = op.transpose(tmx.transpose)
        broadcast_to = op.broadcast_to(tmx.broadcast_to)

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
                    output = op.einsum(tmx.einsum)(equation, *operands)
                else:
                    output = operands[0]
            elif len(operands) > 1:
                output = op.einsum(tmx.einsum)(equation, *operands)
            else:
                output = None

            # Multiply scalars
            if len(scalars) > 0:
                if output is None:
                    output = backend.multiply(*scalars)
                else:
                    output = backend.multiply(output, *scalars)

            return output

        @staticmethod
        @einx.trace
        def arange(start, stop=None, step=None, dtype="int32"):
            args = [start]
            if stop is not None:
                args.append(stop)
            if step is not None:
                args.append(step)
            return op.arange(tmx.arange)(*args, dtype=to_dtype(dtype))

        stack = op.stack(tmx.stack)
        concatenate = op.concatenate(tmx.concatenate)

        add = associative_binary_to_nary(op.elementwise(tmx.add))
        subtract = op.elementwise(tmx.subtract)
        multiply = associative_binary_to_nary(op.elementwise(tmx.multiply))
        true_divide = op.elementwise(tmx.divide)
        floor_divide = op.elementwise(tmx.floor_divide)
        divide = op.elementwise(tmx.divide)
        mod = op.elementwise(tmx.remainder)
        logical_and = associative_binary_to_nary(op.elementwise(tmx.logical_and))
        logical_or = associative_binary_to_nary(op.elementwise(tmx.logical_or))
        where = op.elementwise(tmx.where)
        less = op.elementwise(tmx.less)
        less_equal = op.elementwise(tmx.less_equal)
        greater = op.elementwise(tmx.greater)
        greater_equal = op.elementwise(tmx.greater_equal)
        equal = op.elementwise(tmx.equal)
        not_equal = op.elementwise(tmx.not_equal)
        maximum = associative_binary_to_nary(op.elementwise(tmx.maximum))
        minimum = associative_binary_to_nary(op.elementwise(tmx.minimum))

        sum = op.reduce(tmx.sum)
        mean = op.reduce(tmx.mean)
        var = op.reduce(tmx.var)
        prod = op.reduce(tmx.prod)
        count_nonzero = op.reduce(tmx.count_nonzero)
        any = op.reduce(tmx.any)
        all = op.reduce(tmx.all)
        min = op.reduce(tmx.min)
        max = op.reduce(tmx.max)
        logsumexp = op.reduce(tmx.logsumexp)

        log = op.elementwise(tmx.log)
        exp = op.elementwise(tmx.exp)
        sqrt = op.elementwise(tmx.sqrt)
        rsqrt = op.elementwise(tmx.rsqrt)
        square = op.elementwise(tmx.square)

        @staticmethod
        @einx.trace
        def get_at(tensor, coordinates):
            return tensor[coordinates]

        @staticmethod
        @einx.trace
        def set_at(tensor, coordinates, updates):
            return einx.tracer.apply(
                tensor.at[coordinates].set, args=[updates], output=einx.tracer.Tensor(tensor.shape)
            )

        @staticmethod
        @einx.trace
        def add_at(tensor, coordinates, updates):
            return einx.tracer.apply(
                tensor.at[coordinates].add, args=[updates], output=einx.tracer.Tensor(tensor.shape)
            )

        @staticmethod
        @einx.trace
        def subtract_at(tensor, coordinates, updates):
            return einx.tracer.apply(
                tensor.at[coordinates].add, args=[-updates], output=einx.tracer.Tensor(tensor.shape)
            )

        softmax = op.keep_shape(tmx.softmax)

        stop_gradient = op.keep_shape(tmx.stop_gradient)

        vmap = op.vmap(tmx.vmap)

        sqrt = tmx.sqrt
        rsqrt = tmx.rsqrt
        square = tmx.square

        class random:
            @einx.trace
            def bernoulli(rng, p, shape):
                einx.tracer.apply(
                    tmx.random.bernoulli,
                    args=[p, shape, rng],
                    output=einx.tracer.Tensor(shape),
                )

    return mlx()
