from .base import *
import einx.tracer as tracer
from einx.tracer.tensor import op
import einx, types
from functools import partial


def create():
    import torch as torch_

    version = tuple(int(i) for i in torch_.__version__.split(".")[:2])
    if version < (2, 0):
        message = "einx with PyTorch requires PyTorch version >= 2, but found "
        f"{torch_.__version__}. einx functions are disabled for PyTorch."
        print(f"WARNING: {message}")
        return ErrorBackend(message)

    @einx.trace
    def move_scalars_to_device(args, scalar_indices=None):
        device = None
        for arg in args:
            if isinstance(arg, einx.tracer.Tensor) and not isinstance(arg, einx.tracer.Scalar):
                device = arg.device
                break
        if device is None:
            raise ValueError("Failed to determine the PyTorch device placement of parameters")

        def to_tensor(i, x):
            if einx.tracer.is_scalar(x) and (scalar_indices is None or i in scalar_indices):
                return einx.tracer.apply(
                    ttorch.asarray,
                    args=[x],
                    kwargs={"device": device},
                    output=einx.tracer.Tensor(()),
                )
            else:
                return x

        return [to_tensor(i, arg) for i, arg in enumerate(args)]

    def move_scalars_to_device_in_elementwise(op, scalar_indices=None):
        @einx.trace
        @functools.wraps(op)
        def wrapper(*args, **kwargs):
            args = move_scalars_to_device(args, scalar_indices)
            return op(*args, **kwargs)

        return wrapper

    MARKER_DECORATED_CONSTRUCT_GRAPH = "__einx_decorated_construct_graph"

    ttorch = tracer.import_("torch")
    import torch as torch_

    def to_tuple(x):
        if isinstance(x, tuple):
            return x
        elif isinstance(x, list):
            return tuple(x)
        elif isinstance(x, np.ndarray):
            return tuple(x.tolist())
        else:
            raise ValueError(f"Cannot convert {type(x)} to tuple")

    to_tuple2 = to_tuple

    def to_dtype(x):
        if isinstance(x, str):
            return vars(torch_)[x]
        else:
            return x

    to_dtype2 = to_dtype

    if "compiler" in dir(torch_):
        tcompiler = ttorch.compiler
        compiler = torch_.compiler
    else:
        tcompiler = tracer.import_("torch._dynamo", "_dynamo")
        import torch._dynamo as compiler

    import torch._dynamo as _dynamo

    if "capture_func_transforms" in vars(_dynamo.config):
        # Allow torch.vmap to be used inside torch.compile
        _dynamo.config.capture_func_transforms = True

    class torch(Backend):
        name = "torch"
        tensor_types = [torch_.Tensor]
        function_name = (
            "_call_impl"  # Workaround for: https://github.com/pytorch/pytorch/issues/124269
        )

        to_tuple = staticmethod(to_tuple2)
        to_dtype = staticmethod(to_dtype2)

        @staticmethod
        @einx.trace
        def to_tensor(arg, shape):
            assert False

        @staticmethod
        @einx.trace
        def all_to_tensor(tensors, convert_scalars=False):
            device = None
            for tensor in tensors:
                if type(tensor) == einx.tracer.Tensor:
                    device = tensor.device
                    break
            if device is None:
                device = ttorch.device("cpu")

            def to_tensor(tensor):
                if isinstance(tensor, einx.tracer.TensorRequiringConversion) or (
                    convert_scalars and einx.tracer.is_scalar(tensor)
                ):
                    return einx.tracer.apply(
                        ttorch.asarray,
                        args=[tensor],
                        kwargs={"device": device},
                        output=einx.tracer.Tensor(tensor.shape),
                    )
                else:
                    return tensor

            tensors = [to_tensor(tensor) for tensor in tensors]

            return tensors

        @staticmethod
        @einx.trace
        def reshape(tensor, shape):
            if einx.tracer.get_shape(tensor) == shape:
                return tensor
            else:
                return op.reshape(ttorch.reshape)(tensor, to_tuple(shape))

        @staticmethod
        @einx.trace
        def transpose(tensor, perm):
            return op.transpose(ttorch.permute)(tensor, to_tuple(perm))

        @staticmethod
        @einx.trace
        def broadcast_to(tensor, shape):
            return op.broadcast_to(ttorch.broadcast_to)(tensor, to_tuple(shape))

        @staticmethod
        @einx.trace
        def einsum(equation, *tensors):
            tensors = move_scalars_to_device(tensors)
            return op.einsum(ttorch.einsum)(equation, *tensors)

        @staticmethod
        @einx.trace
        def arange(n, dtype="int32"):
            return op.arange(ttorch.arange)(n, dtype=to_dtype(dtype))

        stack = op.stack(ttorch.stack)
        concatenate = op.concatenate(ttorch.concatenate)

        add = associative_binary_to_nary(op.elementwise(ttorch.add))
        subtract = op.elementwise(ttorch.subtract)
        multiply = associative_binary_to_nary(op.elementwise(ttorch.multiply))
        true_divide = op.elementwise(ttorch.true_divide)
        floor_divide = op.elementwise(ttorch.floor_divide)
        divide = op.elementwise(ttorch.divide)
        logical_and = move_scalars_to_device_in_elementwise(
            associative_binary_to_nary(op.elementwise(ttorch.logical_and))
        )
        logical_or = move_scalars_to_device_in_elementwise(
            associative_binary_to_nary(op.elementwise(ttorch.logical_or))
        )
        where = move_scalars_to_device_in_elementwise(
            op.elementwise(ttorch.where), scalar_indices=[0]
        )
        less = op.elementwise(einx.tracer.Operator("<"))
        less_equal = op.elementwise(einx.tracer.Operator("<="))
        greater = op.elementwise(einx.tracer.Operator(">"))
        greater_equal = op.elementwise(einx.tracer.Operator(">="))
        equal = op.elementwise(einx.tracer.Operator("=="))
        not_equal = op.elementwise(einx.tracer.Operator("!="))
        maximum = move_scalars_to_device_in_elementwise(
            associative_binary_to_nary(op.elementwise(ttorch.maximum))
        )
        minimum = move_scalars_to_device_in_elementwise(
            associative_binary_to_nary(op.elementwise(ttorch.minimum))
        )

        sum = op.reduce(ttorch.sum)
        mean = op.reduce(ttorch.mean)
        var = op.reduce(ttorch.var)
        std = op.reduce(ttorch.std)
        prod = op.reduce(ttorch.prod)
        count_nonzero = op.reduce(ttorch.count_nonzero)
        any = op.reduce(ttorch.any)
        all = op.reduce(ttorch.all)
        min = op.reduce(ttorch.min)
        max = op.reduce(ttorch.max)
        logsumexp = op.reduce(ttorch.logsumexp)

        log = op.elementwise(ttorch.log)
        exp = op.elementwise(ttorch.exp)
        sqrt = op.elementwise(ttorch.sqrt)
        rsqrt = op.elementwise(ttorch.rsqrt)
        square = op.elementwise(ttorch.square)

        @staticmethod
        @einx.trace
        def get_at(tensor, coordinates):
            if isinstance(coordinates, tuple):
                if (
                    any(isinstance(c, (slice, int, einx.tracer.Scalar)) for c in coordinates)
                    or coordinates[0].ndim > 0
                ):
                    return tensor[coordinates]
                else:
                    # Fix for https://github.com/pytorch/functorch/issues/747
                    # Scalar coordinates cause problems with torch.vmap and throw an error:
                    # "RuntimeError: vmap: It looks like you're calling .item() on a Tensor.
                    # We don't support vmap over calling .item() on a Tensor ..."
                    # As a workaround, we add a dummy dimension and remove it after the indexing
                    # operation.
                    return tensor[tuple(c[None] for c in coordinates)][0]
            else:
                if (
                    isinstance(coordinates, (slice, int, einx.tracer.Scalar))
                    or coordinates.ndim > 0
                ):
                    return tensor[coordinates]
                else:
                    # See above
                    return tensor[coordinates[None]][0]

        @staticmethod
        @einx.trace
        def set_at(tensor, coordinates, updates):
            if isinstance(coordinates, tuple):
                if (
                    any(isinstance(c, (slice, int, einx.tracer.Scalar)) for c in coordinates)
                    or coordinates[0].ndim > 0
                ):
                    return tensor.__setitem__(coordinates, updates)
                else:
                    # See above
                    coordinates = tuple(c[None] for c in coordinates)
                    updates = updates[None]
                    return tensor.__setitem__(coordinates, updates)
            else:
                if (
                    isinstance(coordinates, (slice, int, einx.tracer.Scalar))
                    or coordinates.ndim > 0
                ):
                    return tensor.__setitem__(coordinates, updates)
                else:
                    # See above
                    coordinates = coordinates[None]
                    updates = updates[None]
                    return tensor.__setitem__(coordinates, updates)

        @staticmethod
        @einx.trace
        def add_at(tensor, coordinates, updates):
            if isinstance(coordinates, tuple):
                if (
                    any(isinstance(c, (slice, int, einx.tracer.Scalar)) for c in coordinates)
                    or coordinates[0].ndim > 0
                ):
                    return tensor.__setitem__(
                        coordinates, tensor.__getitem__(coordinates).__iadd__(updates)
                    )
                else:
                    # See above
                    coordinates = tuple(c[None] for c in coordinates)
                    updates = updates[None]
                    return tensor.__setitem__(
                        coordinates, tensor.__getitem__(coordinates).__iadd__(updates)
                    )
            else:
                if (
                    isinstance(coordinates, (slice, int, einx.tracer.Scalar))
                    or coordinates.ndim > 0
                ):
                    return tensor.__setitem__(
                        coordinates, tensor.__getitem__(coordinates).__iadd__(updates)
                    )
                else:
                    # See above
                    coordinates = coordinates[None]
                    updates = updates[None]
                    return tensor.__setitem__(
                        coordinates, tensor.__getitem__(coordinates).__iadd__(updates)
                    )

            return tensor

        @staticmethod
        @einx.trace
        def subtract_at(tensor, coordinates, updates):
            if isinstance(coordinates, tuple):
                if (
                    any(isinstance(c, (slice, int, einx.tracer.Scalar)) for c in coordinates)
                    or coordinates[0].ndim > 0
                ):
                    return tensor.__setitem__(
                        coordinates, tensor.__getitem__(coordinates).__isub__(updates)
                    )
                else:
                    # See above
                    coordinates = tuple(c[None] for c in coordinates)
                    updates = updates[None]
                    return tensor.__setitem__(
                        coordinates, tensor.__getitem__(coordinates).__isub__(updates)
                    )
            else:
                if (
                    isinstance(coordinates, (slice, int, einx.tracer.Scalar))
                    or coordinates.ndim > 0
                ):
                    return tensor.__setitem__(
                        coordinates, tensor.__getitem__(coordinates).__isub__(updates)
                    )
                else:
                    # See above
                    coordinates = coordinates[None]
                    updates = updates[None]
                    return tensor.__setitem__(
                        coordinates, tensor.__getitem__(coordinates).__isub__(updates)
                    )

            return tensor

        @staticmethod
        @einx.trace
        def flip(tensor, axis):
            if isinstance(axis, int):
                axis = [axis]
            return op.keep_shape(ttorch.flip)(tensor, axis)

        @staticmethod
        @einx.trace
        def roll(tensor, shift, axis):
            if isinstance(axis, int):
                axis = [axis]
            return op.keep_shape(ttorch.roll)(tensor, shift, axis)

        @staticmethod
        @einx.trace
        def softmax(tensor, axis):
            if isinstance(axis, (list, tuple)):
                if len(axis) != 1:
                    raise ValueError(
                        "PyTorch only supports softmax along a single axis, "
                        f"got {len(axis)} axes"
                    )
                axis = axis[0]
            return op.keep_shape(ttorch.softmax)(tensor, axis)

        @staticmethod
        @einx.trace
        def log_softmax(tensor, axis):
            if isinstance(axis, (list, tuple)):
                if len(axis) != 1:
                    raise ValueError(
                        "PyTorch only supports log_softmax along a single axis, "
                        f"got {len(axis)} axes"
                    )
                axis = axis[0]
            return op.keep_shape(ttorch.nn.functional.log_softmax)(tensor, axis)

        @staticmethod
        @einx.trace
        def stop_gradient(x):
            raise NotImplementedError("stop_gradient is currently not implemented for PyTorch")

        @staticmethod
        @einx.trace
        def vmap(op, in_axes, out_axes, input_shapes, output_shapes):
            op = einx.tracer.apply(
                ttorch.vmap,
                args=[op],
                kwargs={
                    "in_dims": tuple(in_axes) if isinstance(in_axes, list) else in_axes,
                    "out_dims": tuple(out_axes) if isinstance(out_axes, list) else out_axes,
                },
                signature="vmap",
                output=einx.tracer.Function(
                    output=[einx.tracer.Tensor(shape) for shape in output_shapes]
                ),
            )
            op = einx.tracer.apply(
                tcompiler.allow_in_graph,
                args=[op],
                comment="Workaround for https://github.com/pytorch/pytorch/issues/94674",
            )
            return op

        class random:
            @einx.trace
            def bernoulli(rng, p, shape):
                return (
                    einx.tracer.apply(
                        ttorch.bernoulli,
                        args=[ttorch.full(to_tuple(shape), p)],
                        kwargs={"generator": rng},
                        output=einx.tracer.Tensor(shape),
                    )
                    > 0.5
                )

    einx.jit.decorate_traced_functions(compiler.allow_in_graph)

    return torch()
