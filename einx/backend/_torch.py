import einx
from .base import Backend, ErrorBackend, associative_binary_to_nary

def to_tuple(x):
    if isinstance(x, tuple):
        return x
    elif isinstance(x, list):
        return tuple(x)
    elif isinstance(x, np.ndarray):
        return tuple(x.tolist())
    else:
        raise ValueError(f"Cannot convert {type(x)} to tuple")

def make_torch_backend():
    import torch as torch_

    version = tuple(int(i) for i in torch_.__version__.split(".")[:2])
    if version < (2, 0):
        message = f"einx with PyTorch requires PyTorch version >= 2, but found {torch_.__version__}. einx functions are disabled for PyTorch."
        print(f"WARNING: {message}")
        return ErrorBackend(message)

    class torch(Backend):
        @staticmethod
        def to_tensor(tensor):
            if torch_.is_tensor(tensor):
                return tensor
            else:
                return torch_.asarray(tensor)

        tensor = torch_.Tensor
        name = "torch"

        cast = lambda tensor, dtype: tensor.type(vars(torch_)[dtype] if isinstance(dtype, str) else dtype)
        reshape = lambda tensor, shape: torch_.reshape(tensor, to_tuple(shape))
        transpose = torch_.permute
        broadcast_to = lambda tensor, shape: torch_.broadcast_to(tensor, to_tuple(shape))
        einsum = torch_.einsum
        dot = torch_.matmul
        swapaxes = torch_.swapaxes
        def arange(n, dtype):
            return torch_.arange(n, dtype=vars(torch_)[dtype])

        stack = torch_.stack
        concatenate = torch_.cat

        zeros = lambda shape, dtype="float32": torch_.zeros(*shape, dtype=vars(torch_)[dtype] if isinstance(dtype, str) else dtype)
        ones = lambda shape, dtype="float32": torch_.ones(*shape, dtype=vars(torch_)[dtype] if isinstance(dtype, str) else dtype)

        add = associative_binary_to_nary(torch_.add)
        subtract = torch_.subtract
        multiply = associative_binary_to_nary(torch_.multiply)
        true_divide = torch_.true_divide
        floor_divide = torch_.floor_divide
        divide = torch_.divide
        logical_and = associative_binary_to_nary(torch_.logical_and)
        logical_or = associative_binary_to_nary(torch_.logical_or)
        where = torch_.where
        less = torch_.less
        less_equal = torch_.less_equal
        greater = torch_.greater
        greater_equal = torch_.greater_equal
        equal = torch_.equal
        not_equal = torch_.not_equal
        @associative_binary_to_nary
        def maximum(a, b):
            return torch_.maximum(torch.to_tensor(a), torch.to_tensor(b)) # TODO: add support for python scalars everywhere
        @associative_binary_to_nary
        def minimum(a, b):
            return torch_.minimum(torch.to_tensor(a), torch.to_tensor(b))

        sum = torch_.sum
        mean = torch_.mean
        var = torch_.var
        std = torch_.std
        prod = torch_.prod
        count_nonzero = torch_.count_nonzero
        any = torch_.any
        all = torch_.all
        min = torch_.min
        max = torch_.max
        logsumexp = torch_.logsumexp

        def get_at(tensor, coordinates):
            if isinstance(coordinates, tuple):
                if coordinates[0].ndim == 0:
                    # Fix for https://github.com/pytorch/functorch/issues/747
                    # Scalar coordinates cause problems with torch.vmap and throw an error:
                    # "RuntimeError: vmap: It looks like you're calling .item() on a Tensor. We don't support vmap over calling .item() on a Tensor ..."
                    # As a workaround, we add a dummy dimension and remove it after the indexing operation.
                    return tensor[tuple(c[None] for c in coordinates)][0]
                else:
                    return tensor[coordinates]
            else:
                if coordinates.ndim == 0:
                    # See above
                    return tensor[coordinates[None]][0]
                else:
                    return tensor[coordinates]
        def set_at(tensor, coordinates, updates):
            tensor[coordinates] = updates
            return tensor
        def add_at(tensor, coordinates, updates):
            tensor[coordinates] += updates
            return tensor
        def subtract_at(tensor, coordinates, updates):
            tensor[coordinates] -= updates
            return tensor

        def flip(tensor, axis):
            if isinstance(axis, int):
                axis = [axis]
            return torch_.flip(tensor, axis)
        def roll(tensor, shift, axis):
            if isinstance(axis, int):
                axis = [axis]
            return torch_.roll(tensor, shift, axis)
        def softmax(tensor, axis):
            if isinstance(axis, (list, tuple)):
                if len(axis) != 1:
                    raise ValueError(f"PyTorch only supports softmax along a single axis, got {len(axis)} axes")
                axis = axis[0]
            return torch_.softmax(tensor, axis)
        def log_softmax(tensor, axis):
            if isinstance(axis, (list, tuple)):
                if len(axis) != 1:
                    raise ValueError(f"PyTorch only supports log_softmax along a single axis, got {len(axis)} axes")
                axis = axis[0]
            return torch_.nn.functional.log_softmax(tensor, axis)

        sqrt = torch_.sqrt
        rsqrt = torch_.rsqrt
        square = torch_.square

        allclose = torch_.allclose

        def vmap(op, in_axes, out_axes, input_shapes=None, output_shapes=None):
            return torch_.vmap(
                op,
                in_dims=tuple(in_axes) if isinstance(in_axes, list) else in_axes,
                out_dims=tuple(out_axes) if isinstance(out_axes, list) else out_axes,
            )

        class random:
            def bernoulli(rng, p, shape):
                return torch_.bernoulli(torch_.full(shape, p), generator=rng) > 0.5

    if "compiler" in dir(torch_):
        einx.lru_cache.decorate_traced_functions(torch_.compiler.allow_in_graph)
    else:
        import torch._dynamo as _dynamo
        einx.lru_cache.decorate_traced_functions(_dynamo.allow_in_graph)

    return torch
