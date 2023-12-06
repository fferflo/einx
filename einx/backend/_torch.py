import einx
from .base import base_backend

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
    import torch._dynamo as _dynamo
    class torch(base_backend):
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

        stack = torch_.stack
        concatenate = torch_.cat

        zeros = lambda shape, dtype="float32": torch_.zeros(*shape, dtype=vars(torch_)[dtype] if isinstance(dtype, str) else dtype)
        ones = lambda shape, dtype="float32": torch_.ones(*shape, dtype=vars(torch_)[dtype] if isinstance(dtype, str) else dtype)

        add = torch_.add
        subtract = torch_.subtract
        multiply = torch_.multiply
        true_divide = torch_.true_divide
        floor_divide = torch_.floor_divide
        divide = torch_.divide
        logical_and = torch_.logical_and
        logical_or = torch_.logical_or
        where = torch_.where
        less = torch_.less
        less_equal = torch_.less_equal
        greater = torch_.greater
        greater_equal = torch_.greater_equal
        equal = torch_.equal
        not_equal = torch_.not_equal
        def maximum(a, b):
            return torch_.maximum(torch.to_tensor(a), torch.to_tensor(b)) # TODO: add support for python scalars everywhere
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

        def get_at(tensor, coordinates):
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

        sqrt = torch_.sqrt
        rsqrt = torch_.rsqrt
        square = torch_.square

        allclose = torch_.allclose

        def vmap(op, in_axes, out_axes):
            return torch_.vmap(
                op,
                in_dims=tuple(in_axes) if isinstance(in_axes, list) else in_axes,
                out_dims=tuple(out_axes) if isinstance(out_axes, list) else out_axes,
            )

        class random:
            def bernoulli(rng, p, shape):
                return torch_.bernoulli(torch_.full(shape, p), generator=rng) > 0.5

    _dynamo.allow_in_graph(einx.dot)
    _dynamo.allow_in_graph(einx.rearrange)
    _dynamo.allow_in_graph(einx.elementwise)
    _dynamo.allow_in_graph(einx.reduce)
    _dynamo.allow_in_graph(einx.vmap)
    _dynamo.allow_in_graph(einx.vmap_with_axis)
    _dynamo.allow_in_graph(einx.nn.norm)
    _dynamo.allow_in_graph(einx.nn.linear)
    _dynamo.allow_in_graph(einx.nn.dropout)

    for op_name in einx.elementwise._op_names + einx.reduce._op_names + einx.vmap_with_axis._op_names:
        _dynamo.allow_in_graph(getattr(einx, op_name))

    return torch
