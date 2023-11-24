import einx

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
    class torch:
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

        elementwise = lambda *args, op, **kwargs: op(*args, **kwargs)
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
        maximum = lambda a, b: torch_.maximum(torch.to_tensor(a), torch.to_tensor(b))
        minimum = lambda a, b: torch_.minimum(torch.to_tensor(a), torch.to_tensor(b)) # TODO: add support for python scalars everywhere

        reduce = lambda *args, op, **kwargs: op(*args, **kwargs)
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

        sqrt = torch_.sqrt
        rsqrt = torch_.rsqrt
        square = torch_.square

        allclose = torch_.allclose

        vmap = lambda op, in_axes, out_axes: torch_.vmap(
            op,
            in_dims=tuple(in_axes) if isinstance(in_axes, list) else in_axes,
            out_dims=tuple(out_axes) if isinstance(out_axes, list) else out_axes,
        )

        def assert_shape(tensor, shape):
            assert tensor.shape == shape, f"Expected shape {shape}, got {tensor.shape}"
            return tensor

    _dynamo.allow_in_graph(einx.dot)
    _dynamo.allow_in_graph(einx.rearrange)
    _dynamo.allow_in_graph(einx.elementwise)
    _dynamo.allow_in_graph(einx.reduce)
    _dynamo.allow_in_graph(einx.nn.meanvar_norm)
    _dynamo.allow_in_graph(einx.nn.linear)

    for op_name in einx.elementwise._op_names + einx.reduce._op_names:
        _dynamo.allow_in_graph(getattr(einx, op_name))

    return torch
