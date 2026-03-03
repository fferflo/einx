import functools
import einx._src.adapter as adapter
import numpy as np
from ._util import _unravel
from einx._src.util.functools import use_name_of


def id(*args):
    if len(args) == 1:
        return args[0]
    else:
        return tuple(args)


def reduce(op):
    @use_name_of(op)
    def reduce(tensor):
        return op(tensor)

    return reduce


def preserve_shape(op):
    @use_name_of(op)
    def preserve_shape(tensor, **kwargs):
        return op(tensor, **kwargs)

    return preserve_shape


def elementwise(op):
    @use_name_of(op)
    def elementwise(*tensors):
        if any(tensor.ndim != 0 for tensor in tensors):
            raise ValueError("All tensors must be scalars for elementwise operation.")
        return op(*tensors)

    return elementwise


def dot(classical):
    return classical.dot


def get_at(classical):
    def get_at(tensor, *coordinates):
        coords2 = []
        for coord in coordinates:
            if coord.ndim == 0:
                coords2.append(coord)
            elif coord.ndim == 1:
                for i in range(coord.shape[0]):
                    coords2.append(classical.get_at(coord, i, axis=0))
            else:
                raise ValueError("Coordinate tensors must be scalars or 1D arrays.")
        if tensor.ndim != len(coords2):
            raise ValueError("Number of coordinates must match the number of dimensions of the tensor.")
        return classical.get_at(tensor, tuple(coords2))

    return get_at


def update_at(op, classical):
    @use_name_of(op)
    def update_at(*tensors):
        tensor = tensors[0]
        coordinates = tensors[1:-1]
        updates = tensors[-1]

        coords2 = []
        for coord in coordinates:
            if coord.ndim == 0:
                coords2.append(coord)
            elif coord.ndim == 1:
                for i in range(coord.shape[0]):
                    coords2.append(classical.get_at(coord, i, axis=0))
            else:
                raise ValueError("Coordinate tensors must be scalars or 1D arrays.")
        if tensor.ndim != len(coords2):
            raise ValueError("Number of coordinates must match the number of dimensions of the tensor.")

        return op(tensor, tuple(coords2), updates)

    return update_at


def argfind(op, classical):
    @use_name_of(op)
    def argfind(tensor):
        idx = op(classical.reshape(tensor, (np.prod(tensor.shape),)))
        if tensor.ndim == 0:
            raise ValueError("Tensor must have at least one dimension for argfind operation.")
        else:
            return _unravel(classical, idx, tensor.shape, axis=0)

    return argfind


def ops(classical):
    return (
        {name: elementwise(getattr(classical, name)) for name in adapter.ops.elementwise}
        | {name: reduce(getattr(classical, name)) for name in adapter.ops.reduce}
        | {name: preserve_shape(getattr(classical, name)) for name in adapter.ops.preserve_shape}
        | {name: argfind(getattr(classical, name), classical) for name in adapter.ops.argfind}
        | {name: update_at(getattr(classical, name), classical) for name in adapter.ops.update_at}
        | {"get_at": get_at(classical), "dot": dot(classical), "id": id}
    )
