import einx._src.adapter as adapter
from .._util import _to_tensor


def einsum_from_mlx(mlx):
    def to_tensor(*args):
        to_tensor_one = _to_tensor(mlx.core.array, forward=[mlx.core.array], convert=["numpy", "scalar"])
        return [to_tensor_one(arg) for arg in args]

    return adapter.einsum_from_numpy(mlx.core.einsum, to_tensor=to_tensor, multiply=adapter.classical_from_mlx.ops(mlx).multiply)
