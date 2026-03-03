from .._util import _to_tensor
from einx._src.util.functools import use_name_of
import einx._src.adapter as adapter


def vmap_from_mlx(mlx):
    def to_tensor(*args):
        to_tensor_one = _to_tensor(mlx.core.array, forward=[mlx.core.array], convert=["numpy", "scalar"])
        return [to_tensor_one(arg) for arg in args]

    return adapter.vmap_from_jax(mlx.core.vmap, to_tensor=to_tensor)
