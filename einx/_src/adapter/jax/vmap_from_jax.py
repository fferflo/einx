from .._util import _to_tensor
from einx._src.util.functools import use_name_of


def vmap_from_jax(op=None, to_tensor=None):
    if to_tensor is None:
        jax = op
        op = jax.vmap

        def to_tensor(*args):
            to_tensor_one = _to_tensor(jax.numpy.asarray, forward=[jax.numpy.ndarray, "scalar"], convert=["numpy"])
            return [to_tensor_one(arg) for arg in args]

    def vmap(func, in_axes=0, out_axes=0):
        func = op(func, in_axes=in_axes, out_axes=out_axes)

        @use_name_of(func)
        def wrapped_func(*args):
            args = to_tensor(*args)
            return func(*args)

        return wrapped_func

    return vmap
