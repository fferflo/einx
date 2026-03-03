import einx._src.adapter as adapter
from .._util import _to_tensor


def einsum_from_jax(jax):
    def to_tensor(*args):
        to_tensor_one = _to_tensor(jax.numpy.asarray, forward=[jax.numpy.ndarray, "numpy", "scalar"], convert=[])
        return [to_tensor_one(arg) for arg in args]

    return adapter.einsum_from_numpy(jax.numpy.einsum, to_tensor=to_tensor)
