import einx._src.adapter as adapter
from .._util import _to_tensor


def einsum_from_arrayapi(xp):
    def to_tensor(*args):
        to_tensor_one = _to_tensor(xp.asarray, forward=[adapter.tensortype_from_arrayapi(xp)], convert=["numpy", "scalar"])
        return [to_tensor_one(arg) for arg in args]

    return adapter.einsum_from_numpy(xp.einsum, to_tensor=to_tensor)
