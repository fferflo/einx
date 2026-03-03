import einx._src.adapter as adapter
from .._util import _to_tensor


def einsum_from_tensorflow(tf):
    def to_tensor(*args):
        to_tensor_one = _to_tensor(tf.convert_to_tensor, forward=[tf.Tensor], convert=["numpy", "scalar"])
        return [to_tensor_one(arg) for arg in args]

    return adapter.einsum_from_numpy(tf.einsum, to_tensor=to_tensor)
