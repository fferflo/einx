import einx._src.adapter as adapter
from .._util import _to_tensor


def einsum_from_torch(torch, get_device):
    def to_tensor(*args):
        to_tensor_one = _to_tensor(lambda x: torch.asarray(x, device=get_device()), forward=[torch.Tensor], convert=["numpy", "scalar"])
        return [to_tensor_one(arg) for arg in args]

    return adapter.einsum_from_numpy(torch.einsum, to_tensor=to_tensor)
