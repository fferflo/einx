from .._util import _to_tensor
import einx._src.adapter as adapter


def vmap_from_torch(torch, get_device):
    def to_tensor(*args):
        to_tensor_one = _to_tensor(lambda x: torch.asarray(x, device=get_device()), forward=[torch.Tensor], convert=["scalar", "numpy"])
        return [to_tensor_one(arg) for arg in args]

    return adapter.vmap_from_jax(lambda op, in_axes, out_axes: torch.vmap(op, in_dims=in_axes, out_dims=out_axes), to_tensor=to_tensor)
