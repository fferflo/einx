import einx._src.tracer as tracer
import einx._src.adapter as adapter
from ..api import api
import types
import inspect
from ..backend import registry
from ..backend import Backend
import threading
from ._util import _make_iskwarg

from .torch import _raise_on_invalid_version
from .torch import _allow_ops_in_graph


def _get_backend_kwargs():
    optimizations = [tracer.optimizer.InlineGraph(), tracer.optimizer.SkipCast()]

    import torch

    def is_supported_tensor(tensor):
        return isinstance(tensor, torch.Tensor)

    def get_shape(tensor):
        return tuple(int(x) for x in tensor.shape)

    return {"optimizations": optimizations, "compiler": tracer.compiler.python, "is_supported_tensor": is_supported_tensor, "get_shape": get_shape}


def adapt(op):
    _raise_on_invalid_version()
    iskwarg = lambda name, iskwarg=_make_iskwarg(op): name != "axis" and iskwarg(name)

    device_stack = adapter.TorchDeviceStack()

    torch = tracer.signature.torch()
    functorchdim = tracer.signature.python.import_("functorch.dim", as_="ftdim")

    op = tracer.signature.python.constant(op)
    op = adapter.namedtensor_from_functorchdim.op(op, torch, functorchdim, get_device=device_stack.get_device)
    op = device_stack.namedtensor.op(op)
    op = adapter.namedtensor_calltensorfactory.op(op, expected_type=torch.Tensor)
    op = adapter.einx_from_namedtensor.op(op, iskwarg=iskwarg, allow_concat=False)

    func = api(op, backend=types.SimpleNamespace(**_get_backend_kwargs()))

    import torch

    torch.compiler.allow_in_graph(func)
    return func
