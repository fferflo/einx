import einx._src.tracer as tracer
import einx._src.adapter as adapter
from ..api import api
import types
import inspect
import functools
from functools import partial
from ..backend import registry
from ..backend import Backend
from einx._src.frontend.errors import ImportBackendError
import threading
from ._util import _make_iskwarg, _unsupported_op
from ._docs import _make_doc_adapt_numpylike_reduce
from ._docs import _make_doc_adapt_numpylike_elementwise
from ._docs import _make_doc_adapt_with_vmap


def _raise_on_invalid_version():
    import torch

    version = tuple(int(i) for i in torch.__version__.split(".")[:2])
    if version < (2, 2):
        raise ImportBackendError(f"einx with PyTorch requires PyTorch version >= 2.2, but found {torch.__version__}. einx functions are disabled for PyTorch.")


_has_allowed_in_graph = False
_has_allowed_in_graph_lock = threading.Lock()


def _allow_ops_in_graph():
    global _has_allowed_in_graph
    if not _has_allowed_in_graph:
        with _has_allowed_in_graph_lock:
            if not _has_allowed_in_graph:
                import torch
                from einx._src.frontend.ops import ops

                for op in ops:
                    torch.compiler.allow_in_graph(op)
                _has_allowed_in_graph = True


def _get_backend_kwargs():
    torch = tracer.signature.python.import_("torch")
    optimizations = [
        tracer.optimizer.classical.SkipReshape(torch.reshape),
        tracer.optimizer.classical.SkipTranspose(torch.permute),
        tracer.optimizer.classical.SkipBroadcastTo(torch.broadcast_to),
        tracer.optimizer.classical.SkipConcatenate(torch.cat),
        tracer.optimizer.InlineGraph(),
        tracer.optimizer.SkipCast(),
    ]

    import torch

    def is_supported_tensor(tensor):
        return isinstance(tensor, torch.Tensor)

    def get_shape(tensor):
        return tuple(int(x) for x in tensor.shape)

    return {"optimizations": optimizations, "compiler": tracer.compiler.python, "is_supported_tensor": is_supported_tensor, "get_shape": get_shape}


def adapt_with_vmap(op, signature=None):
    _raise_on_invalid_version()
    iskwarg = _make_iskwarg(op)

    device_stack = adapter.TorchDeviceStack()

    torch = tracer.signature.torch()

    classical = adapter.classical_from_torch.ops(torch, get_device=device_stack.get_device)
    vmap = adapter.vmap_from_torch(torch, get_device=device_stack.get_device)

    op = tracer.signature.python.constant(op)
    op = adapter.decomposednamedtensor_from_vmap.op(op, vmap, expected_type=torch.Tensor)
    op = adapter.namedtensor_from_decomposednamedtensor.op(op, classical)
    op = device_stack.namedtensor.op(op)
    op = adapter.namedtensor_calltensorfactory.op(op, expected_type=torch.Tensor)
    op = adapter.einx_from_namedtensor.op(op, iskwarg=iskwarg, el_op=signature, implicit_output="bijective")

    op = api(op, backend=types.SimpleNamespace(**_get_backend_kwargs()))
    import torch

    torch.compiler.allow_in_graph(op)

    return op


adapt_with_vmap.__doc__ = _make_doc_adapt_with_vmap("torch", "``torch.vmap``")


def adapt_numpylike_reduce(op):
    _raise_on_invalid_version()
    iskwarg = lambda name, iskwarg=_make_iskwarg(op): name != "axis" and iskwarg(name)

    device_stack = adapter.TorchDeviceStack()

    torch = tracer.signature.torch()

    classical = adapter.classical_from_torch.ops(torch, get_device=device_stack.get_device)

    op = tracer.signature.python.constant(op)
    op = adapter.decomposednamedtensor_from_classical.reduce(op, expected_type=torch.Tensor)
    op = adapter.namedtensor_from_decomposednamedtensor.op(op, classical)
    op = device_stack.namedtensor.op(op)
    op = adapter.namedtensor_calltensorfactory.op(op, expected_type=torch.Tensor)
    op = adapter.einx_from_namedtensor.reduce(op, iskwarg=iskwarg)

    op = api(op, backend=types.SimpleNamespace(**_get_backend_kwargs()))
    import torch

    torch.compiler.allow_in_graph(op)

    return op


adapt_numpylike_reduce.__doc__ = _make_doc_adapt_numpylike_reduce()


def adapt_numpylike_elementwise(op):
    _raise_on_invalid_version()
    iskwarg = _make_iskwarg(op)

    device_stack = adapter.TorchDeviceStack()

    torch = tracer.signature.torch()

    classical = adapter.classical_from_torch.ops(torch, get_device=device_stack.get_device)

    op = tracer.signature.python.constant(op)
    op = adapter.decomposednamedtensor_from_classical.elementwise(op, classical, expected_type=torch.Tensor)
    op = adapter.namedtensor_from_decomposednamedtensor.op(op, classical)
    op = device_stack.namedtensor.op(op)
    op = adapter.namedtensor_calltensorfactory.op(op, expected_type=torch.Tensor)
    op = adapter.einx_from_namedtensor.elementwise(op, iskwarg=iskwarg)

    op = api(op, backend=types.SimpleNamespace(**_get_backend_kwargs()))
    import torch

    torch.compiler.allow_in_graph(op)

    return op


adapt_numpylike_elementwise.__doc__ = _make_doc_adapt_numpylike_elementwise()


def _backend_creator(create):
    def new_create():
        _raise_on_invalid_version()
        _allow_ops_in_graph()

        device_stack = adapter.TorchDeviceStack()

        torch = tracer.signature.torch()

        classical = adapter.classical_from_torch.ops(torch, get_device=device_stack.get_device)

        decomposednamedtensor_ops, name = create(torch, classical, device_stack)

        namedtensor_ops = adapter.namedtensor_from_decomposednamedtensor.ops(decomposednamedtensor_ops, classical)
        namedtensor_ops = device_stack.namedtensor.ops(namedtensor_ops)
        namedtensor_ops = adapter.namedtensor_calltensorfactory.ops(namedtensor_ops, expected_type=torch.Tensor)
        einx_ops = adapter.einx_from_namedtensor.ops(namedtensor_ops)

        return Backend(ops=einx_ops, name=name, priority=-5, **_get_backend_kwargs())

    return new_create


@_backend_creator
def create_backend_numpylike(torch, classical, device_stack):
    decomposednamedtensor_ops = (
        {name: adapter.decomposednamedtensor_from_classical.elementwise(getattr(classical, name), classical) for name in adapter.ops.elementwise}
        | {name: adapter.decomposednamedtensor_from_classical.reduce(getattr(classical, name)) for name in adapter.ops.reduce}
        | {name: adapter.decomposednamedtensor_from_classical.preserve_shape(getattr(classical, name)) for name in adapter.ops.preserve_shape}
        | {name: adapter.decomposednamedtensor_from_classical.argfind(getattr(classical, name), classical) for name in adapter.ops.argfind}
        | {name: adapter.decomposednamedtensor_from_classical.update_at_ravelled(getattr(classical, name), classical) for name in adapter.ops.update_at}
        | {
            "get_at": adapter.decomposednamedtensor_from_classical.get_at_ravelled(classical),
            "dot": adapter.decomposednamedtensor_from_classical.dot(classical),
        }
    )

    return decomposednamedtensor_ops, "torch.numpylike"


registry.register_on_import("torch", "torch.numpylike", create_backend_numpylike)


@_backend_creator
def create_backend_einsum(torch, classical, device_stack):
    einsum = adapter.einsum_from_torch(torch, get_device=device_stack.get_device)

    decomposednamedtensor_ops = {
        "id": adapter.decomposednamedtensor_from_einsum.id(einsum),
        "sum": adapter.decomposednamedtensor_from_einsum.sum(einsum),
        "multiply": adapter.decomposednamedtensor_from_einsum.multiply(einsum),
        "dot": adapter.decomposednamedtensor_from_einsum.dot(einsum),
    }

    return decomposednamedtensor_ops, "torch.einsum"


registry.register_on_import("torch", "torch.einsum", create_backend_einsum)


@_backend_creator
def create_backend_vmap(torch, classical, device_stack):
    vmap = adapter.vmap_from_torch(torch, get_device=device_stack.get_device)

    elementary_ops = adapter.elementary_from_classical.ops(classical)

    get_at_error_message = (
        "get_at is not supported by the torch.vmap backend. As of testing this, "
        "torch.vmap is not compatible with scalar indexing operations and raises the following error:\n\"vmap: It looks like you're calling "
        ".item() on a Tensor. We don't support vmap over calling .item() on a Tensor, please try to rewrite what you're doing with other operations. "
        'If error is occurring somewhere inside PyTorch internals, please file a bug report."\n'
        "Please use another PyTorch backend for this operation."
    )

    decomposednamedtensor_ops = (
        {
            name: adapter.decomposednamedtensor_from_vmap.op(
                elementary_ops[name], vmap, expected_type=torch.Tensor, allow_squeeze_unsqueeze=True, classical=classical
            )
            for name in adapter.ops.all
        }
        | {"get_at": _unsupported_op("get_at", "torch.vmap", get_at_error_message)}
        | {name: _unsupported_op(name, "torch.vmap") for name in adapter.ops.update_at}
    )

    return decomposednamedtensor_ops, "torch.vmap"


registry.register_on_import("torch", "torch.vmap", create_backend_vmap)


def create_backend():
    einx_ops_numpylike = create_backend_numpylike().ops
    einx_ops_einsum = create_backend_einsum().ops
    einx_ops = einx_ops_numpylike | {"dot": einx_ops_einsum["dot"]}

    return Backend(ops=einx_ops, name="torch", priority=0, **_get_backend_kwargs())


registry.register_on_import("torch", "torch", create_backend)
