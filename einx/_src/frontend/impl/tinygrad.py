import einx._src.tracer as tracer
import einx._src.adapter as adapter
from ..api import api
from ..types import Tensor
import types
import inspect
import functools
from functools import partial
from ..backend import registry
from ..backend import Backend
from ._util import _make_iskwarg
from ._docs import _make_doc_adapt_numpylike_reduce
from ._docs import _make_doc_adapt_numpylike_elementwise


def _get_backend_kwargs():
    tinygrad = tracer.signature.python.import_("tinygrad")
    optimizations = [
        tracer.optimizer.classical.SkipReshape(tinygrad.reshape),
        tracer.optimizer.classical.SkipTranspose(tinygrad.permute),
        tracer.optimizer.classical.SkipBroadcastTo(tinygrad.expand),
        tracer.optimizer.classical.SkipConcatenate(tinygrad.cat),
        tracer.optimizer.InlineGraph(),
        tracer.optimizer.SkipCast(),
    ]

    import tinygrad

    def is_supported_tensor(tensor):
        return isinstance(tensor, tinygrad.Tensor)

    def get_shape(tensor):
        return tuple(int(x) for x in tensor.shape)

    return {"optimizations": optimizations, "compiler": tracer.compiler.python, "is_supported_tensor": is_supported_tensor, "get_shape": get_shape}


def adapt_numpylike_reduce(op):
    iskwarg = lambda name, iskwarg=_make_iskwarg(op): name != "axis" and iskwarg(name)
    tinygrad = tracer.signature.tinygrad()

    classical = adapter.classical_from_tinygrad.ops(tinygrad)

    op = tracer.signature.python.constant(op)
    op = adapter.decomposednamedtensor_from_classical.reduce(op, expected_type=tinygrad.Tensor)
    op = adapter.namedtensor_from_decomposednamedtensor.op(op, classical)
    op = adapter.namedtensor_calltensorfactory.op(op, expected_type=tinygrad.Tensor)
    op = adapter.einx_from_namedtensor.reduce(op, iskwarg=iskwarg)

    return api(op, backend=types.SimpleNamespace(**_get_backend_kwargs()))


adapt_numpylike_reduce.__doc__ = _make_doc_adapt_numpylike_reduce()


def adapt_numpylike_elementwise(op):
    iskwarg = lambda name, iskwarg=_make_iskwarg(op): name != "axis" and iskwarg(name)
    tinygrad = tracer.signature.tinygrad()

    classical = adapter.classical_from_tinygrad.ops(tinygrad)

    op = tracer.signature.python.constant(op)
    op = adapter.decomposednamedtensor_from_classical.elementwise(op, classical, expected_type=tinygrad.Tensor)
    op = adapter.namedtensor_from_decomposednamedtensor.op(op, classical)
    op = adapter.namedtensor_calltensorfactory.op(op, expected_type=tinygrad.Tensor)
    op = adapter.einx_from_namedtensor.elementwise(op, iskwarg=iskwarg)

    return api(op, backend=types.SimpleNamespace(**_get_backend_kwargs()))


adapt_numpylike_elementwise.__doc__ = _make_doc_adapt_numpylike_elementwise()


def _backend_creator(create):
    def new_create():
        tinygrad = tracer.signature.tinygrad()

        classical = adapter.classical_from_tinygrad.ops(tinygrad)

        decomposednamedtensor_ops, name = create(tinygrad, classical)

        namedtensor_ops = adapter.namedtensor_from_decomposednamedtensor.ops(decomposednamedtensor_ops, classical)
        namedtensor_ops = adapter.namedtensor_calltensorfactory.ops(namedtensor_ops, expected_type=tinygrad.Tensor)
        einx_ops = adapter.einx_from_namedtensor.ops(namedtensor_ops)

        return Backend(ops=einx_ops, name=name, priority=-5, **_get_backend_kwargs())

    return new_create


@_backend_creator
def create_backend_numpylike(tinygrad, classical):
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

    return decomposednamedtensor_ops, "tinygrad.numpylike"


registry.register_on_import("tinygrad", "tinygrad.numpylike", create_backend_numpylike)


@_backend_creator
def create_backend_einsum(tinygrad, classical):
    einsum = adapter.einsum_from_tinygrad(tinygrad)

    decomposednamedtensor_ops = {
        "id": adapter.decomposednamedtensor_from_einsum.id(einsum),
        "sum": adapter.decomposednamedtensor_from_einsum.sum(einsum),
        "multiply": adapter.decomposednamedtensor_from_einsum.multiply(einsum),
        "dot": adapter.decomposednamedtensor_from_einsum.dot(einsum),
    }

    return decomposednamedtensor_ops, "tinygrad.einsum"


registry.register_on_import("tinygrad", "tinygrad.einsum", create_backend_einsum)


def create_backend():
    einx_ops_numpylike = create_backend_numpylike().ops
    einx_ops_einsum = create_backend_einsum().ops
    einx_ops = einx_ops_numpylike | {"dot": einx_ops_einsum["dot"]}

    return Backend(ops=einx_ops, name="tinygrad", priority=0, **_get_backend_kwargs())


registry.register_on_import("tinygrad", "tinygrad", create_backend)
