from ..backend import registry
from ..backend import Backend
import einx._src.adapter as adapter
import einx._src.tracer as tracer
from ._util import _make_iskwarg
from ._util import _unsupported_op
from ..api import api
import types
from ._docs import _make_doc_adapt_numpylike_reduce
from ._docs import _make_doc_adapt_numpylike_elementwise


def _get_backend_kwargs():
    np = tracer.signature.python.import_("numpy", as_="np")
    optimizations = [
        tracer.optimizer.classical.SkipReshape(np.reshape),
        tracer.optimizer.classical.SkipTranspose(np.transpose),
        tracer.optimizer.classical.SkipBroadcastTo(np.broadcast_to),
        tracer.optimizer.classical.SkipConcatenate(np.concatenate),
        tracer.optimizer.InlineGraph(),
        tracer.optimizer.SkipCast(),
    ]

    import numpy as np

    def is_supported_tensor(tensor):
        return isinstance(tensor, np.ndarray)

    def get_shape(tensor):
        return tuple(int(x) for x in tensor.shape)

    return {"optimizations": optimizations, "compiler": tracer.compiler.python, "is_supported_tensor": is_supported_tensor, "get_shape": get_shape}


def adapt_numpylike_reduce(op):
    iskwarg = lambda name, iskwarg=_make_iskwarg(op): name != "axis" and iskwarg(name)
    np = tracer.signature.numpy()

    classical = adapter.classical_from_numpy.ops(np)

    op = tracer.signature.python.constant(op)
    op = adapter.decomposednamedtensor_from_classical.reduce(op, expected_type=np.ndarray)
    op = adapter.namedtensor_from_decomposednamedtensor.op(op, classical)
    op = adapter.namedtensor_calltensorfactory.op(op, expected_type=np.ndarray)
    op = adapter.einx_from_namedtensor.reduce(op, iskwarg=iskwarg)

    return api(op, backend=types.SimpleNamespace(**_get_backend_kwargs()))


adapt_numpylike_reduce.__doc__ = _make_doc_adapt_numpylike_reduce()


def adapt_numpylike_elementwise(op):
    iskwarg = _make_iskwarg(op)
    np = tracer.signature.numpy()

    classical = adapter.classical_from_numpy.ops(np)

    op = tracer.signature.python.constant(op)
    op = adapter.decomposednamedtensor_from_classical.elementwise(op, classical, expected_type=np.ndarray)
    op = adapter.namedtensor_from_decomposednamedtensor.op(op, classical)
    op = adapter.namedtensor_calltensorfactory.op(op, expected_type=np.ndarray)
    op = adapter.einx_from_namedtensor.elementwise(op, iskwarg=iskwarg)

    return api(op, backend=types.SimpleNamespace(**_get_backend_kwargs()))


adapt_numpylike_elementwise.__doc__ = _make_doc_adapt_numpylike_elementwise()


def _backend_creator(create):
    def new_create():
        numpy = tracer.signature.numpy()
        classical = adapter.classical_from_numpy.ops(numpy)

        decomposednamedtensor_ops, name = create(numpy, classical)

        namedtensor_ops = adapter.namedtensor_from_decomposednamedtensor.ops(decomposednamedtensor_ops, classical)
        namedtensor_ops = adapter.namedtensor_calltensorfactory.ops(namedtensor_ops, expected_type=numpy.ndarray)
        einx_ops = adapter.einx_from_namedtensor.ops(namedtensor_ops)

        return Backend(ops=einx_ops, name=name, priority=-5, **_get_backend_kwargs())

    return new_create


@_backend_creator
def create_backend_numpylike(numpy, classical):
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

    return decomposednamedtensor_ops, "numpy.numpylike"


registry.register_on_import("numpy", "numpy.numpylike", create_backend_numpylike)


@_backend_creator
def create_backend_einsum(numpy, classical):
    einsum = adapter.einsum_from_numpy(numpy)

    decomposednamedtensor_ops = {
        "id": adapter.decomposednamedtensor_from_einsum.id(einsum),
        "sum": adapter.decomposednamedtensor_from_einsum.sum(einsum),
        "multiply": adapter.decomposednamedtensor_from_einsum.multiply(einsum),
        "dot": adapter.decomposednamedtensor_from_einsum.dot(einsum),
    }

    return decomposednamedtensor_ops, "numpy.einsum"


registry.register_on_import("numpy", "numpy.einsum", create_backend_einsum)


def create_backend():
    einx_ops_numpylike = create_backend_numpylike().ops
    einx_ops_einsum = create_backend_einsum().ops
    einx_ops = einx_ops_numpylike | {"dot": einx_ops_einsum["dot"]}

    return Backend(ops=einx_ops, name="numpy", priority=-1, **_get_backend_kwargs())


registry.register_on_import("numpy", "numpy", create_backend)
