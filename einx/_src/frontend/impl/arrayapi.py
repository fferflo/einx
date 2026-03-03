from ..backend import registry
from ..backend import Backend
import einx._src.adapter as adapter
import einx._src.tracer as tracer
from ._util import _make_iskwarg
from ..api import api
import types
from functools import partial
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

    import array_api_compat
    import numpy as np

    def is_supported_tensor(tensor):
        try:
            array_api_compat.array_namespace(tensor)
            return True
        except:
            return False

    def get_shape(x):
        if isinstance(x, int | float | bool | np.integer | np.floating | np.bool_):
            return ()
        elif hasattr(x, "shape"):
            return tuple(int(x) for x in x.shape)
        else:
            raise TypeError(f"Cannot get shape of object of type {type(x)}")

    return {"optimizations": optimizations, "compiler": tracer.compiler.python, "is_supported_tensor": is_supported_tensor, "get_shape": get_shape}


def adapt_numpylike_reduce(op):
    iskwarg = lambda name, iskwarg=_make_iskwarg(op): name != "axis" and iskwarg(name)
    xp_stack = adapter.ArrayApiNamespaceStack()
    xp = tracer.signature.arrayapi(xp_stack.get_xp)

    classical = adapter.classical_from_arrayapi.ops(xp)

    op = tracer.signature.python.constant(op)
    op = adapter.decomposednamedtensor_from_classical.reduce(op, expected_type=partial(adapter.tensortype_from_arrayapi, xp))
    op = adapter.namedtensor_from_decomposednamedtensor.op(op, classical)
    op = adapter.namedtensor_calltensorfactory.op(op, context=xp_stack, expected_type=partial(adapter.tensortype_from_arrayapi, xp))
    op = adapter.einx_from_namedtensor.reduce(op, iskwarg=iskwarg)

    return api(op, backend=types.SimpleNamespace(**_get_backend_kwargs()))


adapt_numpylike_reduce.__doc__ = _make_doc_adapt_numpylike_reduce()


def adapt_numpylike_elementwise(op):
    iskwarg = _make_iskwarg(op)
    xp_stack = adapter.ArrayApiNamespaceStack()
    xp = tracer.signature.arrayapi(xp_stack.get_xp)

    classical = adapter.classical_from_arrayapi.ops(xp)

    op = tracer.signature.python.constant(op)
    op = adapter.decomposednamedtensor_from_classical.elementwise(op, classical, expected_type=partial(adapter.tensortype_from_arrayapi, xp))
    op = adapter.namedtensor_from_decomposednamedtensor.op(op, classical)
    op = adapter.namedtensor_calltensorfactory.op(op, context=xp_stack, expected_type=partial(adapter.tensortype_from_arrayapi, xp))
    op = adapter.einx_from_namedtensor.elementwise(op, iskwarg=iskwarg)

    return api(op, backend=types.SimpleNamespace(**_get_backend_kwargs()))


adapt_numpylike_elementwise.__doc__ = _make_doc_adapt_numpylike_elementwise()


def _backend_creator(create):
    def new_create():
        xp_stack = adapter.ArrayApiNamespaceStack()

        xp = tracer.signature.arrayapi(xp_stack.get_xp)

        classical = adapter.classical_from_arrayapi.ops(xp)

        decomposednamedtensor_ops, name = create(xp, classical)

        namedtensor_ops = adapter.namedtensor_from_decomposednamedtensor.ops(decomposednamedtensor_ops, classical)
        namedtensor_ops = adapter.namedtensor_calltensorfactory.ops(
            namedtensor_ops, context=xp_stack, expected_type=partial(adapter.tensortype_from_arrayapi, xp)
        )
        einx_ops = adapter.einx_from_namedtensor.ops(namedtensor_ops)

        return Backend(ops=einx_ops, name=name, priority=-15, **_get_backend_kwargs())

    return new_create


@_backend_creator
def create_backend_numpylike(xp, classical):
    einsum = adapter.einsum_from_arrayapi(xp)

    decomposednamedtensor_ops = (
        {name: adapter.decomposednamedtensor_from_classical.elementwise(getattr(classical, name), classical) for name in adapter.ops.elementwise}
        | {name: adapter.decomposednamedtensor_from_classical.reduce(getattr(classical, name)) for name in adapter.ops.reduce}
        | {name: adapter.decomposednamedtensor_from_classical.preserve_shape(getattr(classical, name)) for name in adapter.ops.preserve_shape}
        | {name: adapter.decomposednamedtensor_from_classical.argfind(getattr(classical, name), classical) for name in adapter.ops.argfind}
        | {name: adapter.decomposednamedtensor_from_classical.update_at_ravelled(getattr(classical, name), classical) for name in adapter.ops.update_at}
        | {"get_at": adapter.decomposednamedtensor_from_classical.get_at_ravelled(classical), "dot": adapter.decomposednamedtensor_from_einsum.dot(einsum)}
    )

    return decomposednamedtensor_ops, "arrayapi.numpylike"


registry.register_on_import("array_api_compat", "arrayapi.numpylike", create_backend_numpylike)


@_backend_creator
def create_backend_einsum(xp, classical):
    einsum = adapter.einsum_from_arrayapi(xp)

    decomposednamedtensor_ops = {
        "id": adapter.decomposednamedtensor_from_einsum.id(einsum),
        "sum": adapter.decomposednamedtensor_from_einsum.sum(einsum),
        "multiply": adapter.decomposednamedtensor_from_einsum.multiply(einsum),
        "dot": adapter.decomposednamedtensor_from_einsum.dot(einsum),
    }

    return decomposednamedtensor_ops, "arrayapi.einsum"


registry.register_on_import("array_api_compat", "arrayapi.einsum", create_backend_einsum)


def create_backend():
    einx_ops_numpylike = create_backend_numpylike().ops
    einx_ops_einsum = create_backend_einsum().ops
    einx_ops = einx_ops_numpylike | {"dot": einx_ops_einsum["dot"]}

    return Backend(ops=einx_ops, name="arrayapi", priority=-2, **_get_backend_kwargs())


registry.register_on_import("array_api_compat", "arrayapi", create_backend)
