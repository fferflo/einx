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
from ._util import _make_iskwarg, _unsupported_op
from ._docs import _make_doc_adapt_numpylike_reduce
from ._docs import _make_doc_adapt_numpylike_elementwise
from ._docs import _make_doc_adapt_with_vmap


def _get_backend_kwargs():
    jnp = tracer.signature.python.import_("jax.numpy", as_="jnp")
    optimizations = [
        tracer.optimizer.classical.SkipReshape(jnp.reshape),
        tracer.optimizer.classical.SkipTranspose(jnp.transpose),
        tracer.optimizer.classical.SkipBroadcastTo(jnp.broadcast_to),
        tracer.optimizer.classical.SkipConcatenate(jnp.concatenate),
        tracer.optimizer.InlineGraph(),
        tracer.optimizer.SkipCast(),
    ]

    import jax
    import jax.numpy as jnp

    try:
        tensor_types = (jax.Array, jax.core.Tracer)
    except:
        tensor_types = jnp.ndarray

    def is_supported_tensor(tensor):
        return isinstance(tensor, tensor_types)

    def get_shape(tensor):
        return tuple(int(x) for x in tensor.shape)

    return {"optimizations": optimizations, "compiler": tracer.compiler.python, "is_supported_tensor": is_supported_tensor, "get_shape": get_shape}


def adapt_with_vmap(op, signature=None):
    iskwarg = _make_iskwarg(op)
    jax = tracer.signature.jax()

    classical = adapter.classical_from_jax.ops(jax)
    vmap = adapter.vmap_from_jax(jax)

    op = tracer.signature.python.constant(op)
    op = adapter.decomposednamedtensor_from_vmap.op(op, vmap, expected_type=jax.numpy.ndarray)
    op = adapter.namedtensor_from_decomposednamedtensor.op(op, classical)
    op = adapter.namedtensor_calltensorfactory.op(op, expected_type=jax.numpy.ndarray)
    op = adapter.einx_from_namedtensor.op(op, iskwarg=iskwarg, el_op=signature, implicit_output="bijective")

    return api(op, backend=types.SimpleNamespace(**_get_backend_kwargs()))


adapt_with_vmap.__doc__ = _make_doc_adapt_with_vmap("jax", "``jax.vmap``")


def adapt_numpylike_reduce(op):
    iskwarg = lambda name, iskwarg=_make_iskwarg(op): name != "axis" and iskwarg(name)
    jax = tracer.signature.jax()

    classical = adapter.classical_from_jax.ops(jax)

    op = tracer.signature.python.constant(op)
    op = adapter.decomposednamedtensor_from_classical.reduce(op, expected_type=jax.numpy.ndarray)
    op = adapter.namedtensor_from_decomposednamedtensor.op(op, classical)
    op = adapter.namedtensor_calltensorfactory.op(op, expected_type=jax.numpy.ndarray)
    op = adapter.einx_from_namedtensor.reduce(op, iskwarg=iskwarg)

    return api(op, backend=types.SimpleNamespace(**_get_backend_kwargs()))


adapt_numpylike_reduce.__doc__ = _make_doc_adapt_numpylike_reduce()


def adapt_numpylike_elementwise(op):
    iskwarg = _make_iskwarg(op)
    jax = tracer.signature.jax()

    classical = adapter.classical_from_jax.ops(jax)

    op = tracer.signature.python.constant(op)
    op = adapter.decomposednamedtensor_from_classical.elementwise(op, classical, expected_type=jax.numpy.ndarray)
    op = adapter.namedtensor_from_decomposednamedtensor.op(op, classical)
    op = adapter.namedtensor_calltensorfactory.op(op, expected_type=jax.numpy.ndarray)
    op = adapter.einx_from_namedtensor.elementwise(op, iskwarg=iskwarg)

    return api(op, backend=types.SimpleNamespace(**_get_backend_kwargs()))


adapt_numpylike_elementwise.__doc__ = _make_doc_adapt_numpylike_elementwise()


def _backend_creator(create):
    def new_create():
        jax = tracer.signature.jax()
        classical = adapter.classical_from_jax.ops(jax)

        decomposednamedtensor_ops, name = create(jax, classical)

        namedtensor_ops = adapter.namedtensor_from_decomposednamedtensor.ops(decomposednamedtensor_ops, classical)
        namedtensor_ops = adapter.namedtensor_calltensorfactory.ops(namedtensor_ops, expected_type=jax.numpy.ndarray)
        einx_ops = adapter.einx_from_namedtensor.ops(namedtensor_ops)

        return Backend(ops=einx_ops, name=name, priority=-5, **_get_backend_kwargs())

    return new_create


@_backend_creator
def create_backend_numpylike(jax, classical):
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

    return decomposednamedtensor_ops, "jax.numpylike"


registry.register_on_import("jax", "jax.numpylike", create_backend_numpylike)


@_backend_creator
def create_backend_einsum(jax, classical):
    einsum = adapter.einsum_from_jax(jax)

    decomposednamedtensor_ops = {
        "id": adapter.decomposednamedtensor_from_einsum.id(einsum),
        "sum": adapter.decomposednamedtensor_from_einsum.sum(einsum),
        "multiply": adapter.decomposednamedtensor_from_einsum.multiply(einsum),
        "dot": adapter.decomposednamedtensor_from_einsum.dot(einsum),
    }

    return decomposednamedtensor_ops, "jax.einsum"


registry.register_on_import("jax", "jax.einsum", create_backend_einsum)


@_backend_creator
def create_backend_vmap(jax, classical):
    vmap = adapter.vmap_from_jax(jax)
    elementary_ops = adapter.elementary_from_classical.ops(classical)

    decomposednamedtensor_ops = {
        name: adapter.decomposednamedtensor_from_vmap.op(
            elementary_ops[name], vmap, expected_type=jax.numpy.ndarray, allow_squeeze_unsqueeze=True, classical=classical
        )
        for name in adapter.ops.all
    } | {name: _unsupported_op(name, "jax.vmap") for name in adapter.ops.update_at}

    return decomposednamedtensor_ops, "jax.vmap"


registry.register_on_import("jax", "jax.vmap", create_backend_vmap)


def create_backend():
    einx_ops_numpylike = create_backend_numpylike().ops
    einx_ops_einsum = create_backend_einsum().ops
    einx_ops = einx_ops_numpylike | {"dot": einx_ops_einsum["dot"]}

    return Backend(ops=einx_ops, name="jax", priority=0, **_get_backend_kwargs())


registry.register_on_import("jax", "jax", create_backend)
