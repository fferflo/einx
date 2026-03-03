import einx._src.tracer as tracer
import einx._src.adapter as adapter
from ..types import Tensor
from ..backend import registry
from ..backend import Backend
from ._util import _make_iskwarg, _unsupported_op
from ..api import api
import types
from ._docs import _make_doc_adapt_numpylike_reduce
from ._docs import _make_doc_adapt_numpylike_elementwise
from ._docs import _make_doc_adapt_with_vmap


def _get_backend_kwargs():
    mx = tracer.signature.python.import_("mlx.core", as_="mx")
    optimizations = [
        tracer.optimizer.classical.SkipReshape(mx.reshape),
        tracer.optimizer.classical.SkipTranspose(mx.transpose),
        tracer.optimizer.classical.SkipBroadcastTo(mx.broadcast_to),
        tracer.optimizer.classical.SkipConcatenate(mx.concatenate),
        tracer.optimizer.InlineGraph(),
        tracer.optimizer.SkipCast(),
    ]

    import mlx.core as mx

    def is_supported_tensor(tensor):
        return isinstance(tensor, mx.array)

    def get_shape(tensor):
        return tuple(int(x) for x in tensor.shape)

    return {"optimizations": optimizations, "compiler": tracer.compiler.python, "is_supported_tensor": is_supported_tensor, "get_shape": get_shape}


def adapt_with_vmap(op, signature=None):
    iskwarg = _make_iskwarg(op)
    mlx = tracer.signature.mlx()

    classical = adapter.classical_from_mlx.ops(mlx)
    vmap = adapter.vmap_from_mlx(mlx)

    op = tracer.signature.python.constant(op)
    op = adapter.decomposednamedtensor_from_vmap.op(op, vmap, expected_type=mlx.core.array)
    op = adapter.namedtensor_from_decomposednamedtensor.op(op, classical)
    op = adapter.namedtensor_calltensorfactory.op(op, expected_type=mlx.core.array)
    op = adapter.einx_from_namedtensor.op(op, iskwarg=iskwarg, el_op=signature, implicit_output="bijective")

    return api(op, backend=types.SimpleNamespace(**_get_backend_kwargs()))


adapt_with_vmap.__doc__ = _make_doc_adapt_with_vmap("mlx", "``mlx.core.vmap``")


def adapt_numpylike_reduce(op):
    iskwarg = lambda name, iskwarg=_make_iskwarg(op): name != "axis" and iskwarg(name)
    mlx = tracer.signature.mlx()

    classical = adapter.classical_from_mlx.ops(mlx)

    op = tracer.signature.python.constant(op)
    op = adapter.decomposednamedtensor_from_classical.reduce(op, expected_type=mlx.core.array)
    op = adapter.namedtensor_from_decomposednamedtensor.op(op, classical)
    op = adapter.namedtensor_calltensorfactory.op(op, expected_type=mlx.core.array)
    op = adapter.einx_from_namedtensor.reduce(op, iskwarg=iskwarg)

    return api(op, backend=types.SimpleNamespace(**_get_backend_kwargs()))


adapt_numpylike_reduce.__doc__ = _make_doc_adapt_numpylike_reduce()


def adapt_numpylike_elementwise(op):
    iskwarg = _make_iskwarg(op)
    mlx = tracer.signature.mlx()

    classical = adapter.classical_from_mlx.ops(mlx)

    op = tracer.signature.python.constant(op)
    op = adapter.decomposednamedtensor_from_classical.elementwise(op, classical, expected_type=mlx.core.array)
    op = adapter.namedtensor_from_decomposednamedtensor.op(op, classical)
    op = adapter.namedtensor_calltensorfactory.op(op, expected_type=mlx.core.array)
    op = adapter.einx_from_namedtensor.elementwise(op, iskwarg=iskwarg)

    return api(op, backend=types.SimpleNamespace(**_get_backend_kwargs()))


adapt_numpylike_elementwise.__doc__ = _make_doc_adapt_numpylike_elementwise()


def _backend_creator(create):
    def new_create():
        mlx = tracer.signature.mlx()

        classical = adapter.classical_from_mlx.ops(mlx)

        decomposednamedtensor_ops, name = create(mlx, classical)

        namedtensor_ops = adapter.namedtensor_from_decomposednamedtensor.ops(decomposednamedtensor_ops, classical)
        namedtensor_ops = adapter.namedtensor_calltensorfactory.ops(namedtensor_ops, expected_type=mlx.core.array)
        einx_ops = adapter.einx_from_namedtensor.ops(namedtensor_ops)

        return Backend(ops=einx_ops, name=name, priority=-5, **_get_backend_kwargs())

    return new_create


@_backend_creator
def create_backend_numpylike(mlx, classical):
    einsum = adapter.einsum_from_mlx(mlx)

    decomposednamedtensor_ops = (
        {name: adapter.decomposednamedtensor_from_classical.elementwise(getattr(classical, name), classical) for name in adapter.ops.elementwise}
        | {name: adapter.decomposednamedtensor_from_classical.reduce(getattr(classical, name)) for name in adapter.ops.reduce}
        | {name: adapter.decomposednamedtensor_from_classical.preserve_shape(getattr(classical, name)) for name in adapter.ops.preserve_shape}
        | {name: adapter.decomposednamedtensor_from_classical.argfind(getattr(classical, name), classical) for name in adapter.ops.argfind}
        | {name: adapter.decomposednamedtensor_from_classical.update_at_ravelled(getattr(classical, name), classical) for name in adapter.ops.update_at}
        | {"get_at": adapter.decomposednamedtensor_from_classical.get_at_ravelled(classical), "dot": adapter.decomposednamedtensor_from_einsum.dot(einsum)}
    )

    return decomposednamedtensor_ops, "mlx.numpylike"


registry.register_on_import("mlx", "mlx.numpylike", create_backend_numpylike)


@_backend_creator
def create_backend_einsum(mlx, classical):
    einsum = adapter.einsum_from_mlx(mlx)

    decomposednamedtensor_ops = {
        "id": adapter.decomposednamedtensor_from_einsum.id(einsum),
        "sum": adapter.decomposednamedtensor_from_einsum.sum(einsum),
        "multiply": adapter.decomposednamedtensor_from_einsum.multiply(einsum),
        "dot": adapter.decomposednamedtensor_from_einsum.dot(einsum),
    }

    return decomposednamedtensor_ops, "mlx.einsum"


registry.register_on_import("mlx", "mlx.einsum", create_backend_einsum)


@_backend_creator
def create_backend_vmap(mlx, classical):
    vmap = adapter.vmap_from_mlx(mlx)
    elementary_ops = adapter.elementary_from_classical.ops(classical)

    decomposednamedtensor_ops = {
        name: adapter.decomposednamedtensor_from_vmap.op(
            elementary_ops[name], vmap, expected_type=mlx.core.array, allow_squeeze_unsqueeze=True, classical=classical
        )
        for name in adapter.ops.all
    } | {name: _unsupported_op(name, "mlx.vmap") for name in adapter.ops.update_at}

    return decomposednamedtensor_ops, "mlx.vmap"


registry.register_on_import("mlx", "mlx.vmap", create_backend_vmap)


def create_backend():
    einx_ops_numpylike = create_backend_numpylike().ops
    einx_ops_einsum = create_backend_einsum().ops
    einx_ops = einx_ops_numpylike | {"dot": einx_ops_einsum["dot"]}

    return Backend(ops=einx_ops, name="mlx", priority=0, **_get_backend_kwargs())


registry.register_on_import("mlx", "mlx", create_backend)
