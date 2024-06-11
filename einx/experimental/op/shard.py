import einx
import einx.op.util as util
import numpy as np
from functools import partial
from typing import Callable, Union, Any
import numpy.typing as npt

tP = einx.tracer.import_("PartitionSpec", "P", from_="jax.sharding")
tNamedSharding = einx.tracer.import_("NamedSharding", from_="jax.sharding")
tMesh = einx.tracer.import_("Mesh", from_="jax.sharding")
tjax = einx.tracer.import_("jax")
tnp = einx.tracer.import_("numpy", as_="np")


def _is_composed(expr):
    node = expr
    while node is not None:
        if isinstance(node, einx.expr.stage3.Composition):
            return True
        node = node.parent
    return False


@einx.jit(
    trace=lambda t, c: lambda expr_in, tensor_in, expr_out, backend=None: c(
        expr_in,
        t(tensor_in),
        expr_out,
    )
)
def shard_stage3(expr_in, tensor_in, expr_out, mesh=None, backend=None):
    import jax

    for root in [expr_in, expr_out]:
        for expr in root.all():
            if isinstance(expr, einx.expr.stage3.Concatenation):
                raise ValueError("Concatenation not allowed")
            if isinstance(expr, einx.expr.stage3.Marker):
                child = expr
                while child.parent is not None:
                    if (
                        isinstance(child.parent, einx.expr.stage3.List)
                        and _is_composed(child.parent)
                        and child is not child.parent.children[0]
                    ):
                        raise ValueError(
                            "If device axes are used within a composition they "
                            "must appear as the left-most member of the composition"
                        )
                    child = child.parent

    # Call tensor factories
    tensor_in = einx.tracer.call_factory(tensor_in, expr_in.shape, backend=backend)
    (tensor_in,) = backend.all_to_tensor([tensor_in])

    # Flatten expressions
    (expr_in,), (tensor_in,) = util.flatten([expr_in], [tensor_in], backend=backend)
    marked_axes = tuple(
        axis
        for axis in expr_in
        if isinstance(axis, einx.expr.stage3.Axis) and einx.expr.stage3.is_marked(axis)
    )

    if mesh is None:
        # Construct new mesh
        devices = tnp.array(tjax.devices()).reshape(tuple(a.value for a in marked_axes))
        mesh = tMesh(devices, axis_names=tuple(a.name for a in marked_axes))
    elif isinstance(mesh, jax.sharding.Mesh):
        # Got mesh -> check that marked axes match mesh
        marked_names = set(a.name for a in marked_axes)
        mesh_names = set(str(a) for a in mesh.axis_names)
        if not marked_names.issubset(mesh_names):
            raise ValueError(
                f"Marked axes must be subset of mesh axes. Got marked axes {marked_names} and mesh axes {mesh_names}"
            )
    else:
        # Got list of devices -> construct new mesh
        devices = tnp.array(mesh).reshape(tuple(a.value for a in marked_axes))
        mesh = tMesh(devices, axis_names=tuple(a.name for a in marked_axes))

    # Construct partition spec
    axes = tuple(axis for axis in expr_in if isinstance(axis, einx.expr.stage3.Axis))
    partition_spec = [axis.name if einx.expr.stage3.is_marked(axis) else None for axis in axes]
    partition_spec = tP(*partition_spec)

    # Shard tensor
    sharding = tNamedSharding(mesh, partition_spec)
    tensor_in = tjax.device_put(tensor_in, sharding)

    # Unflatten output expressions
    (tensor_in,) = util.unflatten([expr_in], [tensor_in], [expr_out], backend=backend)

    return tensor_in, expr_in


@einx.lru_cache
def parse(description, tensor_shape, cse=True, mesh=None, jax_devices=None, **parameters):
    import jax

    description, parameters = einx.op.util._clean_description_and_parameters(
        description, parameters
    )

    op = einx.expr.stage1.parse_op(description)

    if len(op) != 1:
        raise ValueError(f"Expected exactly one expression, got {len(op)}")

    def solve(eqs):
        return einx.expr.solve(
            [einx.expr.Equation(op[0][0], tensor_shape)]
            + eqs
            + [
                einx.expr.Equation(k, np.asarray(v)[..., np.newaxis], depth1=None, depth2=None)
                for k, v in parameters.items()
            ],
            cse=cse,
        )[0]

    if mesh is None:
        # If no mesh is given, create new mesh of all devices
        try:
            expr_in = solve([])
        except einx.expr.SolveException as e:
            # Try with additional constraint of total number of devices
            expr_mesh = einx.expr.stage1.Composition(einx.expr.stage1.get_marked(op[0][0]))
            mesh_eq = einx.expr.Equation(expr_mesh, [len(jax.devices())])
            try:
                expr_in = solve([mesh_eq])
            except einx.expr.SolveException:
                # If it still fails, reraise original exception
                raise e
    elif isinstance(mesh, jax.sharding.Mesh):
        # Add constraints for existing mesh axes
        expr_mesh = einx.expr.stage1.Marker(
            einx.expr.stage1.List.maybe([
                einx.expr.stage1.NamedAxis(name) for name in mesh.axis_names
            ])
        )
        mesh_eq = einx.expr.Equation(expr_mesh, mesh.devices.shape)

        expr_in = solve([mesh_eq])
    elif isinstance(mesh, (list, tuple)):
        # Add constraint for number of devices
        expr_mesh = einx.expr.stage1.Composition(einx.expr.stage1.get_marked(op[0][0]))
        mesh_eq = einx.expr.Equation(expr_mesh, [len(mesh)])
        expr_in = solve([mesh_eq])

    expr_out = expr_in.__deepcopy__()

    return expr_in, expr_out


@einx.traceback_util.filter
@einx.jit(
    trace=lambda t, c: lambda description, tensor, mesh=None, backend=None, **kwargs: c(
        description, t(tensor), mesh=mesh, **kwargs
    )
)
def shard(
    description: str,
    tensor: einx.Tensor,
    mesh: Any = None,
    backend: Union[einx.Backend, str, None] = "jax",
    cse: bool = True,
    **parameters: npt.ArrayLike,
) -> einx.Tensor:
    """Shards a tensor over a mesh of devices.

    *This function is currently experimental and will likely change in future versions.*

    *This function is currently only supported for Jax: A sharding is created
    based on the given expression, and applied to the tensor using* ``jax.device_put``.

    The tensor is sharded across the marked axes in the input expression. The marked axes
    match the axis names and shape of the mesh:

    >>> x = jnp.ones((2, 4, 128))
    >>> x = einx.experimental.shard("[d1 d2] c")
    >>> x.sharding
    NamedSharding(mesh=Mesh('d1': 2, 'd2': 4), spec=PartitionSpec('d1', 'd2', None))

    Axis compositions can be used to apply the
    `sharding rules of Jax <https://jax.readthedocs.io/en/latest/notebooks/Distributed_arrays_and_automatic_parallelization.html>`_,
    where tensor axes are evenly divided by the number of shards:

    >>> x = jnp.ones((128, 640, 480, 3))
    >>> x = einx.experimental.shard("([batch] _) ...", x)
    >>> x.sharding
    NamedSharding(mesh=Mesh('batch': 8), spec=PartitionSpec('batch',))

    If possible, the sharding is created over all devices. ``_`` is a regular axis name,
    and its value is determined by :doc:`einx's expression solver </faq/solver>`.

    Optionally, an existing mesh can be passed:

    >>> from jax.sharding import Mesh
    >>> devices = np.asarray(jax.devices()).reshape(4, 2)
    >>> mesh = Mesh(devices, axis_names=("d1", "d2"))
    >>> x = jnp.ones((4, 1024, 1024))
    >>> x = einx.experimental.shard("a ([d2] b) ([d1] c)", x, mesh=mesh)
    >>> x.sharding
    NamedSharding(mesh=Mesh('d1': 4, 'd2': 2), spec=PartitionSpec(None, 'd2', 'd1'))

    The array is replicated over all mesh axes that are not part of the expression:

    >>> x = jnp.ones((1024, 1024))
    >>> x = einx.experimental.shard("a ([d1] b)", x, mesh=mesh)
    >>> x.sharding
    NamedSharding(mesh=Mesh('d1': 4, 'd2': 2), spec=PartitionSpec(None, 'd1',))

    Args:
        description: Description string in Einstein notation (see above).
        tensor: Input tensor or tensor factory matching the description string.
        mesh: Mesh or list of devices to shard the tensor over. If not given, a new mesh over all
            available devices will be created matching the axes in the given expression.
            Defaults to ``None``.
        cse: Whether to apply common subexpression elimination to the expressions. Defaults
            to True.
        graph: Whether to return the graph representation of the operation instead of
            computing the result. Defaults to False.
        **parameters: Additional parameters that specify values for single axes, e.g. ``a=4``.

    Returns:
        The sharded tensor if ``graph=False``, otherwise the graph
        representation of the operation.
    """
    if backend.name != "jax":
        raise NotImplementedError("einx.experimental.shard is currently only supported for Jax")
    expr_in, expr_out = parse(
        description, einx.tracer.get_shape(tensor), mesh=mesh, cse=cse, **parameters
    )
    tensor, expr_out = shard_stage3(expr_in, tensor, expr_out, mesh=mesh, backend=backend)
    return tensor


shard.parse = parse
