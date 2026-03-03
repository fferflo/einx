import numpy as np
import einx._src.namedtensor.stage3 as stage3
from einx._src.namedtensor import NamedTensor
import functools
from einx._src.adapter._util import _squeeze_transpose_broadcast
from einx._src.util.functools import use_name_of
from einx._src.frontend.errors import SemanticError

py_id = id


class Decomposer:
    def __init__(self, classical):
        self.classical = classical

    def __call__(self, func):
        @use_name_of(func)
        def inner(tensors, exprs_out):
            exprs_in = [tensors.expr for tensors in tensors]
            tensors = [tensors.value for tensors in tensors]

            # Decompose
            exprs_in, tensors = self._decompose(exprs_in, tensors)
            exprs_out_flat, _ = self._decompose(exprs_out, None)

            # Remove unitary vectorized axes from input
            exprs_in2 = []
            tensors2 = []

            def is_squeezable_axis(expr):
                return isinstance(expr, stage3.Axis) and not stage3.is_in_brackets(expr) and expr.value == 1

            for expr_in, tensor in zip(exprs_in, tensors, strict=False):
                expr_in = stage3.remove(expr_in, is_squeezable_axis, keep_children=False)
                tensor = self.classical.reshape(tensor, expr_in.shape)
                exprs_in2.append(expr_in)
                tensors2.append(tensor)
            exprs_in = exprs_in2
            tensors = tensors2

            # Remove broadcast axes from output
            in_axis_names = {axis.name for expr in exprs_in for axis in expr if isinstance(axis, stage3.Axis)}

            def is_broadcast_axis(expr):
                return isinstance(expr, stage3.Axis) and expr.name not in in_axis_names and not stage3.is_in_brackets(expr)

            exprs_out_flat_without_broadcast = [stage3.remove(expr, is_broadcast_axis, keep_children=False) for expr in exprs_out_flat]

            # Call inner function
            tensors = [NamedTensor(tensor, expr) for tensor, expr in zip(tensors, exprs_in, strict=False)]
            tensors = func(tensors, exprs_out_flat_without_broadcast)
            exprs_in = [t.expr for t in tensors]
            tensors = [t.value for t in tensors]

            # Transpose and broadcast to output shape
            x = [
                _squeeze_transpose_broadcast(self.classical, expr_in, tensor, expr_out_flat)
                for expr_in, tensor, expr_out_flat in zip(exprs_in, tensors, exprs_out_flat, strict=False)
            ]
            tensors = [x[1] for x in x]
            exprs_in = [x[0] for x in x]

            # Compose
            tensors = self._compose(exprs_in, tensors, exprs_out)

            return [NamedTensor(tensor, expr) for tensor, expr in zip(tensors, exprs_out, strict=False)]

        return inner

    def _decompose_single(self, expr, tensor=None):
        def readd_brackets(a):
            if stage3.is_in_brackets(a):
                return stage3.Brackets.create(a.__deepcopy__())
            else:
                return a.__deepcopy__()

        # Decompose flattened axes
        if any(isinstance(e, stage3.FlattenedAxis) for e in expr):

            def unflatten(e):
                if isinstance(e, stage3.FlattenedAxis):
                    return readd_brackets(e.inner)
                else:
                    return readd_brackets(e)

            expr = stage3.List.create([unflatten(e) for e in expr])

            if tensor is not None:
                tensor = self.classical.reshape(tensor, expr.shape)

            return self._decompose_single(expr, tensor)

        # Decompose repeated axes (not in brackets)
        axis_counts = {}
        for axis in expr:
            if isinstance(axis, stage3.Axis) and not stage3.is_in_brackets(axis):
                axis_counts[axis.name] = axis_counts.get(axis.name, 0) + 1
        if any(count > 1 for count in axis_counts.values()):
            for axis_name, count in axis_counts.items():
                if count > 1:
                    indices_in = [i for i, a in enumerate(expr) if isinstance(a, stage3.Axis) and a.name == axis_name]
                    index_out = indices_in[0]

                    expr = stage3.List.create([
                        readd_brackets(a) for i, a in enumerate(expr) if not (isinstance(a, stage3.Axis) and a.name == axis_name and i != indices_in[0])
                    ])
                    if tensor is not None:
                        tensor = self.classical.diagonal(tensor, axes_in=indices_in, axis_out=index_out)

            return self._decompose_single(expr, tensor)

        # Decompose concatenated axes
        if any(isinstance(e, stage3.ConcatenatedAxis) for e in expr):
            concat_index, concat_expr = [(i, e) for i, e in enumerate(expr) if isinstance(e, stage3.ConcatenatedAxis)][0]
            splits = np.cumsum([c.shape[0] for c in concat_expr.children])[:-1]
            splits = splits.tolist()

            if tensor is not None:
                subtensors = self.classical.split(tensor, splits, axis=concat_index)
            else:
                subtensors = [None for _ in concat_expr.children]

            subexprs = []
            for i in range(len(concat_expr.children)):
                subexpr = stage3.map(expr, lambda expr: expr.children[i].__deepcopy__() if py_id(expr) == py_id(concat_expr) else None, include_children=False)
                subexprs.append(subexpr)

            return self._decompose(subexprs, subtensors)

        # No more decomposition possible
        return [expr], [tensor]

    def _decompose(self, exprs, tensors=None):
        if tensors is None:
            tensors = [None for _ in exprs]

        tensors_out = []
        exprs_out = []
        for expr, tensor in zip(exprs, tensors, strict=False):
            subexprs, subtensors = self._decompose_single(expr, tensor)
            exprs_out.extend(subexprs)
            tensors_out.extend(subtensors)

        return exprs_out, tensors_out

    def _compose_next(self, exprs_in, tensors_in, expr_out):
        def unflatten(e):
            if isinstance(e, stage3.FlattenedAxis):
                return e.inner.__deepcopy__()
            else:
                return e.__deepcopy__()

        expr_out_flat = stage3.List.create([unflatten(e) for e in expr_out])

        if any(isinstance(e, stage3.ConcatenatedAxis) for e in expr_out_flat):
            concat_index, concat_expr = [(i, e) for i, e in enumerate(expr_out_flat) if isinstance(e, stage3.ConcatenatedAxis)][0]

            tensors_out = []
            for i in range(len(concat_expr.children)):
                # Extract subexpression of i-th child in concatenation
                subexpr = stage3.map(
                    expr_out_flat, lambda expr: expr.children[i].__deepcopy__() if py_id(expr) == py_id(concat_expr) else None, include_children=False
                )

                # Get subtensor
                subtensor = self._compose_next(exprs_in, tensors_in, subexpr)

                tensors_out.append(subtensor)

            tensor_out = self.classical.concatenate(tensors_out, axis=concat_index)
        else:
            _ = next(exprs_in)  # next_expr_in
            tensor_out = next(tensors_in)

        tensor_out = self.classical.reshape(tensor_out, expr_out.shape)

        return tensor_out

    def _compose(self, exprs_in, tensors_in, exprs_out):
        iter_exprs_in = iter(exprs_in)
        iter_tensors_in = iter(tensors_in)
        tensors_out = []
        for expr_out in exprs_out:
            t = self._compose_next(iter_exprs_in, iter_tensors_in, expr_out)
            tensors_out.append(t)

        return tensors_out


def op(op, classical):
    decomposer = Decomposer(classical)

    @use_name_of(op)
    def inner(*tensors, out, **kwargs):
        if not isinstance(out, list | tuple):
            out = [out]

        @decomposer
        def inner(tensors, out):
            if not isinstance(out, list | tuple):
                out = [out]
            result = op(*tensors, out=out if len(out) > 1 else out[0], **kwargs)
            if len(out) == 1:
                result = [result]  # Must return a list
            return result

        tensors = inner(tensors, out)
        if len(out) == 1:
            return tensors[0]
        else:
            return tensors

    return inner


def _matchable(expr_in, expr_out):
    axes_in = {axis.name for axis in expr_in if isinstance(axis, stage3.Axis) and axis.value != 1}
    axes_out = {axis.name for axis in expr_out if isinstance(axis, stage3.Axis) and axis.value != 1}
    return axes_in.issubset(axes_out)


def _to_ord_str(idx):
    if idx == 0:
        return "1st"
    elif idx == 1:
        return "2nd"
    elif idx == 2:
        return "3rd"
    else:
        return f"{idx + 1}th"


def id(id, classical):
    if id is None:

        def id_inner(*xs, out):
            return xs if len(xs) > 1 else xs[0]

    else:
        id_inner = id

    def id(*tensors, out):
        exprs_in = [t.expr for t in tensors]
        exprs_out = out if isinstance(out, list | tuple) else [out]
        if len(exprs_in) != len(exprs_out):
            inputs = "input" if len(exprs_in) == 1 else "inputs"
            outputs = "output" if len(exprs_out) == 1 else "outputs"
            raise SemanticError(
                message=(
                    f"The number of input and output expressions (after decomposition of axis concatenations) must be the same, "
                    f"but got {len(exprs_in)} {inputs} and {len(exprs_out)} {outputs}.\n%EXPR%"
                ),
            )
        for i, (expr_in, expr_out) in enumerate(zip(exprs_in, exprs_out, strict=False)):
            if not _matchable(expr_in, expr_out):
                raise SemanticError(
                    message=(
                        f'The {_to_ord_str(i)} input expression "{expr_in}" is not compatible with the {_to_ord_str(i)} output '
                        f'expression "{expr_out}" (after decomposition of axis concatenations).\n%EXPR%'
                    ),
                )
        return id_inner(*tensors, out=out)

    return op(id, classical)


def ops(decomposednamedtensor_ops, classical):
    ops = {name: (id if name == "id" else globals()["op"])(op, classical) for name, op in decomposednamedtensor_ops.items()}
    if "id" not in ops:
        ops["id"] = id(None, classical)
    return ops
