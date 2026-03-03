from . import stage1
from . import stage2
from . import stage3
import numpy as np


def _idx_to_ordinal(idx, n):
    if n <= 0:
        return ""
    if idx % 100 == 0:
        return "1st "
    elif idx % 100 == 1:
        return "2nd "
    elif idx % 100 == 2:
        return "3rd "
    else:
        return f"{idx + 1}th "


def _arr_to_str(l):
    if l is None:
        return "None"
    elif isinstance(l, np.ndarray):
        if l.ndim == 0:
            return str(l.tolist())
        else:
            return str(tuple(l.tolist()))
    elif isinstance(l, list):
        return str(tuple(l))
    else:
        return str(l)


def solve(
    exprs_in, exprs_out, tensor_shapes, invocation, parameters=None, cse=True, cse_concat=False, cse_in_brackets=False, verbose=False, equations_stage3=None
):
    if parameters is None:
        parameters = {}
    if len(exprs_in) != len(tensor_shapes):
        raise ValueError(f"The number of input expressions (found {len(exprs_in)}) must match the number of input tensors (found {len(tensor_shapes)})")

    for k, v in parameters.items():
        if not isinstance(k, str):
            raise ValueError(f"Axis names passed as constraints must be strings, but found value {k} of type {type(k)}")
        try:
            x = np.asarray(v)
        except Exception as e:
            raise ValueError(f"Values passed for axis names as constraints must be convertible to numpy arrays, but found value {v} of type {type(v)}") from e
        if not (isinstance(v, tuple) and v == ()) and not (isinstance(v, list) and v == []) and not np.issubdtype(x.dtype, np.integer):
            raise ValueError(
                f"Values passed for axis names as constraints must have an integral type, but found value {v} with type {type(v)} and dtype {x.dtype}"
            )

    # Remove unused constraints
    used_axisnames = {expr.name for expr in list(exprs_in) + list(exprs_out) for expr in expr.nodes() if isinstance(expr, stage1.Axis)}
    parameters = {k: v for k, v in parameters.items() if k in used_axisnames}

    if verbose:
        print("Stage1:")
        for expr_in, tensor_shape in zip(exprs_in, tensor_shapes, strict=False):
            print(f"  IN  {expr_in} = {_arr_to_str(tensor_shape)}")
        for expr_out in exprs_out:
            print(f"  OUT {expr_out}")

    # Transform to stage2
    equations = (
        [
            stage2.Equation(
                expr,
                tensor_shape,
                desc1=f'{_idx_to_ordinal(i, len(exprs_in))}input expression ("{expr}")',
                desc2=f"{_idx_to_ordinal(i, len(exprs_in))}input tensor (with shape ({', '.join([str(x) for x in tensor_shape])}))"
                if tensor_shape is not None
                else None,
            )
            for i, (expr, tensor_shape) in enumerate(zip(exprs_in, tensor_shapes, strict=False))
        ]
        + [stage2.Equation(expr, desc1=f'{_idx_to_ordinal(i, len(exprs_out))}output expression ("{expr}")') for i, expr in enumerate(exprs_out)]
        + [
            stage2.Equation(
                k,
                np.asarray(v)[..., np.newaxis].astype("int32"),
                depth1=None,
                depth2=None,
                desc1=f"axis {k}",
                desc2=f"constraint ({_arr_to_str(np.asarray(v))})",
            )
            for k, v in parameters.items()
        ]
    )

    exprs1, exprs2 = stage2.solve(equations, invocation=invocation)

    if verbose:
        print("Stage2:")
        for expr1, expr2 in zip(exprs1, exprs2, strict=False):
            print(f"  {expr1} = {expr2}")

    # Apply common-subexpression-elimination
    if cse:
        exprs = stage2.cse(exprs1 + exprs2, cse_concat=cse_concat, cse_in_brackets=cse_in_brackets)
        exprs1, exprs2 = exprs[: len(exprs1)], exprs[len(exprs1) :]

        if verbose:
            print("Stage2 (after CSE):")
            for expr1, expr2 in zip(exprs1, exprs2, strict=False):
                print(f"  {expr1} = {expr2}")

    # Transform to stage3
    equations = [stage3.Equation(expr1, expr2, eq.desc1, eq.desc2) for eq, expr1, expr2 in zip(equations, exprs1, exprs2, strict=False)]

    if equations_stage3 is not None:
        exprs_in = exprs1[: len(exprs_in)]
        exprs_out = exprs1[len(exprs_in) : len(exprs_in) + len(exprs_out)]
        equations.extend(equations_stage3(exprs_in, exprs_out))

    exprs1, exprs2 = stage3.solve(equations, invocation=invocation)

    if verbose:
        print("Stage3:")
        for expr1, expr2 in zip(exprs1, exprs2, strict=False):
            print(f"  {expr1} = {expr2}")

    exprs_in = exprs1[: len(exprs_in)]
    exprs_out = exprs1[len(exprs_in) : len(exprs_in) + len(exprs_out)]

    return exprs_in, exprs_out
