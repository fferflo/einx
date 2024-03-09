from . import stage1, stage2, stage3
import numpy as np
import einx


def _get_expansion(expr):
    if isinstance(expr, stage1.Expression):
        return (expr.expansion(),)
    elif isinstance(expr, (stage2.Expression, stage3.Expression)):
        return (len(expr),)
    elif isinstance(expr, np.ndarray):
        return tuple(expr.shape)
    else:
        return None


def _input_expr(expr):
    if expr is None or isinstance(
        expr, (str, stage1.Expression, stage2.Expression, stage3.Expression)
    ):
        return expr
    else:
        if isinstance(expr, np.ndarray):
            pass
        elif expr == [] or expr == ():
            expr = np.asarray(expr).astype("int32")
        else:
            try:
                expr = np.asarray(expr)
            except Exception as e:
                raise ValueError(f"Invalid expression '{expr}'") from e
        if not np.issubdtype(expr.dtype, np.integer):
            raise ValueError(f"Invalid expression '{expr}', must be integers")
        expr = " ".join([str(i) for i in expr.flatten()])
        return expr


class Equation:
    def __init__(self, expr1, expr2=None, depth1=0, depth2=0):
        self.expr1 = _input_expr(expr1)
        self.expr2 = _input_expr(expr2)
        self.expansion1 = _get_expansion(expr1)
        self.expansion2 = _get_expansion(expr2)
        self.depth1 = depth1
        self.depth2 = None if expr2 is None else depth2

    def __repr__(self):
        return f"{self.expr} = {self.value.tolist()} (expansion={self.expansion} at "
        f"depth={self.depth})"


def _to_str(l):  # Print numpy arrays in a single line rather than with line breaks
    if l is None:
        return "None"
    elif isinstance(l, np.ndarray):
        return str(tuple(l.tolist()))
    elif isinstance(l, list):
        return str(tuple(l))
    else:
        return str(l)


def solve(
    equations, cse=True, cse_concat=True, cse_in_markers=False, after_stage2=None, verbose=False
):
    if any(not isinstance(c, Equation) for c in equations):
        raise ValueError("All arguments must be of type Equation")

    exprs1 = [t.expr1 for t in equations]
    exprs2 = [t.expr2 for t in equations]
    expansions1 = [t.expansion1 for t in equations]
    expansions2 = [t.expansion2 for t in equations]
    depths1 = [t.depth1 for t in equations]
    depths2 = [t.depth2 for t in equations]

    if verbose:
        print("Stage0:")
        for expr1, expr2, expansion1, expansion2, depth1, depth2 in zip(
            exprs1, exprs2, expansions1, expansions2, depths1, depths2
        ):
            print(
                f"    {_to_str(expr1)} (expansion={_to_str(expansion1)} at depth={depth1}) = "
                f"{_to_str(expr2)} (expansion={_to_str(expansion2)} at depth={depth2})"
            )

    exprs1 = [(stage1.parse_arg(expr) if isinstance(expr, str) else expr) for expr in exprs1]
    exprs2 = [(stage1.parse_arg(expr) if isinstance(expr, str) else expr) for expr in exprs2]

    expansions1 = [
        expansion if expansion is not None else _get_expansion(expr)
        for expansion, expr in zip(expansions1, exprs1)
    ]
    expansions2 = [
        expansion if expansion is not None else _get_expansion(expr)
        for expansion, expr in zip(expansions2, exprs2)
    ]

    if verbose:
        print("Stage1:")
        for expr1, expr2, expansion1, expansion2, depth1, depth2 in zip(
            exprs1, exprs2, expansions1, expansions2, depths1, depths2
        ):
            print(
                f"    {_to_str(expr1)} (expansion={_to_str(expansion1)} at depth={depth1}) = "
                f"{_to_str(expr2)} (expansion={_to_str(expansion2)} at depth={depth2})"
            )

    exprs1, exprs2 = stage2.solve(exprs1, exprs2, expansions1, expansions2, depths1, depths2)

    if verbose:
        print("Stage2:")
        for expr1, expr2 in zip(exprs1, exprs2):
            print(f"    {_to_str(expr1)} = {_to_str(expr2)}")

    if cse:
        exprs = stage2.cse(exprs1 + exprs2, cse_concat=cse_concat, cse_in_markers=cse_in_markers)
        exprs1, exprs2 = exprs[: len(exprs1)], exprs[len(exprs1) :]

        if verbose:
            print("Stage2.CSE:")
            for expr1, expr2 in zip(exprs1, exprs2):
                print(f"    {_to_str(expr1)} = {_to_str(expr2)}")

    if after_stage2 is not None:
        return solve(
            equations + after_stage2(exprs1, exprs2),
            cse=cse,
            cse_concat=cse_concat,
            cse_in_markers=cse_in_markers,
            after_stage2=None,
            verbose=verbose,
        )

    exprs1, exprs2 = stage3.solve(exprs1, exprs2)

    if verbose:
        print("Stage3:")
        for expr1, expr2 in zip(exprs1, exprs2):
            assert expr1 is None or expr2 is None or expr1.shape == expr2.shape
            shape = expr1.shape if expr1 is not None else expr2.shape
            shape = " ".join(str(i) for i in shape)
            print(f"    {_to_str(expr1)} = {_to_str(expr2)} = {shape}")

    return exprs1
