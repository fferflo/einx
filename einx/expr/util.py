from . import stage1, stage2, stage3
import numpy as np
import einx


def idx_to_ordinal(idx):
    if idx % 100 == 0:
        return "1st"
    elif idx % 100 == 1:
        return "2nd"
    elif idx % 100 == 2:
        return "3rd"
    else:
        return f"{idx + 1}th"


class CallSignature:
    def __init__(self, text, parameters=None):
        if parameters is None:
            parameters = {}
        self.text = text
        self.parameters = parameters

    def get_pos_for_literal(self, literal):
        pos = []
        for i in range(len(self.text)):
            if self.text[i:].startswith(literal):
                pos.extend(range(i, i + len(literal)))
        assert all(p >= 0 and p < len(self.text) for p in pos), f"{pos}"
        return pos

    def get_pos_for_exprs(self, exprs):
        pos = []
        for expr in exprs:
            pos.extend(range(expr.begin_pos, expr.end_pos))
        assert all(p >= 0 and p < len(self.text) for p in pos), f"{pos}"
        return pos

    def get_pos_for_axisnames(self, exprs, axisnames):
        pos = []
        for expr in exprs:
            if expr is not None:
                for expr in expr.all():
                    if (
                        isinstance(expr, (stage1.NamedAxis, stage2.NamedAxis, stage3.Axis))
                        and expr.name in axisnames
                    ):
                        pos.extend(range(expr.begin_pos, expr.end_pos))
        assert all(p >= 0 and p < len(self.text) for p in pos), f"{pos}"
        return pos

    def get_pos_for_ellipses(self, exprs):
        pos = []
        for expr in exprs:
            if expr is not None:
                for expr in expr.all():
                    if isinstance(expr, stage1.Ellipsis):
                        if expr.begin_pos >= 0:
                            pos.extend(range(expr.end_pos - 3, expr.end_pos))
        assert all(p >= 0 and p < len(self.text) for p in pos), f"{pos}"
        return pos

    def get_pos_for_concatenations(self, exprs):
        pos = []
        for expr in exprs:
            if expr is not None:
                for expr in expr.all():
                    if isinstance(expr, stage1.Concatenation):
                        if expr.begin_pos >= 0:
                            pos.extend(range(expr.begin_pos, expr.end_pos))
        assert all(p >= 0 and p < len(self.text) for p in pos), f"{pos}"
        return pos

    def get_pos_for_brackets(self, exprs):
        pos = []
        for expr in exprs:
            if expr is not None:
                for expr in expr.all():
                    if isinstance(expr, stage1.Marker):
                        if expr.begin_pos >= 0:
                            pos.extend([expr.begin_pos, expr.end_pos - 1])
        assert all(p >= 0 and p < len(self.text) for p in pos), f"{pos}"
        return pos


class SyntaxError(Exception):
    def __init__(self, expression, pos, message, post_message=None):
        if isinstance(pos, int):
            self.pos = [pos]
        else:
            self.pos = list(pos)
        self.expression = expression
        self.message = f'{message}\nExpression: "{self.expression}"\n'
        self.message += " " * 13 + "".join([
            ("^" if i in self.pos else " ") for i in range(len(self.expression))
        ])
        if post_message is not None:
            self.message += f"\n{post_message}"
        assert all(p >= 0 and p < len(self.expression) for p in self.pos)

    def __str__(self):
        return self.message


SyntaxError.__module__ = "einx"
SyntaxError.__qualname__ = "SyntaxError"


class DimensionError(Exception):
    def __init__(self, message, text=None, pos=None, postfix=None, constraints=None):
        if constraints is not None:
            if len(constraints) == 0:
                constraints = None
            else:
                constraints_str = "Constraints:\n"
                for expr1, expr2 in constraints:
                    constraints_str += f"    {expr1} = {expr2}\n"
                constraints = constraints_str[:-1]

        if text is not None:
            if isinstance(pos, int):
                self.pos = [pos]
            elif pos is None:
                self.pos = []
            else:
                self.pos = list(pos)
            assert all(p >= 0 and p < len(text) for p in self.pos), f"{self.pos}"

            self.text = text
            self.message = f'{message}\nExpression: "{self.text}"'
            if len(self.pos) > 0:
                self.message += (
                    "\n"
                    + " " * 13
                    + "".join([("^" if i in self.pos else " ") for i in range(len(self.text))])
                )
            if constraints is not None:
                self.message += f"\n{constraints}"
        else:
            self.message = message
        if postfix is not None:
            self.message += f"\n{postfix}"

    def __str__(self):
        return self.message


DimensionError.__module__ = "einx"
DimensionError.__qualname__ = "DimensionError"


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
    def __init__(self, expr1, expr2=None, depth1=0, depth2=0, desc1=None, desc2=None):
        self.expr1 = _input_expr(expr1)
        self.expr2 = _input_expr(expr2)
        if desc1 is None:
            desc1 = f'expression "{self.expr1}"'
        if desc2 is None:
            desc2 = f'expression "{self.expr2}"'
        self.expansion1 = _get_expansion(expr1)
        self.expansion2 = _get_expansion(expr2)
        self.depth1 = depth1
        self.depth2 = None if expr2 is None else depth2
        self.desc1 = desc1
        self.desc2 = desc2

    def __repr__(self):
        return f"{self.expr} = {self.value.tolist()} (expansion={self.expansion} at "
        f"depth={self.depth})"


def input_equations(exprs, tensor_shapes=None):
    if tensor_shapes is None:
        tensor_shapes = [None] * len(exprs)
    if len(exprs) != len(tensor_shapes):
        raise ValueError("Number of expressions and tensor shapes must match")

    def shape_to_str(shape):
        if shape is None:
            return ""
        else:
            return f" (with shape ({', '.join([str(x) for x in shape])}))"

    if len(exprs) == 1:
        return [
            einx.expr.Equation(
                exprs[0],
                tensor_shapes[0],
                desc1=f'input expression ("{exprs[0]}")',
                desc2=f"input tensor{shape_to_str(tensor_shapes[0])}",
            )
        ]
    else:
        return [
            einx.expr.Equation(
                expr,
                tensor_shape,
                desc1=f'{einx.expr.idx_to_ordinal(i)} input expression ("{expr}")',
                desc2=f"{einx.expr.idx_to_ordinal(i)} input tensor{shape_to_str(tensor_shape)}",
            )
            for i, (expr, tensor_shape) in enumerate(zip(exprs, tensor_shapes))
        ]


def output_equations(exprs):
    if len(exprs) == 1:
        return [
            einx.expr.Equation(
                exprs[0],
                desc1=f'output expression ("{exprs[0]}")',
            )
        ]
    else:
        return [
            einx.expr.Equation(
                expr,
                desc1=f'{einx.expr.idx_to_ordinal(i)} output expression ("{expr}")',
            )
            for i, expr in enumerate(exprs)
        ]


def constraint_equations(parameters):
    return [
        einx.expr.Equation(
            k,
            np.asarray(v)[..., np.newaxis],
            depth1=None,
            depth2=None,
            desc1=f"axis {k}",
            desc2=f"constraint ({_to_str(np.asarray(v))})",
        )
        for k, v in parameters.items()
    ]


def _to_str(l):  # Print numpy arrays in a single line rather than with line breaks
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
    equations,
    cse=True,
    cse_concat=True,
    cse_in_markers=False,
    after_stage2=None,
    verbose=False,
    signature=None,
):
    if any(not isinstance(c, Equation) for c in equations):
        raise ValueError("All arguments must be of type Equation")

    exprs1 = [t.expr1 for t in equations]
    exprs2 = [t.expr2 for t in equations]
    expansions1 = [t.expansion1 for t in equations]
    expansions2 = [t.expansion2 for t in equations]
    depths1 = [t.depth1 for t in equations]
    depths2 = [t.depth2 for t in equations]
    descs1 = [t.desc1 for t in equations]
    descs2 = [t.desc2 for t in equations]

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

    exprs1, exprs2 = stage2.solve(
        exprs1,
        exprs2,
        expansions1,
        expansions2,
        depths1,
        depths2,
        descs1,
        descs2,
        signature=signature,
    )

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

    exprs1, exprs2 = stage3.solve(exprs1, exprs2, descs1, descs2, signature=signature)

    if verbose:
        print("Stage3:")
        for expr1, expr2 in zip(exprs1, exprs2):
            assert expr1 is None or expr2 is None or expr1.shape == expr2.shape
            shape = expr1.shape if expr1 is not None else expr2.shape
            shape = " ".join(str(i) for i in shape)
            print(f"    {_to_str(expr1)} = {_to_str(expr2)} = {shape}")

    return exprs1
