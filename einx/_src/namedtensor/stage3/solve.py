import einx._src.util.solver as solver
from .. import stage2
from einx._src.frontend.errors import AxisSizeError
from .tree import *


class Equation:
    def __init__(self, expr1, expr2=None, desc1=None, desc2=None):
        self.expr1 = expr1
        self.expr2 = expr2
        if desc1 is None:
            desc1 = f'expression "{self.expr1}"'
        if desc2 is None:
            desc2 = f'expression "{self.expr2}"'
        self.desc1 = desc1
        self.desc2 = desc2


def solve(equations, invocation, verbose=False):
    exprs1 = [eq.expr1 for eq in equations]
    exprs2 = [eq.expr2 for eq in equations]
    if any(expr is not None and not isinstance(expr, stage2.Expression) for expr in exprs1 + exprs2):
        raise ValueError("Can only expand stage2.Expression")

    if verbose:
        print("Inputs:")
        for eq in equations:
            print(f"  {eq.expr1} ({eq.desc1}) = {eq.expr2} ({eq.desc2})")

    equations = []

    symbolic_expr_values = {}
    for root in exprs1 + exprs2:
        if root is not None:
            for expr in root.nodes():
                symbolic_expr_values[id(expr)] = solver.Variable(f"symbolic_expr_values[{id(expr)}]", str(expr))

    # Add equations: Relations between expressions and their children
    for root in exprs1 + exprs2:
        if root is not None:
            for expr in root.nodes():
                if isinstance(expr, stage2.List):
                    equations.append((solver.Product([symbolic_expr_values[id(c)] for c in expr.children]), symbolic_expr_values[id(expr)]))
                elif isinstance(expr, stage2.ConcatenatedAxis):
                    equations.append((solver.Sum([symbolic_expr_values[id(c)] for c in expr.children]), symbolic_expr_values[id(expr)]))
                elif isinstance(expr, stage2.Brackets) or isinstance(expr, stage2.FlattenedAxis):
                    equations.append((symbolic_expr_values[id(expr)], symbolic_expr_values[id(expr.inner)]))

    # Add equations: Same root values
    for root1, root2 in zip(exprs1, exprs2, strict=False):
        if root1 is not None and root2 is not None:
            assert root1.ndim == root2.ndim
            for expr1, expr2 in zip(root1, root2, strict=False):
                equations.append((symbolic_expr_values[id(expr1)], symbolic_expr_values[id(expr2)]))

    # Add equations: Constant axis values
    for root in exprs1 + exprs2:
        if root is not None:
            for expr in root.nodes():
                if isinstance(expr, stage2.Axis) and expr.value is not None:
                    equations.append((symbolic_expr_values[id(expr)], int(expr.value)))

    # Add equations: Multiple occurrences of the same axis must have the same value
    sympy_axis_values = {}
    for root in exprs1 + exprs2:
        if root is not None:
            for axis in root.nodes():
                if isinstance(axis, stage2.Axis):
                    if axis.name not in sympy_axis_values:
                        sympy_axis_values[axis.name] = solver.Variable(f"sympy_axis_values[{axis.name}]", axis.name)
                    equations.append((symbolic_expr_values[id(axis)], sympy_axis_values[axis.name]))

    # Solve
    def solve(equations):
        solutions = solver.solve(equations)

        axis_values = {}
        for k, v in solutions.items():
            if k.startswith("symbolic_expr_values["):
                axis_values[int(k[len("symbolic_expr_values[") : -1])] = int(v)

        failed_axes = set()
        for root in exprs1 + exprs2:
            if root is not None:
                for expr in root.nodes():
                    if isinstance(expr, stage2.Axis):
                        if id(expr) not in axis_values:
                            failed_axes.add(expr)
        if len(failed_axes) > 0:
            failed_axes = sorted({str(x) for x in failed_axes})
            raise solver.SolveExceptionTooManySolutions(", ".join(failed_axes))

        # Raise exception on non-positive values
        failed_exprs = set()
        for root in exprs1 + exprs2:
            if root is not None:
                for expr in root.nodes():
                    if isinstance(expr, stage2.Axis) and axis_values[id(expr)] <= 0:
                        failed_exprs.add(expr)
        if len(failed_exprs) > 0:
            raise solver.SolveExceptionNoSolution()

        return axis_values

    try:
        axis_values = solve(equations)
    except solver.SolveExceptionTooManySolutions as e:
        if e.message is None:
            raise AxisSizeError(
                invocation=invocation,
                message="Failed to uniquely determine the size of all axes in the expression. Please provide more constraints.\n%EXPR%",
                constraints=zip(exprs1, exprs2, strict=False),
            ) from e
        else:
            axis_names = e.message.split(", ")
            axesaxis = "axes" if len(axis_names) > 1 else "axis"
            raise AxisSizeError(
                invocation,
                message=f"Failed to uniquely determine the size of the {axesaxis} {', '.join(axis_names)}. Please provide more constraints.\n%EXPR%",
                pos=invocation.indicator.get_pos_for_axisnames(exprs1 + exprs2, axis_names),
                constraints=zip(exprs1, exprs2, strict=False),
            ) from e
    except solver.SolveExceptionNoSolution as e:
        raise AxisSizeError(
            invocation,
            message="Failed to determine the size of all axes in the expression under the given constraints.\n%EXPR%",
            constraints=zip(exprs1, exprs2, strict=False),
        ) from e

    # Map stage2 expressions to stage3 expressions
    def map(expr):
        if isinstance(expr, stage2.Axis):
            assert id(expr) in axis_values and axis_values[id(expr)] > 0
            return Axis(expr.name, axis_values[id(expr)], expr.begin_pos, expr.end_pos)
        elif isinstance(expr, stage2.List):
            return List.create([map(child) for child in expr.children], expr.begin_pos, expr.end_pos)
        elif isinstance(expr, stage2.ConcatenatedAxis):
            return ConcatenatedAxis.create([map(child) for child in expr.children], expr.begin_pos, expr.end_pos)
        elif isinstance(expr, stage2.Brackets):
            return Brackets.create(map(expr.inner), expr.begin_pos, expr.end_pos)
        elif isinstance(expr, stage2.FlattenedAxis):
            return FlattenedAxis.create(map(expr.inner), expr.begin_pos, expr.end_pos)
        else:
            raise AssertionError(type(expr))

    exprs1 = [map(root) if root is not None else None for root in exprs1]
    exprs2 = [map(root) if root is not None else None for root in exprs2]

    return exprs1, exprs2
