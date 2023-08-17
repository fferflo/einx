from . import stage1, stage2, stage3, cse
import numpy as np

class Condition:
    def __init__(self, expr: str, value=None, shape=None, depth=None):
        self.expr = expr

        self.value = np.asarray(value) if not value is None else None
        self.shape = np.asarray(shape) if not shape is None else None
        self.depth = depth

        if not self.value is None:
            if self.shape is None:
                self.shape = np.asarray(self.value.shape)
            elif np.any(self.shape != self.value.shape):
                raise ValueError(f"Got conflicting value.shape {value.shape} and shape {shape} for expression {expr}")

    def __repr__(self):
        return f"{self.expr} = {self.value} (shape={self.shape} at depth={self.depth})"

    def __hash__(self):
        return hash((self.expr, self.value, self.shape, self.depth))

def solve(conditions, stages=3, verbose=False):
    if any(not isinstance(c, Condition) for c in conditions):
        raise ValueError("All arguments must be of type Condition")
    if not stages in [1, 2, 3]:
        raise ValueError("stages must be 1, 2 or 3")

    expressions = [t.expr for t in conditions]
    values = [t.value for t in conditions]
    shapes = [t.shape for t in conditions]
    depths = [t.depth for t in conditions]

    if verbose:
        print("Stage0:")
        for expr, value, shape, depth in zip(expressions, values, shapes, depths):
            print(f"    {expr} = {value} (shape={shape} at depth={depth})")

    expressions = [(stage1.parse(expr) if isinstance(expr, str) else expr) for expr in expressions]

    if verbose:
        print("Stage1:")
        for expr, value, shape, depth in zip(expressions, values, shapes, depths):
            print(f"    {expr} = {value} (shape={shape} at depth={depth})")

    if stages == 1:
        return expressions

    expressions, shapes, depths = stage2.solve(expressions, shapes, depths)
    def broadcast(value, shape):
        value = np.asarray(value)
        while len(value.shape) < len(shape):
            value = value[:, np.newaxis]
        value = np.broadcast_to(value, shape)
        return value
    values = [(broadcast(value, shape).reshape([-1]) if not value is None else None) for value, shape in zip(values, shapes)]

    if verbose:
        print("Stage2:")
        for expr, value in zip(expressions, values):
            print(f"    {expr} = {value}")

    if stages == 2:
        return expressions

    expressions, values = stage3.solve(expressions, values)

    if verbose:
        print("Stage3:")
        for expr, value in zip(expressions, values):
            print(f"    {expr} = {value}")

    return expressions

def get_flattened_axes(expr):
    if isinstance(expr, list):
        result = []
        for expr in expr:
            result.extend(get_flattened_axes(expr))
        return result
    elif isinstance(expr, stage3.Variable):
        if expr.value is None:
            raise ValueError(f"Failed to determine value of variable {expr.name}")
        return [expr]
    elif isinstance(expr, stage3.Ellipsis):
        return get_flattened_axes(expr.inner)
    elif isinstance(expr, stage3.Group):
        return get_flattened_axes(expr.unexpanded_children)
    elif isinstance(expr, cse.CommonSubexpression):
        if expr.value is None:
            raise ValueError(f"Failed to determine value of expression '{expr.name}'")
        return [expr]
    else:
        raise ValueError("Invalid type of expr")

def get_flattened_shape(expr):
    return np.asarray([v.value for v in get_flattened_axes(expr)])

def get_isolated_axes(exprs):
    return [set(get_flattened_axes(expr)).difference([v for expr2 in exprs for v in get_flattened_axes(expr2) if id(expr2) != id(expr)]) for expr in exprs]