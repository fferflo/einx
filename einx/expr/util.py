from . import stage1, stage2, stage3
import numpy as np
import einx

class Condition:
    def __init__(self, expr, value=None, shape=None, depth=None):
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
        return f"{self.expr} = {self.value.tolist()} (shape={self.shape} at depth={self.depth})"

    def __hash__(self):
        return hash((self.expr, self.value, self.shape, self.depth))

def _to_str(l): # Print numpy arrays in a single line rather than with line breaks
    if l is None:
        return "None"
    elif isinstance(l, np.ndarray):
        return str(tuple(l.tolist()))
    elif isinstance(l, list):
        return str(tuple(l))
    else:
        return str(l)

def solve(conditions, cse=True, cse_concat=True, verbose=False):
    if any(not isinstance(c, Condition) for c in conditions):
        raise ValueError("All arguments must be of type Condition")

    expressions = [t.expr for t in conditions]
    values = [t.value for t in conditions]
    shapes = [t.shape for t in conditions]
    depths = [t.depth for t in conditions]

    if verbose:
        print("Stage0:")
        for expr, value, shape, depth in zip(expressions, values, shapes, depths):
            print(f"    {expr} = {_to_str(value)} (shape={_to_str(shape)} at depth={depth})")

    expressions = [(stage1.parse(expr) if isinstance(expr, str) else expr) for expr in expressions]

    if verbose:
        print("Stage1:")
        for expr, value, shape, depth in zip(expressions, values, shapes, depths):
            print(f"    {expr} = {_to_str(value)} (shape={_to_str(shape)} at depth={depth})")

    expressions = stage2.solve(expressions, shapes, depths)

    # Broadcast values to the solved shapes. E.g. for a parameter 'a=2' that is used in an expression '(a...)...' (where each
    # ellipsis expands twice) this would broadcast to '[[2, 2], [2, 2]]' and return the flattened result.
    def broadcast(value, oldshape, newshape):
        if value is None:
            return None
        if list(oldshape) == list(newshape):
            return value
        value = np.asarray(value).reshape([-1])
        n_old = np.prod(oldshape) if len(oldshape) > 0 else 1
        n_new = np.prod(newshape)
        assert n_new > 0
        p = n_new // n_old
        assert n_new % n_old == 0
        value = value[np.newaxis]
        value = np.broadcast_to(value, [p, value.shape[1]])
        value = value.reshape([-1])
        return value
    values = [broadcast(value, oldshape, expr.shape) for value, oldshape, expr in zip(values, shapes, expressions)]

    if verbose:
        print("Stage2:")
        for expr, value in zip(expressions, values):
            print(f"    {expr} = {_to_str(value)}")

    if cse:
        expressions = stage2.cse(expressions, cse_concat=cse_concat)

        if verbose:
            print("Stage2.CSE:")
            for expr, value in zip(expressions, values):
                print(f"    {expr} = {_to_str(value)}")

    expressions = stage3.solve(expressions, values)

    if verbose:
        print("Stage3:")
        for expr in expressions:
            print(f"    {expr} = {_to_str(expr.shape)}")

    return expressions
