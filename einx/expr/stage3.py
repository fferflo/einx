from . import stage2
import numpy as np
import sympy, math

class Node:
    def __init__(self, value, ellipsis_indices, name=None):
        self.value = int(value) if not value is None else None
        self.ellipsis_indices = ellipsis_indices
        self.name = name if not name is None else f"__{self.__class__.__name__}__{id(self)}"

    def __repr__(self):
        return str(self)

    @property
    def variables(self):
        return [x for x in self.traverse() if isinstance(x, Variable)]

class Variable(Node):
    def __init__(self, name, value, ellipsis_indices):
        if name is None:
            name = f"__constantdim{id(self)}({value})"
        Node.__init__(self, value, ellipsis_indices, name=name)

    def __str__(self):
        return str(self.name)

    def expand(self):
        return [self]

    def traverse(self):
        yield self

    def __eq__(self, other):
        return other.__class__ == Variable and str(self) == str(other) and self.value == other.value

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(str(self))

    def copy(self):
        return Variable(self.name, self.value, self.ellipsis_indices)

class Ellipsis(Node):
    def __init__(self, inner, value, ellipsis_indices):
        Node.__init__(self, value, ellipsis_indices)
        self.inner = inner
        if self.is_expanded:
            for child in self.inner:
                child.parent = self
        else:
            self.inner.parent = self

    @property
    def is_expanded(self):
        return isinstance(self.inner, list)

    def __str__(self):
        if self.is_expanded:
            return " ".join([str(x) for x in self.inner])
        else:
            return str(self.inner) + "..."

    def expand(self):
        return expand(self.inner)

    def traverse(self):
        yield self
        for x in [self.inner] if not self.is_expanded else self.inner:
            for y in x.traverse():
                yield y

    def __eq__(self, other):
        return other.__class__ == Ellipsis and self.inner == other.inner

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(str(self))

    def copy(self):
        return Ellipsis(copy(self.inner), self.value, self.ellipsis_indices)

class Group(Node):
    def __init__(self, unexpanded_children, value, ellipsis_indices, front, back):
        Node.__init__(self, value, ellipsis_indices)
        self.unexpanded_children = unexpanded_children
        for child in self.unexpanded_children:
            child.parent = self
        self.front = front
        self.back = back

    def __str__(self):
        return self.front + " ".join([str(c) for c in self.unexpanded_children]) + self.back

    def traverse(self):
        yield self
        for x in self.unexpanded_children:
            for y in x.traverse():
                yield y

    def expand(self):
        return [self]

    @property
    def expanded_children(self):
        return expand(self.unexpanded_children)

    def __eq__(self, other):
        return other.__class__ == Group and self.front == other.front and self.back == other.back and self.unexpanded_children == other.unexpanded_children

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(str(self))

    def copy(self):
        return Group(copy(self.unexpanded_children), self.value, self.ellipsis_indices, self.front, self.back)

class Root(Group):
    def __init__(self, unexpanded_children, value):
        Group.__init__(self, unexpanded_children, value, ellipsis_indices=[], front="", back="")
        assert not value is None
        self.parent = None

    def copy(self):
        return Root(copy(self.unexpanded_children), self.value)

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other):
        return other.__class__ == Root and self.unexpanded_children == other.unexpanded_children

    shape = property(lambda self: tuple(c.value for c in self.expanded_children))

def solve(expressions, values):
    if any(not isinstance(expr, stage2.Root) for expr in expressions):
        raise ValueError("Can only expand stage2.Root expressions")
    if len(values) != len(expressions):
        raise ValueError("Number of expressions and values must be equal")
    values = [(np.asarray(value).reshape([-1]) if not value is None else None) for value in values]

    equations = []
    sympy_expr_value = {expr.name: sympy.Symbol(f"{expr.name}", integer=True) for root in expressions for expr in root.traverse()}

    # Node relations
    for root in expressions:
        for expr in root.traverse():
            if isinstance(expr, stage2.Group):
                equations.append(sympy.Eq(
                    math.prod([sympy_expr_value[c.name] for c in expr.unexpanded_children]),
                    sympy_expr_value[expr.name],
                ))
            elif isinstance(expr, stage2.Ellipsis):
                if expr.is_expanded:
                    equations.append(sympy.Eq(
                        math.prod([sympy_expr_value[c.name] for c in expr.inner]),
                        sympy_expr_value[expr.name],
                    ))
                else:
                    equations.append(sympy.Eq(
                        sympy_expr_value[expr.inner.name],
                        sympy_expr_value[expr.name],
                    ))

    # Root values
    for i, (expr, value) in enumerate(zip(expressions, values)):
        if not value is None:
            assert len(value) == len(expr.expanded_children)
            for child, v in zip(expr.expanded_children, value):
                equations.append(sympy.Eq(sympy_expr_value[child.name], int(v)))

    # Constants
    for expr in expressions:
        for expr in expr.traverse():
            if isinstance(expr, stage2.Variable) and not expr.value is None:
                equations.append(sympy.Eq(sympy_expr_value[expr.name], int(expr.value)))


    # Solve
    equations = list(set(equations))
    axis_values = sympy.solve(equations, set=True)
    if axis_values == []:
        axis_values = {}
    elif isinstance(axis_values, tuple) and len(axis_values) == 2:
        variables, solutions = axis_values
        if len(solutions) != 1:
            raise ValueError("Failed to solve axis values")
        solutions = next(iter(solutions))
        axis_values = {str(k): int(v) for k, v in zip(variables, solutions) if v.is_number}
    else:
        raise ValueError("Failed to solve axis")

    required_exprs = [expr for root in expressions for expr in root.expanded_children]
    failed_exprs = [expr for expr in required_exprs if not expr.name in axis_values]
    if len(failed_exprs) > 0:
        raise ValueError(f"Failed to solve for axis values. Could not determine value for expressions: {[str(expr) for expr in failed_exprs]}")

    # Extract root values
    values = [np.asarray([axis_values[child.name] for child in expr.expanded_children]) for expr in expressions]

    # Modify expressions
    def replace(expr):
        if isinstance(expr, list):
            return [replace(expr) for expr in expr]
        elif isinstance(expr, stage2.Variable):
            return Variable(expr.name, axis_values[expr.name] if expr.name in axis_values else None, ellipsis_indices=expr.ellipsis_indices)
        elif isinstance(expr, stage2.Ellipsis):
            inner = replace(expr.inner)
            return Ellipsis(inner, axis_values[expr.name] if expr.name in axis_values else None, ellipsis_indices=expr.ellipsis_indices)
        elif isinstance(expr, stage2.Group):
            unexpanded_children = replace(expr.unexpanded_children)
            return Group(unexpanded_children, axis_values[expr.name] if expr.name in axis_values else None, ellipsis_indices=expr.ellipsis_indices, front=expr.front, back=expr.back)
        else:
            assert False
    expressions = [Root(replace(expr.unexpanded_children), axis_values[expr.name]) for expr in expressions]

    return expressions, values



def expand(expr):
    result = []
    if isinstance(expr, list):
        for expr in expr:
            result.extend(expand(expr))
    else:
        result.extend(expr.expand())
    return result

def copy(expr):
    if isinstance(expr, list):
        result = []
        for expr in expr:
            result.append(copy(expr))
        return result
    else:
        return expr.copy()
    return result

def value(expr):
    if isinstance(expr, list):
        values = [value(c) for c in expr]
        if all(not v is None for v in values):
            return math.prod(values)
        elif "parent" in dir(expr[0]) and isinstance(expr[0].parent, Group) and not expr[0].parent.value is None:
            v = expr[0].parent.value
            for c in expr[0].parent.unexpanded_children:
                if not c in expr:
                    if c.value is None:
                        break
                    v //= c.value
            else:
                return v
        return None
    else:
        return expr.value

def remove(expr, pred, keepdims=False, drop_empty_groups=False):
    def traverse(expr):
        if isinstance(expr, list):
            result = []
            for expr in expr:
                result.extend(traverse(expr))
            return result
        if pred(expr):
            if keepdims:
                return [Variable(None, 1, ellipsis_indices=expr.ellipsis_indices)]
            else:
                return []
        if isinstance(expr, Variable):
            return [expr.copy()]
        elif isinstance(expr, Ellipsis):
            inner = traverse(expr.inner)
            return [Ellipsis(inner, value(inner), ellipsis_indices=expr.ellipsis_indices)]
        elif isinstance(expr, Group):
            unexpanded_children = traverse(expr.unexpanded_children)
            if len(unexpanded_children) == 0 and len(expr.unexpanded_children) > 0 and drop_empty_groups:
                return []
            else:
                return [Group(unexpanded_children, value(unexpanded_children), expr.ellipsis_indices, expr.front, expr.back)]
        else:
            assert False

    unexpanded_children = traverse(expr.unexpanded_children)
    v = value(unexpanded_children)
    if v is None:
        raise ValueError("Failed to remove subexpression, resulting axis value could not be determined")
    return Root(unexpanded_children, v)

def prune_group(expr, pred):
    def traverse(expr):
        if isinstance(expr, list):
            result = []
            for expr in expr:
                result.extend(traverse(expr))
            return result
        if isinstance(expr, Variable):
            return [expr.copy()]
        elif isinstance(expr, Ellipsis):
            inner = traverse(expr.inner)
            return [Ellipsis(inner, expr.value, ellipsis_indices=expr.ellipsis_indices)]
        elif isinstance(expr, Group):
            unexpanded_children = traverse(expr.unexpanded_children)
            if pred(expr):
                return unexpanded_children
            else:
                return [Group(unexpanded_children, expr.value, expr.ellipsis_indices, expr.front, expr.back)]
        else:
            assert False

    unexpanded_children = traverse(expr.unexpanded_children)
    return Root(unexpanded_children, value(unexpanded_children))

def cache_hash(expr):
    from . import cse
    if isinstance(expr, list):
        h = 878123
        for expr in expr:
            h += cache_hash(expr)
            h *= 978123
        return h
    elif isinstance(expr, Variable):
        return (hash(expr.name) if not expr.name.startswith("__constantdim") else 81273) + 981723 * hash(expr.value)
    elif isinstance(expr, Ellipsis):
        return 7911131823 + cache_hash(expr.inner)
    elif isinstance(expr, Group):
        return 9888734 + cache_hash(expr.unexpanded_children) + hash(expr.front) + hash(expr.back)
    elif isinstance(expr, Root):
        return 234 + cache_hash(expr.unexpanded_children)
    elif isinstance(expr, cse.CommonSubexpression):
        return 123 + cache_hash(expr.children)
    else:
        assert False, type(expr)