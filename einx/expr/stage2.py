from . import stage1
import sympy, re
import numpy as np

class Node:
    def __init__(self, ellipsis_indices, name=None):
        self.parent = None
        self.ellipsis_indices = ellipsis_indices
        self.name = name if not name is None else f"__{self.__class__.__name__}__{id(self)}"

class Variable(Node):
    def __init__(self, unexpanded_variable, ellipsis_indices):
        name = unexpanded_variable.name
        for i in ellipsis_indices:
            if i is None:
                name += "__X"
            else:
                name += f"__{i}"
        Node.__init__(self, ellipsis_indices, name=name)
        self.value = unexpanded_variable.value

    def __str__(self):
        return self.name if self.value is None else str(self.value)

    def traverse(self):
        yield self

class Ellipsis(Node):
    def __init__(self, inner, ellipsis_indices):
        Node.__init__(self, ellipsis_indices)
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

    def traverse(self):
        yield self
        for x in [self.inner] if not self.is_expanded else self.inner:
            for y in x.traverse():
                yield y

class Group(Node):
    def __init__(self, unexpanded_children, ellipsis_indices, front, back):
        Node.__init__(self, ellipsis_indices)
        self.unexpanded_children = unexpanded_children
        for child in unexpanded_children:
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

    @property
    def expanded_children(self):
        return expand(self.unexpanded_children)

class Root(Group):
    def __init__(self, unexpanded_children):
        Group.__init__(self, unexpanded_children, ellipsis_indices=[], front="", back="")
        self.parent = None

def expand(expr):
    children = []
    if isinstance(expr, list):
        for expr in expr:
            children.extend(expand(expr))
    elif isinstance(expr, Variable) or (isinstance(expr, Group) and expr.front == "("):
        children.append(expr)
    elif isinstance(expr, Group) and expr.front != "(":
        children.extend(expand(expr.unexpanded_children))
    elif isinstance(expr, Ellipsis):
        children.extend(expand(expr.inner))
    return children

def solve(expressions, shapes, depths):
    if any(not isinstance(expr, stage1.Root) for expr in expressions):
        raise ValueError("Can only expand stage1.Root expressions")
    if len(shapes) != len(expressions) or len(depths) != len(expressions):
        raise ValueError("Number of expressions, shapes and depths must be equal")
    depths = [d for d in depths]

    # Find expression depths
    equations = []
    sympy_expr_depth = {}
    for expr in expressions:
        for expr in expr.traverse():
            sympy_expr_depth[expr.name] = sympy.Symbol(expr.name, integer=True)
    for expr in expressions:
        for expr in expr.traverse():
            if isinstance(expr, stage1.Group):
                for child in expr.children:
                    equations.append(sympy.Eq(sympy_expr_depth[expr.name], sympy_expr_depth[child.name]))
            elif isinstance(expr, stage1.Variable):
                pass
            elif isinstance(expr, stage1.Ellipsis):
                equations.append(sympy.Eq(sympy_expr_depth[expr.name] + 1, sympy_expr_depth[expr.inner.name]))
            else:
                assert False
    for expr, depth in zip(expressions, depths):
        if not depth is None:
            equations.append(sympy.Eq(sympy_expr_depth[expr.name], depth))
    expr_depth = sympy.solve(equations)
    if not isinstance(expr_depth, dict):
        raise ValueError("Failed to solve expression depths")
    expr_depth = {str(k): int(v) for k, v in expr_depth.items() if v.is_number}

    for i, expr in enumerate(expressions):
        for expr in expr.traverse():
            if not expr.name in expr_depth:
                raise ValueError(f"Failed to determine depth of expression {expr}")
        if depths[i] is None and not shapes[i] is None:
            assert len(shapes[i]) >= 1
            missing_depth = expr_depth[expr.name] - (len(shapes[i]) - 1)
            assert missing_depth >= 0
            if missing_depth > 0:
                shapes[i] = [shapes[i][0]] + [None] * missing_depth + list(shapes[i][1:])


    equations = []

    # Create Sympy symbols
    sympy_ellipsis_expansion = {}
    for expr in expressions:
        for ellipsis in expr.traverse():
            if isinstance(ellipsis, stage1.Ellipsis):
                sympy_ellipsis_expansion[ellipsis.name] = sympy.Symbol(f"__ellipsisexpansion__{ellipsis.name}", integer=True)

    sympy_expr_expansion = {}
    for expr in expressions:
        for expr in expr.traverse():
            for i in range(expr_depth[expr.name]):
                key = (expr.name, i)
                sympy_expr_expansion[key] = sympy.Symbol(f"__exprexpansion__{i}__{expr.name}", integer=True)

    # Node relations
    for expr in expressions:
        for expr in expr.traverse():
            if isinstance(expr, stage1.Group):
                for i in range(expr_depth[expr.name]):
                    for child in expr.children:
                        equations.append(sympy.Eq(sympy_expr_expansion[(expr.name, i)], sympy_expr_expansion[(child.name, i)]))
            elif isinstance(expr, stage1.Variable):
                pass
            elif isinstance(expr, stage1.Ellipsis):
                for i in range(expr_depth[expr.name]):
                    equations.append(sympy.Eq(sympy_expr_expansion[(expr.name, i)], sympy_expr_expansion[(expr.inner.name, i)]))
            else:
                assert False

    # Expression expansion = ellipsis expansion
    for expr in expressions:
        for expr in expr.traverse():
            assert expr_depth[expr.name] >= len(expr.ellipses)
            for i, ellipsis in enumerate(expr.ellipses):
                key = (expr.name, expr_depth[expr.name] - len(expr.ellipses) + i)
                equations.append(sympy.Eq(sympy_expr_expansion[key], sympy_ellipsis_expansion[ellipsis.name]))

    # Root expression expansions
    def get_expansion(expr):
        if isinstance(expr, stage1.Root) or (isinstance(expr, stage1.Group) and expr.front != "("):
            return sum(get_expansion(c) for c in expr.children)
        elif isinstance(expr, stage1.Variable) or (isinstance(expr, stage1.Group) and expr.front == "("):
            return 1
        elif isinstance(expr, stage1.Ellipsis):
            return sympy_ellipsis_expansion[expr.name]
        else:
            assert False
    for expr, shape in zip(expressions, shapes):
        if not shape is None:
            assert not shape[0] is None
            equations.append(sympy.Eq(get_expansion(expr), int(shape[0])))

    # Shapes
    for expr, shape in zip(expressions, shapes):
        if not shape is None:
            for child in expr.children:
                for i in range(1, len(shape)):
                    if not shape[i] is None:
                        key = (child.name, i - 1)
                        equations.append(sympy.Eq(sympy_expr_expansion[key], int(shape[i])))

    # Solve
    if all(eq.is_Boolean and bool(eq) for eq in equations):
        expansion_values = {}
    else:
        expansion_values = sympy.solve(equations, set=True)
        if expansion_values == []:
            expansion_values = {}
        elif isinstance(expansion_values, tuple) and len(expansion_values) == 2:
            variables, solutions = expansion_values
            if len(solutions) != 1:
                raise ValueError("Failed to solve ellipsis expansion")
            solutions = next(iter(solutions))
            expansion_values = {str(k): v for k, v in zip(variables, solutions) if v.is_number}
        else:
            raise ValueError("Failed to solve ellipsis expansion")

    # Extract root shapes
    shapes = []
    for root in expressions:
        if len(root.children) > 0:
            # Child shapes
            child_shapes = set()
            for child in root.children:
                shape = []
                for i in range(expr_depth[root.name]):
                    sympy_name = f"__exprexpansion__{i}__{root.name}"
                    if not sympy_name in expansion_values:
                        raise ValueError(f"Failed solving expansion of expression {root}")
                    shape.append(expansion_values[sympy_name])
                child_shapes.add(tuple(np.asarray(shape).tolist()))
            if len(child_shapes) != 1:
                raise ValueError(f"Failed solving expansion of expression {root}")
            child_shape = next(iter(child_shapes))

            # Child num
            for child in root.children:
                if isinstance(child, stage1.Ellipsis) and not f"__ellipsisexpansion__{child.name}" in expansion_values:
                    raise ValueError(f"Failed to determine expansion of ellipsis {child.name}")
            def get_expansion(expr):
                if isinstance(expr, stage1.Root) or (isinstance(expr, stage1.Group) and expr.front != "("):
                    return sum(get_expansion(c) for c in expr.children)
                elif isinstance(expr, stage1.Variable) or (isinstance(expr, stage1.Group) and expr.front == "("):
                    return 1
                elif isinstance(expr, stage1.Ellipsis):
                    return expansion_values[f"__ellipsisexpansion__{expr.name}"]
                else:
                    assert False
            child_num = get_expansion(root)

            shapes.append(np.asarray([child_num] + list(child_shape)))
        else:
            assert get_expansion(root) == 0
            shapes.append(np.asarray([0]))

    # Modify expressions
    unexpanded_variable_names = set(expr.name for expr in expr.traverse() if isinstance(expr, stage1.Variable))
    def replace(expr, ellipsis_indices):
        if isinstance(expr, list):
            return [replace(expr, ellipsis_indices=ellipsis_indices) for expr in expr]
        elif isinstance(expr, stage1.Variable):
            v = Variable(expr, ellipsis_indices=ellipsis_indices)
            if len(ellipsis_indices) > 0 and v.name in unexpanded_variable_names:
                raise ValueError(f"Expanded variable name {v.name} was already defined as unexpanded variable")
            return v
        elif isinstance(expr, stage1.Ellipsis):
            sympy_name = f"__ellipsisexpansion__{expr.name}"
            if sympy_name in expansion_values:
                inner = [replace(expr.inner, ellipsis_indices=ellipsis_indices + [i]) for i in range(expansion_values[sympy_name])]
            else:
                inner = replace(expr.inner, ellipsis_indices=ellipsis_indices + [None])
            return Ellipsis(inner, ellipsis_indices=ellipsis_indices)
        elif isinstance(expr, stage1.Group):
            children = [replace(c, ellipsis_indices=ellipsis_indices) for c in expr.children]
            return Group(children, ellipsis_indices=ellipsis_indices, front=expr.front, back=expr.back)
        else:
            assert False

    def init_depth(expr, remaining_shape, ellipsis_indices):
        if len(remaining_shape) > 0:
            children = []
            for s in range(remaining_shape[0]):
                children.extend(init_depth(expr, remaining_shape=remaining_shape[1:], ellipsis_indices=ellipsis_indices + [s]))
            return children
        else:
            return replace(expr, ellipsis_indices=ellipsis_indices)

    expressions = [Root(init_depth(expr.children, remaining_shape=shape[1:], ellipsis_indices=[])) for expr, shape in zip(expressions, shapes)]

    return expressions, shapes, depths
