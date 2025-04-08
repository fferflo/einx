from . import stage2, solver
import numpy as np
from functools import partial
import einx


class Expression:
    def __init__(self, value, begin_pos=None, end_pos=None):
        if not isinstance(value, (int, np.integer)):
            raise TypeError(f"Expected int, got {type(value)}")
        self.value = int(value)
        self.parent = None
        self.begin_pos = begin_pos
        self.end_pos = end_pos

    @property
    def shape(self):
        return tuple(x.value for x in self)


class Composition(Expression):
    @staticmethod
    def maybe(inner, begin_pos=None, end_pos=None):
        if len(inner) == 0:
            return Axis(None, 1, begin_pos=begin_pos, end_pos=end_pos)
        elif isinstance(inner, list):
            if len(inner) == 1:
                return inner[0]
            else:
                return Composition(
                    List.maybe(inner, begin_pos=begin_pos + 1, end_pos=end_pos - 1),
                    begin_pos=begin_pos,
                    end_pos=end_pos,
                )
        elif isinstance(inner, List) and len(inner) == 1:
            return inner.children[0]
        else:
            return Composition(inner, begin_pos=begin_pos, end_pos=end_pos)

    def __init__(self, inner, begin_pos=None, end_pos=None):
        Expression.__init__(self, inner.value, begin_pos=begin_pos, end_pos=end_pos)
        self.inner = inner
        inner.parent = self
        assert len(inner) > 0

    def __str__(self):
        return f"({self.inner})"

    def __len__(self):
        return 1

    def __iter__(self):
        yield self

    def __deepcopy__(self):
        return Composition(self.inner.__deepcopy__(), self.begin_pos, self.end_pos)

    def __eq__(self, other):
        return isinstance(other, Composition) and self.inner == other.inner

    def __hash__(self):
        return 8716123 + hash(self.inner)

    def all(self):
        yield self
        yield from self.inner.all()


class List(Expression):
    def maybe(l, begin_pos=None, end_pos=None):
        if not isinstance(l, list):
            raise TypeError(f"Expected list, got {type(l)}")
        if len(l) == 1:
            return l[0]
        else:
            return List(l, begin_pos=begin_pos, end_pos=end_pos)

    def __init__(self, children, begin_pos=None, end_pos=None):
        Expression.__init__(
            self,
            np.prod([c.value for c in children]).astype(int),
            begin_pos=begin_pos,
            end_pos=end_pos,
        )
        self.children = children
        for c in children:
            if isinstance(c, List):
                raise ValueError("List cannot have another List as direct child")
            c.parent = self

    def __str__(self):
        return " ".join([str(c) for c in self.children])

    def __getitem__(self, i):
        return self.children[i]

    def __len__(self):
        return sum(len(c) for c in self.children)

    def __iter__(self):
        for c in self.children:
            yield from c

    def __deepcopy__(self):
        return List([c.__deepcopy__() for c in self.children], self.begin_pos, self.end_pos)

    def __eq__(self, other):
        return isinstance(other, List) and self.children == other.children

    def __hash__(self):
        return 6563 + hash(tuple(self.children))

    def all(self):
        yield self
        for c in self.children:
            yield from c.all()


class Axis(Expression):
    def __init__(self, name, value, begin_pos=None, end_pos=None):
        Expression.__init__(self, value, begin_pos=begin_pos, end_pos=end_pos)
        self.name = name if name is not None else f"unnamed.{id(self)}"

    def __str__(self):
        return self.name if not self.is_unnamed else str(self.value)

    def __len__(self):
        return 1

    def __iter__(self):
        yield self

    def __deepcopy__(self):
        return Axis(self.name, self.value, self.begin_pos, self.end_pos)

    def __eq__(self, other):
        if not isinstance(other, Axis):
            return False
        if self.is_unnamed != other.is_unnamed:
            return False
        if self.value != other.value:
            return False
        if self.is_unnamed:
            return True
        else:
            return self.name == other.name

    def __hash__(self):
        return 9817234 + (hash(self.name) if not self.is_unnamed else 0) + hash(self.value)

    def all(self):
        yield self

    @property
    def is_unnamed(self):
        return self.name.startswith("unnamed.")


class Concatenation(Expression):
    @staticmethod
    def maybe(l, begin_pos=None, end_pos=None):
        if not isinstance(l, list):
            raise TypeError(f"Expected list, got {type(l)}")
        if len(l) == 1:
            return l[0]
        else:
            return Concatenation(l, begin_pos=begin_pos, end_pos=end_pos)

    def __init__(self, children, begin_pos=None, end_pos=None):
        if len(children) == 0:
            raise ValueError("Concatenation must have at least one child")
        Expression.__init__(
            self,
            np.sum([c.value for c in children]).astype("int32"),
            begin_pos=begin_pos,
            end_pos=end_pos,
        )
        self.children = children
        for c in children:
            if len(c) != 1:
                raise ValueError(
                    "Concatenation can only be used on expressions of length 1, but"
                    f"got expression '{c}'"
                )
            c.parent = self

    def __str__(self):
        return "+".join([str(c) for c in self.children])

    def __len__(self):
        return 1

    def __iter__(self):
        yield self

    def __deepcopy__(self):
        return Concatenation(
            [c.__deepcopy__() for c in self.children], self.begin_pos, self.end_pos
        )

    def __eq__(self, other):
        return isinstance(other, Concatenation) and self.children == other.children

    def __hash__(self):
        return 123 + hash(tuple(self.children))

    def all(self):
        yield self
        for c in self.children:
            yield from c.all()


class Marker(Expression):
    def __init__(self, inner, begin_pos=None, end_pos=None):
        if len(inner) == 0:
            raise ValueError("Marker cannot have empty list as child")
        Expression.__init__(self, inner.value, begin_pos=begin_pos, end_pos=end_pos)
        self.inner = inner
        inner.parent = self

    def __str__(self):
        return f"[{self.inner}]"

    def __len__(self):
        return len(self.inner)

    def __iter__(self):
        yield from self.inner

    def __deepcopy__(self):
        return Marker(self.inner.__deepcopy__(), self.begin_pos, self.end_pos)

    def __eq__(self, other):
        return isinstance(other, Marker) and self.inner == other.inner

    def __hash__(self):
        return 6433236 + hash(self.inner)

    def all(self):
        yield self
        yield from self.inner.all()


def solve(exprs1, exprs2, descs1, descs2, signature=None):
    exprs1 = list(exprs1)
    exprs2 = list(exprs2)
    if any(
        expr is not None and not isinstance(expr, stage2.Expression) for expr in exprs1 + exprs2
    ):
        raise ValueError("Can only expand stage2.Expression")
    if len(exprs1) != len(exprs2):
        raise ValueError("Number of expressions must be equal")

    equations = []

    symbolic_expr_values = {}
    for root in exprs1 + exprs2:
        if root is not None:
            for expr in root.all():
                symbolic_expr_values[id(expr)] = solver.Variable(
                    f"symbolic_expr_values[{id(expr)}]", str(expr)
                )

    # Add equations: Relations between expressions and their children
    for root in exprs1 + exprs2:
        if root is not None:
            for expr in root.all():
                if isinstance(expr, stage2.List):
                    equations.append((
                        solver.Product([symbolic_expr_values[id(c)] for c in expr.children]),
                        symbolic_expr_values[id(expr)],
                    ))
                elif isinstance(expr, stage2.Concatenation):
                    equations.append((
                        solver.Sum([symbolic_expr_values[id(c)] for c in expr.children]),
                        symbolic_expr_values[id(expr)],
                    ))
                elif isinstance(expr, stage2.Marker) or isinstance(expr, stage2.Composition):
                    equations.append((
                        symbolic_expr_values[id(expr)],
                        symbolic_expr_values[id(expr.inner)],
                    ))

    # Add equations: Same root values
    for root1, root2 in zip(exprs1, exprs2):
        if root1 is not None and root2 is not None:
            assert len(root1) == len(root2)
            for expr1, expr2 in zip(root1, root2):
                equations.append((
                    symbolic_expr_values[id(expr1)],
                    symbolic_expr_values[id(expr2)],
                ))

    # Add equations: Unnamed axes
    for root in exprs1 + exprs2:
        if root is not None:
            for expr in root.all():
                if isinstance(expr, stage2.UnnamedAxis):
                    equations.append((
                        symbolic_expr_values[id(expr)],
                        int(expr.value),
                    ))

    # Add equations: Multiple occurrences of the same named axis must have the same value
    sympy_axis_values = {}
    for root in exprs1 + exprs2:
        if root is not None:
            for axis in root.all():
                if isinstance(axis, stage2.NamedAxis):
                    if axis.name not in sympy_axis_values:
                        sympy_axis_values[axis.name] = solver.Variable(
                            f"sympy_axis_values[{axis.name}]", axis.name
                        )
                    equations.append((
                        symbolic_expr_values[id(axis)],
                        sympy_axis_values[axis.name],
                    ))

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
                for expr in root.all():
                    if isinstance(expr, stage2.NamedAxis):
                        if id(expr) not in axis_values:
                            failed_axes.add(expr)
        if len(failed_axes) > 0:
            failed_axes = sorted({str(x) for x in failed_axes})
            raise solver.SolveExceptionTooManySolutions(", ".join(failed_axes))

        # Raise exception on non-positive values
        failed_exprs = set()
        for root in exprs1 + exprs2:
            if root is not None:
                for expr in root.all():
                    if isinstance(expr, stage2.NamedAxis) and axis_values[id(expr)] <= 0:
                        failed_exprs.add(expr)
        if len(failed_exprs) > 0:
            raise solver.SolveExceptionNoSolution()

        return axis_values

    try:
        axis_values = solve(equations)
    except solver.SolveExceptionTooManySolutions as e:
        if e.message is None:
            raise einx.DimensionError(
                "Failed to uniquely determine the size of all axes in the expression. Please "
                "provide more constraints.",
                text=signature.text,
                constraints=[
                    (expr1, expr2)
                    for expr1, expr2 in zip(exprs1, exprs2)
                    if expr1 is not None and expr2 is not None
                ],
            ) from e
        else:
            axis_names = e.message.split(", ")
            raise einx.DimensionError(
                f"Failed to uniquely determine the size of the axes {', '.join(axis_names)}. "
                "Please provide more constraints.",
                text=signature.text,
                pos=signature.get_pos_for_axisnames(exprs1 + exprs2, axis_names),
                constraints=[
                    (expr1, expr2)
                    for expr1, expr2 in zip(exprs1, exprs2)
                    if expr1 is not None and expr2 is not None
                ],
            ) from e
    except solver.SolveExceptionNoSolution as e:
        raise einx.DimensionError(
            "Failed to determine the size of all axes in the expression under the given "
            "constraints.",
            text=signature.text,
            constraints=[
                (expr1, expr2)
                for expr1, expr2 in zip(exprs1, exprs2)
                if expr1 is not None and expr2 is not None
            ],
        ) from e

    # Map stage2 expressions to stage3 expressions
    def map(expr):
        if isinstance(expr, stage2.NamedAxis):
            assert id(expr) in axis_values and axis_values[id(expr)] > 0
            return Axis(expr.name, axis_values[id(expr)], expr.begin_pos, expr.end_pos)
        elif isinstance(expr, stage2.UnnamedAxis):
            assert id(expr) in axis_values and axis_values[id(expr)] > 0
            return Axis(None, axis_values[id(expr)], expr.begin_pos, expr.end_pos)
        elif isinstance(expr, stage2.List):
            return List([map(child) for child in expr.children], expr.begin_pos, expr.end_pos)
        elif isinstance(expr, stage2.Concatenation):
            return Concatenation(
                [map(child) for child in expr.children], expr.begin_pos, expr.end_pos
            )
        elif isinstance(expr, stage2.Marker):
            return Marker(map(expr.inner), expr.begin_pos, expr.end_pos)
        elif isinstance(expr, stage2.Composition):
            return Composition.maybe(map(expr.inner), expr.begin_pos, expr.end_pos)
        else:
            raise AssertionError(type(expr))

    exprs1 = [map(root) if root is not None else None for root in exprs1]
    exprs2 = [map(root) if root is not None else None for root in exprs2]

    return exprs1, exprs2


def expr_map(f):
    def outer(expr, *args, **kwargs):
        # Wrap the user function to return a list of expressions
        def f2(expr):
            t = f(expr, *args, **kwargs)
            if t is None:
                return None, expr_map.CONTINUE
            expr, signal = t

            if isinstance(expr, list) or expr is None:
                return expr, signal
            if isinstance(expr, List):
                return expr.children, signal
            elif isinstance(expr, Expression):
                return [expr], signal
            else:
                raise TypeError(f"Invalid return type {type(expr)}")

        return List.maybe(_expr_map(expr, f2))

    return outer


expr_map.CONTINUE = 1
expr_map.COPY_AND_STOP = 2
expr_map.REPLACE_AND_STOP = 3
expr_map.REPLACE_AND_CONTINUE = 4


def _expr_map(expr, f):
    exprs, signal = f(expr)
    if signal == expr_map.REPLACE_AND_STOP:
        assert isinstance(exprs, list)
        return exprs
    elif signal == expr_map.COPY_AND_STOP:
        return [expr.__deepcopy__()]
    elif signal == expr_map.REPLACE_AND_CONTINUE:
        return [c for expr in exprs for c in _expr_map(expr, f)]

    if isinstance(expr, Axis):
        return [expr.__deepcopy__()]
    elif isinstance(expr, Composition):
        return [Composition.maybe(List.maybe(_expr_map(expr.inner, f)))]
    elif isinstance(expr, List):
        return [c2 for c1 in expr.children for c2 in _expr_map(c1, f)]
    elif isinstance(expr, Concatenation):
        children = [List.maybe(_expr_map(c, f)) for c in expr.children]
        children = [c if len(c) > 0 else Axis(None, 1) for c in children]
        return [Concatenation(children)]
    elif isinstance(expr, Marker):
        x = _expr_map(expr.inner, f)
        if len(x) == 0:
            # Drop empty marker
            return []
        else:
            return [Marker(List.maybe(x))]
    else:
        raise TypeError(f"Invalid expression type {type(expr)}")


@expr_map
def decompose(expr):
    if isinstance(expr, Composition):
        return expr.inner, expr_map.REPLACE_AND_CONTINUE
    elif isinstance(expr, Concatenation):
        return None, expr_map.COPY_AND_STOP


@expr_map
def demark(expr):
    if isinstance(expr, Marker):
        return expr.inner, expr_map.REPLACE_AND_CONTINUE


@expr_map
def replace(expr, f):
    expr = f(expr)
    if expr is not None:
        return expr, expr_map.REPLACE_AND_STOP


@expr_map
def remove(expr, pred):
    if pred(expr):
        return [], expr_map.REPLACE_AND_STOP


def remove_unnamed_trivial_axes(expr):
    def is_concat_child(expr):  # Do not remove direct children of concatenations
        return expr.parent is not None and (
            isinstance(expr.parent, Concatenation)
            or (isinstance(expr.parent, Marker) and is_concat_child(expr.parent))
        )

    return remove(
        expr,
        lambda expr: isinstance(expr, Axis)
        and expr.is_unnamed
        and expr.value == 1
        and not is_concat_child(expr),
    )


@expr_map
def mark(expr, pred):
    if (
        not isinstance(expr, Marker)
        and (expr.parent is None or not isinstance(expr.parent, Marker))
        and pred(expr)
    ):
        return Marker(expr.__deepcopy__()), expr_map.REPLACE_AND_CONTINUE


def any_parent_is(expr, pred, include_self=True):
    if not include_self:
        if expr.parent is None:
            return False
        expr = expr.parent
    while expr is not None:
        if pred(expr):
            return True
        expr = expr.parent
    return False


def is_marked(expr):
    return any_parent_is(expr, lambda expr: isinstance(expr, Marker))


def is_at_root(expr):
    return not any_parent_is(expr, lambda expr: isinstance(expr, Composition))


def is_flat(expr):
    return all(
        not isinstance(expr, Composition) and not isinstance(expr, Concatenation)
        for expr in expr.all()
    )


def get_axes(expr):
    return [expr for expr in expr.all() if isinstance(expr, Axis)]


def get_named_axes(expr):
    return [expr for expr in expr.all() if isinstance(expr, Axis) and not expr.is_unnamed]


def _get_marked(expr):
    if isinstance(expr, Axis):
        return []
    elif isinstance(expr, Marker):
        return [expr.inner.__deepcopy__()]
    elif isinstance(expr, Concatenation):
        return [Concatenation.maybe([x for c in expr.children for x in _get_marked(c)])]
    elif isinstance(expr, Composition):
        return [Composition.maybe(List.maybe(_get_marked(expr.inner)))]
    elif isinstance(expr, List):
        return [List.maybe([x for c in expr.children for x in _get_marked(c)])]
    else:
        raise TypeError(f"Invalid expression type {type(expr)}")


def get_marked(expr):
    return List.maybe(_get_marked(expr))


def get_unmarked(expr):
    return remove(expr, lambda expr: is_marked(expr))
