from . import stage1, solver
import re
import einx
import numpy as np
from collections import defaultdict


class Expression:
    def __init__(self, ellipsis_indices):
        self.ellipsis_indices = ellipsis_indices
        self.parent = None

    @property
    def depth(self):
        return len(self.ellipsis_indices)

    @property
    def shape(self):
        return tuple(i[1] for i in self.ellipsis_indices) + (len(self),)


class Composition(Expression):
    def __init__(self, inner, ellipsis_indices):
        Expression.__init__(self, ellipsis_indices)
        self.inner = inner
        inner.parent = self

    def __str__(self):
        return f"({self.inner})"

    def __len__(self):
        return 1

    def __iter__(self):
        yield self

    def __deepcopy__(self):
        return Composition(self.inner.__deepcopy__(), ellipsis_indices=self.ellipsis_indices)

    def all(self):
        yield self
        yield from self.inner.all()


class List(Expression):
    @staticmethod
    def maybe(l, *args, **kwargs):
        if len(l) == 1:
            return l[0]
        else:
            return List(l, *args, **kwargs)

    def __init__(self, children, ellipsis_indices):
        Expression.__init__(self, ellipsis_indices)
        self.children = children
        for c in children:
            c.parent = self

    def __str__(self):
        return " ".join([str(c) for c in self.children])

    def __len__(self):
        return sum(len(c) for c in self.children)

    def __iter__(self):
        for c in self.children:
            yield from c

    def __deepcopy__(self):
        return List(
            [c.__deepcopy__() for c in self.children], ellipsis_indices=self.ellipsis_indices
        )

    def all(self):
        yield self
        for c in self.children:
            yield from c.all()


class NamedAxis(Expression):
    def __init__(self, name, ellipsis_indices):
        Expression.__init__(self, ellipsis_indices)
        self.name = name

        postfix = ""
        for idx, _num in self.ellipsis_indices:
            postfix = postfix + "." + str(idx)
        if not self.name.endswith(postfix):
            self.name = self.name + postfix

    def __str__(self):
        return self.name

    def __len__(self):
        return 1

    def __iter__(self):
        yield self

    def __deepcopy__(self):
        return NamedAxis(self.name, ellipsis_indices=self.ellipsis_indices)

    def all(self):
        yield self


class UnnamedAxis(Expression):
    def __init__(self, value, ellipsis_indices):
        Expression.__init__(self, ellipsis_indices)
        self.value = value

    def __str__(self):
        return str(self.value)

    def __len__(self):
        return 1

    def __iter__(self):
        yield self

    def __deepcopy__(self):
        return UnnamedAxis(self.value, ellipsis_indices=self.ellipsis_indices)

    def all(self):
        yield self


class Concatenation(Expression):
    def __init__(self, children, ellipsis_indices):
        Expression.__init__(self, ellipsis_indices)
        for c in children:
            if len(c) != 1:
                raise ValueError(
                    "Concatenation can only be used on expressions of length 1, "
                    f"but got expression '{c}'"
                )
        self.children = children
        for c in children:
            c.parent = self

    def __str__(self):
        return "+".join([str(c) for c in self.children])

    def __len__(self):
        return 1

    def __iter__(self):
        yield self

    def __deepcopy__(self):
        return Concatenation(
            [c.__deepcopy__() for c in self.children], ellipsis_indices=self.ellipsis_indices
        )

    def all(self):
        yield self
        for c in self.children:
            yield from c.all()


class Marker(Expression):
    @staticmethod
    def maybe(inner, *args, **kwargs):
        if len(inner) == 0:
            return inner
        else:
            return Marker(inner, *args, **kwargs)

    def __init__(self, inner, ellipsis_indices):
        Expression.__init__(self, ellipsis_indices)
        self.inner = inner
        inner.parent = self
        assert len(inner) > 0

    def __str__(self):
        return f"[{self.inner}]"

    def __len__(self):
        return len(self.inner)

    def __iter__(self):
        yield from self.inner

    def __deepcopy__(self):
        return Marker(self.inner.__deepcopy__(), ellipsis_indices=self.ellipsis_indices)

    def all(self):
        yield self
        yield from self.inner.all()


class SolveDepthException(solver.SolveException):
    def __init__(self, exprs1, exprs2, expansions1, expansions2, depths1, depths2, message):
        assert (
            len({
                len(exprs1),
                len(exprs2),
                len(expansions1),
                len(expansions2),
                len(depths1),
                len(depths2),
            })
            == 1
        )
        self.exprs1 = exprs1
        self.exprs2 = exprs2
        self.expansions1 = expansions1
        self.expansions2 = expansions2
        self.depths1 = depths1
        self.depths2 = depths2
        message_in = message
        message = (
            "Failed to solve for the depth of axes, i.e. the number of outer ellipses.\n"
            "Equations:\n"
        )
        for expr1, expr2 in zip(exprs1, exprs2):
            if expr1 is not None and expr2 is not None:
                message += "    "
                message += f"{einx.expr.util._to_str(expr1)}"
                message += " = "
                message += f"{einx.expr.util._to_str(expr2)}"
                message += "\n"
        message += f"Reason: {message_in}"
        super().__init__(message)


class SolveExpansionException(solver.SolveException):
    def __init__(self, exprs1, exprs2, expansions1, expansions2, depths1, depths2, message):
        assert (
            len({
                len(exprs1),
                len(exprs2),
                len(expansions1),
                len(expansions2),
                len(depths1),
                len(depths2),
            })
            == 1
        )
        self.exprs1 = exprs1
        self.exprs2 = exprs2
        self.expansions1 = expansions1
        self.expansions2 = expansions2
        self.depths1 = depths1
        self.depths2 = depths2
        message_in = message
        message = "Failed to solve for the number of axes in the expressions.\nEquations:\n"
        for expr1, expr2 in zip(exprs1, exprs2):
            if expr1 is not None and expr2 is not None:
                message += "    "
                message += f"{einx.expr.util._to_str(expr1)}"
                message += " = "
                message += f"{einx.expr.util._to_str(expr2)}"
                message += "\n"
        message += f"Reason: {message_in}"
        super().__init__(message)


def solve(exprs1, exprs2, expansions1, expansions2, depths1, depths2):
    exprs1 = list(exprs1)
    exprs2 = list(exprs2)
    expansions1 = list(expansions1)
    expansions2 = list(expansions2)
    depths1 = list(depths1)
    depths2 = list(depths2)
    if any(
        expr is not None and not isinstance(expr, stage1.Expression) for expr in exprs1 + exprs2
    ):
        raise ValueError("Can only expand stage1.Expression")
    if (
        len({
            len(exprs1),
            len(exprs2),
            len(expansions1),
            len(expansions2),
            len(depths1),
            len(depths2),
        })
        != 1
    ):
        raise ValueError("Number of expressions, expansions and depths must be equal")

    # ##### 1. Find expression depths #####
    equations = []

    symbolic_expr_depths = {}
    for root in exprs1 + exprs2:
        if root is not None:
            for expr in root.all():
                symbolic_expr_depths[id(expr)] = solver.Variable(
                    f"symbolic_expr_depths[{id(expr)}]", str(expr)
                )

    # Add equations: Depth relations between subexpressions
    for root in exprs1 + exprs2:
        if root is not None:
            for expr in root.all():
                if isinstance(expr, stage1.Ellipsis):
                    # Ellipsis increases depth by one
                    equations.append((
                        symbolic_expr_depths[id(expr)] + 1,
                        symbolic_expr_depths[id(expr.inner)],
                    ))
                else:
                    # All other expressions have the same depth as their children
                    for child in expr.direct_children:
                        equations.append((
                            symbolic_expr_depths[id(expr)],
                            symbolic_expr_depths[id(child)],
                        ))

    # Add equations: Depth arguments
    for root, depth in zip(exprs1 + exprs2, depths1 + depths2):
        if root is not None and depth is not None:
            equations.append((symbolic_expr_depths[id(root)], depth))

    # Add equations: Root depths
    for root1, root2, expansion1, expansion2 in zip(exprs1, exprs2, expansions1, expansions2):
        if (
            root1 is not None
            and root2 is not None
            and expansion1 is not None
            and expansion2 is not None
        ):
            equations.append((
                symbolic_expr_depths[id(root1)] + len(expansion1),
                symbolic_expr_depths[id(root2)] + len(expansion2),
            ))

    # Add equations: Multiple occurrences of the same named axis must have the same depth
    symbolic_axis_depths = {}
    for root in exprs1 + exprs2:
        if root is not None:
            for axis in root.all():
                if isinstance(axis, stage1.NamedAxis):
                    if axis.name not in symbolic_axis_depths:
                        symbolic_axis_depths[axis.name] = solver.Variable(
                            f"symbolic_axis_depths[{axis.name}]", axis.name
                        )
                    equations.append((
                        symbolic_expr_depths[id(axis)],
                        symbolic_axis_depths[axis.name],
                    ))

    # Add equations: Ellipses with the same id must have the same depth
    symbolic_ellipsis_depths = {}
    for root in exprs1 + exprs2:
        if root is not None:
            for ellipsis in root.all():
                if isinstance(ellipsis, stage1.Ellipsis):
                    if ellipsis.ellipsis_id not in symbolic_ellipsis_depths:
                        symbolic_ellipsis_depths[ellipsis.ellipsis_id] = solver.Variable(
                            f"symbolic_ellipsis_depths[{ellipsis.ellipsis_id}]", str(ellipsis)
                        )
                    equations.append((
                        symbolic_expr_depths[id(ellipsis)],
                        symbolic_ellipsis_depths[ellipsis.ellipsis_id],
                    ))

    # Solve
    try:
        solutions = solver.solve(equations)
    except solver.SolveException as e:
        raise SolveDepthException(
            exprs1, exprs2, expansions1, expansions2, depths1, depths2, str(e)
        ) from e
    expr_depths = {}
    for k, v in solutions.items():
        if k.startswith("symbolic_expr_depths["):
            expr_depths[int(k[len("symbolic_expr_depths[") : -1])] = int(v)

    # Raise exception on missing depths
    failed_exprs = set()
    for root in exprs1 + exprs2:
        if root is not None:
            for expr in root.all():
                if id(expr) not in expr_depths:
                    failed_exprs.add(str(expr))
    if len(failed_exprs) > 0:
        raise SolveDepthException(
            exprs1,
            exprs2,
            expansions1,
            expansions2,
            depths1,
            depths2,
            f"Found no unique solutions for {failed_exprs}",
        )

    # Raise exception on negative depths
    failed_exprs = set()
    for root in exprs1 + exprs2:
        if root is not None:
            for expr in root.all():
                if expr_depths[id(expr)] < 0:
                    failed_exprs.add(str(expr))
    if len(failed_exprs) > 0:
        raise SolveDepthException(
            exprs1,
            exprs2,
            expansions1,
            expansions2,
            depths1,
            depths2,
            f"Got negative depths for {failed_exprs}",
        )

    for exprs, expansions, _depths in zip(
        [exprs1, exprs2], [expansions1, expansions2], [depths1, depths2]
    ):
        for i in range(len(exprs)):
            if exprs[i] is not None:
                missing_depth = expr_depths[id(exprs[i])]
                assert missing_depth >= 0

                # Add missing dimensions to expansions
                if expansions[i] is not None:
                    assert len(expansions[i]) >= 1
                    if missing_depth > 0:
                        expansions[i] = [None] * missing_depth + list(expansions[i])

                # Add missing ellipses around root expressions
                if missing_depth > 0:
                    for _ in range(missing_depth):
                        exprs[i] = stage1.Ellipsis(exprs[i], exprs[i].begin_pos, exprs[i].end_pos)
                        expr_depths[id(exprs[i])] = expr_depths[id(exprs[i].inner)] - 1

    # ##### 2. Find ellipsis expansions #####
    equations = []

    symbolic_expr_expansions = {}
    for root in exprs1 + exprs2:
        if root is not None:
            for expr in root.all():
                for depth in range(expr_depths[id(expr)] + 1):
                    key = (id(expr), depth)
                    symbolic_expr_expansions[key] = solver.Variable(
                        f"symbolic_expr_expansions[{id(expr)},{depth}]", f"{expr} at depth {depth}"
                    )

    # Add equations: Expansion of an expression at depth d (less than own depth)
    # is equal to the expansion of each child at depth d
    for root in exprs1 + exprs2:
        if root is not None:
            for expr in root.all():
                for depth in range(expr_depths[id(expr)]):
                    for child in expr.direct_children:
                        equations.append((
                            symbolic_expr_expansions[(id(expr), depth)],
                            symbolic_expr_expansions[(id(child), depth)],
                        ))

    # Add equations: Relations between expressions and their children
    for root in exprs1 + exprs2:
        if root is not None:
            for expr in root.all():
                depth = expr_depths[id(expr)]
                if isinstance(expr, stage1.List):
                    v = sum(symbolic_expr_expansions[(id(child), depth)] for child in expr.children)
                elif isinstance(expr, stage1.Concatenation):
                    v = 1
                elif isinstance(expr, stage1.NamedAxis):
                    v = 1
                elif isinstance(expr, stage1.UnnamedAxis):
                    v = 1
                elif isinstance(expr, stage1.Composition):
                    v = 1
                elif isinstance(expr, stage1.Marker):
                    v = symbolic_expr_expansions[(id(expr.inner), depth)]
                elif isinstance(expr, stage1.Ellipsis):
                    v = symbolic_expr_expansions[(id(expr.inner), depth)]
                else:
                    raise AssertionError(f"{expr}")
                equations.append((symbolic_expr_expansions[(id(expr), depth)], v))

    # Add equations: Expansions stored in "expansions"
    for expansion1, expansion2, expr1, expr2 in zip(expansions1, expansions2, exprs1, exprs2):
        if expansion1 is not None and expansion2 is not None:
            if len(expansion1) != len(expansion2) or any(
                e1 is not None and e2 is not None and e1 != e2
                for e1, e2 in zip(expansion1, expansion2)
            ):
                raise SolveExpansionException(
                    exprs1,
                    exprs2,
                    expansions1,
                    expansions2,
                    depths1,
                    depths2,
                    f"Expansion '{expansion1}' of expression '{expr1}' does not match expansion "
                    f"'{expansion2}' of expression '{expr2}'",
                )

        if expansion1 is not None and expansion2 is not None:
            expansion = [e1 if e1 is not None else e2 for e1, e2 in zip(expansion1, expansion2)]
        elif expansion1 is not None:
            expansion = expansion1
        elif expansion2 is not None:
            expansion = expansion2
        else:
            expansion = None

        if expansion is not None:
            for depth, e in enumerate(expansion):
                if e is not None:
                    if expr1 is not None and depth <= expr_depths[id(expr1)]:
                        equations.append((symbolic_expr_expansions[(id(expr1), depth)], int(e)))
                    if expr2 is not None and depth <= expr_depths[id(expr2)]:
                        equations.append((symbolic_expr_expansions[(id(expr2), depth)], int(e)))

    # Add equations: Multiple occurrences of the same named axis must have the same expansions
    symbolic_axis_expansions = {}
    for root in exprs1 + exprs2:
        if root is not None:
            for axis in root.all():
                if isinstance(axis, stage1.NamedAxis):
                    for depth in range(expr_depths[id(axis)] + 1):
                        if axis.name not in symbolic_axis_expansions:
                            symbolic_axis_expansions[(axis.name, depth)] = solver.Variable(
                                f"symbolic_axis_expansions[{axis.name},{depth}]",
                                f"{axis.name} at depth {depth}",
                            )
                        equations.append((
                            symbolic_expr_expansions[(id(axis), depth)],
                            symbolic_axis_expansions[(axis.name, depth)],
                        ))

    # Add equations: Ellipses with the same id must have the same expansions
    symbolic_ellipsis_expansions = {}
    for root in exprs1 + exprs2:
        if root is not None:
            for ellipsis in root.all():
                if isinstance(ellipsis, stage1.Ellipsis):
                    for depth in range(expr_depths[id(ellipsis)] + 1):
                        if ellipsis.ellipsis_id not in symbolic_ellipsis_expansions:
                            symbolic_ellipsis_expansions[(ellipsis.ellipsis_id, depth)] = (
                                solver.Variable(
                                    f"symbolic_ellipsis_expansions[{ellipsis.ellipsis_id},{depth}]",
                                    f"{ellipsis} at depth {depth}",
                                )
                            )
                        equations.append((
                            symbolic_expr_expansions[(id(ellipsis), depth)],
                            symbolic_ellipsis_expansions[(ellipsis.ellipsis_id, depth)],
                        ))

    # Add equations: Same root expansions
    for root1, root2 in zip(exprs1, exprs2):
        if root1 is not None and root2 is not None:
            assert expr_depths[id(root1)] == expr_depths[id(root2)]
            for depth in range(expr_depths[id(root1)] + 1):
                equations.append((
                    symbolic_expr_expansions[(id(root1), depth)],
                    symbolic_expr_expansions[(id(root2), depth)],
                ))

    # Solve
    try:
        solutions = solver.solve(equations)
    except solver.SolveException as e:
        raise SolveExpansionException(
            exprs1, exprs2, expansions1, expansions2, depths1, depths2, str(e)
        ) from e

    def to_key(k):
        return int(id_expr), int(depth)

    expansion_values = {}
    for k, v in solutions.items():
        if k.startswith("symbolic_expr_expansions["):
            k = k[len("symbolic_expr_expansions[") : -1]
            id_expr, depth = str(k).split(",")
            try:
                id_expr = int(id_expr)
            except ValueError:
                continue
            depth = int(depth)
            expansion_values[(id_expr, depth)] = int(v)

    failed_exprs = set()
    for root in exprs1 + exprs2:
        if root is not None:
            for expr in root.all():
                if (id(root), expr_depths[id(root)]) not in expansion_values:
                    failed_exprs.add(str(expr))
    if len(failed_exprs) == 1:
        raise SolveExpansionException(
            exprs1,
            exprs2,
            expansions1,
            expansions2,
            depths1,
            depths2,
            f"Found no unique solution for '{failed_exprs.pop()}'",
        )
    elif len(failed_exprs) > 1:
        raise SolveExpansionException(
            exprs1,
            exprs2,
            expansions1,
            expansions2,
            depths1,
            depths2,
            f"Found no unique solutions for {failed_exprs}",
        )

    def is_unnamed(expr):
        for expr in expr.all():
            if isinstance(expr, stage1.NamedAxis):
                return False
        return True

    def get_unnamed_value(expr):
        if isinstance(expr, stage1.List):
            return np.prod([get_unnamed_value(child) for child in expr.children]).astype("int")
        elif isinstance(expr, stage1.Concatenation):
            return np.sum([get_unnamed_value(child) for child in expr.children])
        elif isinstance(expr, stage1.NamedAxis):
            raise AssertionError()
        elif isinstance(expr, stage1.UnnamedAxis):
            return expr.value
        elif isinstance(expr, stage1.Composition):
            return get_unnamed_value(expr.inner)
        elif isinstance(expr, stage1.Marker):
            return get_unnamed_value(expr.inner)
        elif isinstance(expr, stage1.Ellipsis):
            value = get_unnamed_value(expr.inner)
            if value != 1:  # TODO: implement this
                raise NotImplementedError(
                    f"Found unnamed and unexpanded ellipsis '{expr}'. We currently disallow this "
                    "case, since it could can take on multiple values ('2...' could have values "
                    "2, 4, ...) that should be resolved in the solver and then checked to be "
                    "consistent with these constraints."
                )
            return 1
        else:
            raise AssertionError(f"{expr}")

    # Expand ellipses and map stage1 expressions to stage2 expressions
    def map(expr, ellipsis_indices):
        if isinstance(expr, list):
            return [c for expr in expr for c in map(expr, ellipsis_indices=ellipsis_indices)]
        elif isinstance(expr, stage1.NamedAxis):
            return [NamedAxis(expr.name, ellipsis_indices=ellipsis_indices)]
        elif isinstance(expr, stage1.UnnamedAxis):
            return [UnnamedAxis(expr.value, ellipsis_indices=ellipsis_indices)]
        elif isinstance(expr, stage1.List):
            return map(expr.children, ellipsis_indices=ellipsis_indices)
        elif isinstance(expr, stage1.Concatenation):
            return [
                Concatenation(
                    [
                        List.maybe(
                            map(c, ellipsis_indices=ellipsis_indices),
                            ellipsis_indices=ellipsis_indices,
                        )
                        for c in expr.children
                    ],
                    ellipsis_indices=ellipsis_indices,
                )
            ]
        elif isinstance(expr, stage1.Composition):
            return [
                Composition(
                    List.maybe(
                        map(expr.inner, ellipsis_indices=ellipsis_indices),
                        ellipsis_indices=ellipsis_indices,
                    ),
                    ellipsis_indices=ellipsis_indices,
                )
            ]
        elif isinstance(expr, stage1.Marker):
            return [
                Marker.maybe(
                    List.maybe(
                        map(expr.inner, ellipsis_indices=ellipsis_indices),
                        ellipsis_indices=ellipsis_indices,
                    ),
                    ellipsis_indices=ellipsis_indices,
                )
            ]
        elif isinstance(expr, stage1.Ellipsis):
            key = (id(expr), expr_depths[id(expr)])
            if key in expansion_values:
                # Ellipsis is expanded
                expansion = expansion_values[key]
                if expansion < 0:
                    raise SolveExpansionException(
                        exprs1,
                        exprs2,
                        expansions1,
                        expansions2,
                        depths1,
                        depths2,
                        f"Ellipsis '{expr}' has negative expansion {expansion}",
                    )
                return [
                    c
                    for i in range(expansion)
                    for c in map(expr.inner, ellipsis_indices=ellipsis_indices + [(i, expansion)])
                ]
            else:
                # Ellipsis is not expanded
                if is_unnamed(expr):
                    # Contains no named axes -> convert to unnamed axis
                    return [UnnamedAxis(get_unnamed_value(expr), ellipsis_indices=ellipsis_indices)]
                else:
                    # Contains named axes -> convert to named axis
                    return [NamedAxis(str(expr), ellipsis_indices=ellipsis_indices)]
        else:
            raise AssertionError(f"{expr}")

    exprs1 = [
        List.maybe(map(root, ellipsis_indices=[]), ellipsis_indices=[])
        if root is not None
        else None
        for root in exprs1
    ]
    exprs2 = [
        List.maybe(map(root, ellipsis_indices=[]), ellipsis_indices=[])
        if root is not None
        else None
        for root in exprs2
    ]

    return exprs1, exprs2


def cse(expressions, cse_concat=True, cse_in_markers=False, verbose=False):
    expressions = list(expressions)
    if any(expr is not None and not isinstance(expr, Expression) for expr in expressions):
        raise TypeError("Expected expressions to be of type Expression")

    # Find possible expressions, identified by their string representation
    str_to_common_expr = defaultdict(list)
    for root in expressions:
        if root is not None:
            for expr in root.all():
                if expr.parent is not None:
                    str_expr = str(expr)
                    str_to_common_expr[str_expr].append([expr])

                    if isinstance(expr, List):
                        for start_index in range(len(expr.children)):
                            for end_index in range(start_index, len(expr.children)):
                                children = expr.children[start_index : end_index + 1]
                                str_expr = " ".join([str(c) for c in children])
                                str_to_common_expr[str_expr].append(children)

    if verbose:
        print("CSE: All subexpressions")
        for k in str_to_common_expr.keys():
            print(f"    {k}")

    # Keep only expressions
    # 1. with at least one named axis
    # 2. where named axes are not also used outside the expression
    common_exprs = set()
    for str_expr in str_to_common_expr.keys():
        used_axis_ids = set()
        used_axis_names = set()
        for exprlist in str_to_common_expr[str_expr]:
            for expr in exprlist:
                for v in expr.all():
                    if isinstance(v, NamedAxis):
                        used_axis_ids.add(id(v))
                        used_axis_names.add(v.name)

        if len(used_axis_ids) == 0:
            continue

        axes_used_only_in_this_subexpression = True
        for root in expressions:
            if root is not None:
                for global_axis in root.all():
                    if isinstance(global_axis, NamedAxis) and global_axis.name in used_axis_names:
                        axes_used_only_in_this_subexpression = (
                            axes_used_only_in_this_subexpression
                            and id(global_axis) in used_axis_ids
                        )

        if axes_used_only_in_this_subexpression:
            common_exprs.add(str_expr)

    common_exprs = [
        str_to_common_expr[k] for k in common_exprs
    ]  # list of common_expr(=list of exprlist)

    if verbose:
        print("CSE: Removed expressions with axes that are also used outside the expression")
        for v in common_exprs:
            print(f"    {[' '.join([str(y) for y in x]) for x in v]}")

    def remove_duplicates(common_expr):
        new_common_expr = []
        for exprlist1 in common_expr:
            is_duplicate = False
            for exprlist2 in new_common_expr:
                is_duplicate = is_duplicate or (
                    len(exprlist1) == len(exprlist2)
                    and all(id(expr1) == id(expr2) for expr1, expr2 in zip(exprlist1, exprlist2))
                )
            if not is_duplicate:
                new_common_expr.append(exprlist1)
        return new_common_expr

    common_exprs = [remove_duplicates(exprlists) for exprlists in common_exprs]

    if verbose:
        print("CSE: Removed duplicates")
        for v in common_exprs:
            print(
                f"    {[' '.join([str(y) for y in x]) for x in v]} "
                f"{[[id(y) for y in x] for x in v]}"
            )

    # Remove singletons
    def is_singleton(expr):
        if isinstance(expr, list):
            return len(expr) == 1 and is_singleton(expr[0])
        elif isinstance(expr, List):
            return is_singleton(expr.children)
        elif isinstance(expr, NamedAxis):
            return True
        elif isinstance(expr, UnnamedAxis):
            return True
        elif isinstance(expr, Marker):
            return is_singleton(expr.inner)
        else:
            return False

    common_exprs = [common_expr for common_expr in common_exprs if not is_singleton(common_expr[0])]

    if verbose:
        print("CSE: Removed singletons")
        for v in common_exprs:
            print(f"    {[' '.join([str(y) for y in x]) for x in v]}")

    # Remove expressions with/ in markers
    if cse_in_markers:
        common_exprs = [
            common_expr
            for common_expr in common_exprs
            if not any(
                isinstance(expr, Marker)
                for exprlist in common_expr
                for expr in exprlist
                for expr in expr.all()
            )
        ]
    else:
        common_exprs = [
            common_expr
            for common_expr in common_exprs
            if not any(
                einx.expr.stage2.is_marked(expr)
                for exprlist in common_expr
                for expr in exprlist
                for expr in expr.all()
            )
        ]

    # Remove expressions that contain concatenations
    if not cse_concat:
        common_exprs = [
            common_expr
            for common_expr in common_exprs
            if not any(
                isinstance(expr, Concatenation)
                for exprlist in common_expr
                for expr in exprlist
                for expr in expr.all()
            )
        ]

    if verbose:
        print("CSE: Removed expressions with markers")
        for v in common_exprs:
            print(f"    {[' '.join([str(y) for y in x]) for x in v]}")

    # Remove expressions at root level with len > 1
    common_exprs = [
        common_expr
        for common_expr in common_exprs
        if not (
            is_at_root(common_expr[0][0])
            and (len(common_expr[0]) > 1 or len(common_expr[0][0]) > 1)
        )
    ]

    if verbose:
        print("CSE: Removed subexpressions of root with len > 1")
        for v in common_exprs:
            print(f"    {[' '.join([str(y) for y in x]) for x in v]}")

    # Remove subexpressions of subexpressions
    def any_is_parent_of(parent, child):
        if isinstance(parent, list):
            return any(any_is_parent_of(p, child) for p in parent)
        elif isinstance(child, list):
            return any(any_is_parent_of(parent, c) for c in child)
        else:
            return child.parent is not None and (
                id(child.parent) == id(parent) or any_is_parent_of(parent, child.parent)
            )

    common_exprs = [
        common_expr
        for common_expr in common_exprs
        if not any(
            id(common_expr) != id(common_expr2) and any_is_parent_of(common_expr2, common_expr)
            for common_expr2 in common_exprs
        )
    ]

    if verbose:
        print("CSE: Removed subexpressions of subexpressions")
        for v in common_exprs:
            print(f"    {[' '.join([str(y) for y in x]) for x in v]}")

    # All subexpressions have been found. Now replace them with new Axis objects.
    def replace(expr):
        if isinstance(expr, list) and len(expr) == 1:
            return replace(expr[0])
        if not isinstance(expr, list):
            for idx, common_expr in enumerate(common_exprs):
                for exprlist in common_expr:
                    if len(exprlist) == 1 and id(expr) == id(exprlist[0]):
                        return [NamedAxis(f"cse.{idx}", expr.ellipsis_indices)]

        if isinstance(expr, list):
            result = []
            i = 0
            while i < len(expr):
                # Check if a subexpression starts at position i
                exprlist_found = None
                for idx, common_expr in enumerate(common_exprs):
                    for exprlist in common_expr:
                        for j in range(len(exprlist)):
                            if i + j >= len(expr) or id(exprlist[j]) != id(expr[i + j]):
                                break
                        else:
                            exprlist_found = exprlist
                    if exprlist_found is not None:
                        break
                exprlist = exprlist_found

                if exprlist is not None:
                    assert len(exprlist) > 0
                    result.append(NamedAxis(f"cse.{idx}", exprlist[0].ellipsis_indices))
                    i += len(exprlist)
                else:
                    result.extend(replace(expr[i]))
                    i += 1

            return result
        elif isinstance(expr, NamedAxis):
            return [expr.__deepcopy__()]
        elif isinstance(expr, UnnamedAxis):
            return [expr.__deepcopy__()]
        elif isinstance(expr, List):
            return replace(expr.children)
        elif isinstance(expr, Concatenation):
            return [
                Concatenation(
                    [c2 for c1 in expr.children for c2 in replace(c1)], expr.ellipsis_indices
                )
            ]
        elif isinstance(expr, Marker):
            return [
                Marker.maybe(
                    List.maybe(replace(expr.inner), expr.ellipsis_indices), expr.ellipsis_indices
                )
            ]
        elif isinstance(expr, Composition):
            return [
                Composition(
                    List.maybe(replace(expr.inner), expr.ellipsis_indices), expr.ellipsis_indices
                )
            ]
        else:
            raise AssertionError()

    return [
        List.maybe(replace(root), ellipsis_indices=[]) if root is not None else None
        for root in expressions
    ]


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

    if isinstance(expr, NamedAxis):
        return [expr.__deepcopy__()]
    elif isinstance(expr, UnnamedAxis):
        return [expr.__deepcopy__()]
    elif isinstance(expr, Composition):
        return [Composition(List.maybe(_expr_map(expr.inner, f)))]
    elif isinstance(expr, List):
        return [c2 for c1 in expr.children for c2 in _expr_map(c1, f)]
    elif isinstance(expr, Concatenation):
        return [Concatenation([List.maybe(_expr_map(c, f)) for c in expr.children])]
    elif isinstance(expr, Marker):
        x = _expr_map(expr.inner, f)
        if len(x) == 0:
            # Drop empty marker
            return []
        else:
            return [Marker.maybe(List.maybe(x))]
    else:
        raise TypeError(f"Invalid expression type {type(expr)}")


@expr_map
def demark(expr):
    if isinstance(expr, Marker):
        return expr.inner, expr_map.REPLACE_AND_CONTINUE


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


def is_at_root(expr):
    return not any_parent_is(expr, lambda expr: isinstance(expr, Composition), include_self=False)


def is_marked(expr):
    return any_parent_is(expr, lambda expr: isinstance(expr, Marker))


def _get_marked(expr):
    if isinstance(expr, NamedAxis):
        return []
    elif isinstance(expr, UnnamedAxis):
        return []
    elif isinstance(expr, Marker):
        return [expr.inner.__deepcopy__()]
    elif isinstance(expr, Concatenation):
        return [Concatenation.maybe([x for c in expr.children for x in _get_marked(c)])]
    elif isinstance(expr, Composition):
        return [Composition(List.maybe(_get_marked(expr.inner)))]
    elif isinstance(expr, List):
        return [List.maybe([x for c in expr.children for x in _get_marked(c)])]
    else:
        raise TypeError(f"Invalid expression type {type(expr)}")


def get_marked(expr):
    return List.maybe(_get_marked(expr))


def get_unmarked(expr):
    return remove(expr, lambda expr: not is_marked(expr))


@expr_map
def replace(expr, f):
    expr = f(expr)
    if expr is not None:
        return expr, expr_map.REPLACE_AND_STOP


@expr_map
def remove(expr, pred):
    if pred(expr):
        return [], expr_map.REPLACE_AND_STOP
