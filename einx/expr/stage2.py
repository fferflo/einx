from . import stage1, solver
import re, einx
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
        return List([c.__deepcopy__() for c in self.children], ellipsis_indices=self.ellipsis_indices)

    def all(self):
        yield self
        for c in self.children:
            yield from c.all()

class NamedAxis(Expression):
    def __init__(self, name, ellipsis_indices):
        Expression.__init__(self, ellipsis_indices)
        self.name = name

        postfix = ""
        for idx, num in self.ellipsis_indices:
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
                raise ValueError(f"Concatenation can only be used on expressions of length 1, but got expression '{c}'")
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
        return Concatenation([c.__deepcopy__() for c in self.children], ellipsis_indices=self.ellipsis_indices)

    def all(self):
        yield self
        for c in self.children:
            yield from c.all()

class Marker(Expression):
    def __init__(self, inner, ellipsis_indices):
        Expression.__init__(self, ellipsis_indices)
        self.inner = inner
        inner.parent = self

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



class SolveDepthException(Exception):
    def __init__(self, expressions, shapes, depths, message):
        self.expressions = expressions
        self.shapes = shapes
        self.depths = depths
        self.message = f"Failed to solve depths of expressions. {message}\nInput:\n"
        for expr, shape, depth in zip(expressions, shapes, depths):
            self.message += f"    '{expr}' has depth {depth}\n"
        super().__init__(self.message)

class SolveExpansionException(Exception):
    def __init__(self, expressions, shapes, depths, message):
        self.expressions = expressions
        self.shapes = shapes
        self.depths = depths
        self.message = f"Failed to solve expansion of expressions. {message}\nInput:\n"
        for expr, shape, depth in zip(expressions, shapes, depths):
            self.message += f"    '{expr}' has shape {einx.expr.util._to_str(shape)} at depth {depth}\n"
        super().__init__(self.message)

def solve(expressions, shapes, depths):
    if any(not isinstance(expr, stage1.Expression) for expr in expressions):
        raise ValueError("Can only expand stage1.Expression")
    if len(shapes) != len(expressions) or len(depths) != len(expressions):
        raise ValueError("Number of expressions, shapes and depths must be equal")
    depths = [d for d in depths]
    shapes = [s for s in shapes]

    # Semantic check: Cannot contain choices
    for root in expressions:
        if any(isinstance(expr, stage1.Choice) for expr in root.all()):
            raise ValueError(f"[|] Choice not allowed in expression '{root}'")

    # ##### 1. Find expression depths #####
    equations = []

    symbolic_expr_depths = {}
    for root in expressions:
        for expr in root.all():
            symbolic_expr_depths[id(expr)] = solver.Variable(str(id(expr)), str(expr))

    # Add equations: Depth relations between subexpressions
    for root in expressions:
        for expr in root.all():
            if isinstance(expr, stage1.Ellipsis):
                # Ellipsis increases depth by one
                equations.append((symbolic_expr_depths[id(expr)] + 1, symbolic_expr_depths[id(expr.inner)]))
            else:
                # All other expressions have the same depth as their children
                for child in expr.direct_children:
                    equations.append((symbolic_expr_depths[id(expr)], symbolic_expr_depths[id(child)]))

    # Add equations: Depth arguments
    for root, depth in zip(expressions, depths):
        if not depth is None:
            equations.append((symbolic_expr_depths[id(root)], depth))

    # Add equations: Multiple occurrences of the same named axis must have the same depth
    symbolic_axis_depths = {}
    for root in expressions:
        for axis in root.all():
            if isinstance(axis, stage1.NamedAxis):
                if not axis.name in symbolic_axis_depths:
                    symbolic_axis_depths[axis.name] = solver.Variable(axis.name, axis.name)
                equations.append((symbolic_expr_depths[id(axis)], symbolic_axis_depths[axis.name]))

    # Add equations: Ellipses with the same id must have the same depth
    symbolic_ellipsis_depths = {}
    for root in expressions:
        for ellipsis in root.all():
            if isinstance(ellipsis, stage1.Ellipsis):
                if not ellipsis.ellipsis_id in symbolic_ellipsis_depths:
                    symbolic_ellipsis_depths[ellipsis.ellipsis_id] = solver.Variable(ellipsis.ellipsis_id, str(ellipsis))
                equations.append((symbolic_expr_depths[id(ellipsis)], symbolic_ellipsis_depths[ellipsis.ellipsis_id]))

    # Solve
    try:
        expr_depths = solver.solve(equations)
    except solver.SolveException as e:
        raise SolveDepthException(expressions, shapes, depths, str(e))
    expr_depths = {int(k): int(v) for k, v in expr_depths.items() if not str(k) in symbolic_axis_depths}

    failed_exprs = set()
    for root in expressions:
        for expr in root.all():
            if not id(expr) in expr_depths:
                failed_exprs.add(str(expr))
    if len(failed_exprs) == 1:
        raise SolveDepthException(expressions, shapes, depths, f"Found no unique solution for '{failed_exprs.pop()}'")
    elif len(failed_exprs) > 1:
        raise SolveValueError(expressions, shapes, depths, f"Found no unique solutions for {failed_exprs}")

    for i, root in enumerate(expressions):
        # Add missing dimensions to shapes
        if not shapes[i] is None:
            assert len(shapes[i]) >= 1
            missing_depth = expr_depths[id(root)] - (len(shapes[i]) - 1)
            if missing_depth < 0:
                raise ValueError(f"Value passed for expression '{root}' has too many dimensions")
            if missing_depth > 0:
                shapes[i] = [None] * missing_depth + list(shapes[i])

        # Add missing ellipses around root expressions
        missing_depth = expr_depths[id(root)]
        if missing_depth > 0:
            for _ in range(missing_depth):
                expressions[i] = stage1.Ellipsis(expressions[i], expressions[i].begin_pos, expressions[i].end_pos)
                expr_depths[id(expressions[i])] = expr_depths[id(expressions[i].inner)] - 1

    # ##### 2. Find ellipsis expansions #####
    equations = []

    symbolic_expr_expansions = {}
    for root in expressions:
        for expr in root.all():
            for depth in range(expr_depths[id(expr)] + 1):
                key = (id(expr), depth)
                symbolic_expr_expansions[key] = solver.Variable(f"{id(expr)},{depth}", f"{expr} at depth {depth}")

    # Add equations: Expansion of an expression at depth d (less than own depth) is equal to the expansion of each child at depth d
    for root in expressions:
        for expr in root.all():
            for depth in range(expr_depths[id(expr)]):
                for child in expr.direct_children:
                    equations.append((symbolic_expr_expansions[(id(expr), depth)], symbolic_expr_expansions[(id(child), depth)]))

    # Add equations: Relations between expressions and their children
    for root in expressions:
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
                assert False, f"{expr}"
            equations.append((symbolic_expr_expansions[(id(expr), depth)], v))

    # Add equations: Expansions stored in "shapes"
    for expr, shape in zip(expressions, shapes):
        if not shape is None:
            for _ in range(len(shape) - 1):
                expr = expr.inner
            for depth in range(len(shape)):
                if not shape[depth] is None:
                    equations.append((symbolic_expr_expansions[(id(expr), depth)], int(shape[depth])))

    # Add equations: Multiple occurrences of the same named axis must have the same expansions
    symbolic_axis_expansions = {}
    for root in expressions:
        for axis in root.all():
            if isinstance(axis, stage1.NamedAxis):
                for depth in range(expr_depths[id(axis)] + 1):
                    if not axis.name in symbolic_axis_expansions:
                        symbolic_axis_expansions[(axis.name, depth)] = solver.Variable(f"{axis.name},{depth}", f"{axis.name} at depth {depth}")
                    equations.append((symbolic_expr_expansions[(id(axis), depth)], symbolic_axis_expansions[(axis.name, depth)]))

    # Add equations: Ellipses with the same id must have the same expansions
    symbolic_ellipsis_expansions = {}
    for root in expressions:
        for ellipsis in root.all():
            if isinstance(ellipsis, stage1.Ellipsis):
                for depth in range(expr_depths[id(ellipsis)] + 1):
                    if not ellipsis.ellipsis_id in symbolic_ellipsis_expansions:
                        symbolic_ellipsis_expansions[(ellipsis.ellipsis_id, depth)] = solver.Variable(f"{ellipsis.ellipsis_id},{depth}", f"{ellipsis} at depth {depth}")
                    equations.append((symbolic_expr_expansions[(id(ellipsis), depth)], symbolic_ellipsis_expansions[(ellipsis.ellipsis_id, depth)]))

    # Solve
    try:
        solutions = solver.solve(equations)
    except solver.SolveException as e:
        raise SolveExpansionException(expressions, shapes, depths, str(e))
    def to_key(k):
        return int(id_expr), int(depth)
    expansion_values = {}
    for k, v in solutions.items():
        id_expr, depth = str(k).split(",")
        try:
            id_expr = int(id_expr)
        except ValueError:
            continue
        depth = int(depth)
        expansion_values[(id_expr, depth)] = int(v)

    failed_exprs = set()
    for root in expressions:
        for expr in root.all():
            if not (id(root), expr_depths[id(root)]) in expansion_values:
                failed_exprs.add(str(expr))
    if len(failed_exprs) == 1:
        raise SolveExpansionException(expressions, shapes, depths, f"Found no unique solution for '{failed_exprs.pop()}'")
    elif len(failed_exprs) > 1:
        raise SolveExpansionException(expressions, shapes, depths, f"Found no unique solutions for {failed_exprs}")

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
            assert False
        elif isinstance(expr, stage1.UnnamedAxis):
            return expr.value
        elif isinstance(expr, stage1.Composition):
            return get_unnamed_value(expr.inner)
        elif isinstance(expr, stage1.Marker):
            return get_unnamed_value(expr.inner)
        elif isinstance(expr, stage1.Ellipsis):
            value = get_unnamed_value(expr.inner)
            if value != 1: # TODO: implement this
                raise NotImplementedError(f"Found unnamed and unexpanded ellipsis '{expr}'. We currently disallow this case, since it could can take on multiple values ('2...' could have values 2, 4, ...) that should be resolved in the solver and then checked to be consistent with these constraints.")
            return 1
        else:
            assert False, f"{expr}"

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
            return [Concatenation([List.maybe(map(c, ellipsis_indices=ellipsis_indices), ellipsis_indices=ellipsis_indices) for c in expr.children], ellipsis_indices=ellipsis_indices)]
        elif isinstance(expr, stage1.Composition):
            return [Composition(List.maybe(map(expr.inner, ellipsis_indices=ellipsis_indices), ellipsis_indices=ellipsis_indices), ellipsis_indices=ellipsis_indices)]
        elif isinstance(expr, stage1.Marker):
            return [Marker(List.maybe(map(expr.inner, ellipsis_indices=ellipsis_indices), ellipsis_indices=ellipsis_indices), ellipsis_indices=ellipsis_indices)]
        elif isinstance(expr, stage1.Ellipsis):
            key = (id(expr), expr_depths[id(expr)])
            if key in expansion_values:
                # Ellipsis is expanded
                expansion = expansion_values[key]
                if expansion < 0:
                    raise SolveExpansionException(expressions, shapes, depths, f"Ellipsis '{expr}' has negative expansion {expansion}")
                return [c for i in range(expansion) for c in map(expr.inner, ellipsis_indices=ellipsis_indices + [(i, expansion)])]
            else:
                # Ellipsis is not expanded
                if is_unnamed(expr):
                    # Contains no named axes -> convert to unnamed axis
                    return [UnnamedAxis(get_unnamed_value(expr), ellipsis_indices=ellipsis_indices)]
                else:
                    # Contains named axes -> convert to named axis
                    return [NamedAxis(str(expr), ellipsis_indices=ellipsis_indices)]
        else:
            assert False, f"{expr}"
    expressions = [List.maybe(map(root, ellipsis_indices=[]), ellipsis_indices=[]) for root in expressions]

    return expressions




def cse(expressions, cse_concat=True, verbose=False):
    if any(not isinstance(expr, Expression) for expr in expressions):
        raise TypeError("Can only perform common-subexpression-elimination on stage2.Expression")

    # Find possible expressions, identified by their string representation
    str_to_common_expr = defaultdict(list)
    for root in expressions:
        for expr in root.all():
            if not expr.parent is None:
                str_expr = str(expr)
                str_to_common_expr[str_expr].append([expr])

                if isinstance(expr, List):
                    for start_index in range(len(expr.children)):
                        for end_index in range(start_index, len(expr.children)):
                            children = expr.children[start_index:end_index + 1]
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
        for expr in expressions:
            for global_axis in expr.all():
                if isinstance(global_axis, NamedAxis) and global_axis.name in used_axis_names:
                    axes_used_only_in_this_subexpression = axes_used_only_in_this_subexpression and id(global_axis) in used_axis_ids

        if axes_used_only_in_this_subexpression:
            common_exprs.add(str_expr)

    common_exprs = [str_to_common_expr[k] for k in common_exprs] # list of common_expr(=list of exprlist)

    if verbose:
        print("CSE: Removed expressions with axes that are also used outside the expression")
        for v in common_exprs:
            print(f"    {[' '.join([str(y) for y in x]) for x in v]}")

    def remove_duplicates(common_expr):
        new_common_expr = []
        for exprlist1 in common_expr:
            is_duplicate = False
            for exprlist2 in new_common_expr:
                is_duplicate = is_duplicate or (len(exprlist1) == len(exprlist2) and all(id(expr1) == id(expr2) for expr1, expr2 in zip(exprlist1, exprlist2)))
            if not is_duplicate:
                new_common_expr.append(exprlist1)
        return new_common_expr
    common_exprs = [remove_duplicates(exprlists) for exprlists in common_exprs]

    if verbose:
        print("CSE: Removed duplicates")
        for v in common_exprs:
            print(f"    {[' '.join([str(y) for y in x]) for x in v]} {[[id(y) for y in x] for x in v]}")

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

    # Remove expressions with markers
    common_exprs = [common_expr for common_expr in common_exprs if not any(isinstance(expr, Marker) for exprlist in common_expr for expr in exprlist for expr in expr.all())]

    # Remove expressions that contain concatenations
    if not cse_concat:
        common_exprs = [common_expr for common_expr in common_exprs if not any(isinstance(expr, Concatenation) for exprlist in common_expr for expr in exprlist for expr in expr.all())]

    if verbose:
        print("CSE: Removed expressions with markers")
        for v in common_exprs:
            print(f"    {[' '.join([str(y) for y in x]) for x in v]}")

    # Remove expressions at root level with len > 1
    common_exprs = [common_expr for common_expr in common_exprs if not (is_at_root(common_expr[0][0]) and (len(common_expr[0]) > 1 or len(common_expr[0][0]) > 1))]

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
            return not child.parent is None and (id(child.parent) == id(parent) or any_is_parent_of(parent, child.parent))
    common_exprs = [common_expr for common_expr in common_exprs if not any(id(common_expr) != id(common_expr2) and any_is_parent_of(common_expr2, common_expr) for common_expr2 in common_exprs)]

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
                    if not exprlist_found is None:
                        break
                exprlist = exprlist_found

                if not exprlist is None:
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
            return [Concatenation([c2 for c1 in expr.children for c2 in replace(c1)], expr.ellipsis_indices)]
        elif isinstance(expr, Marker):
            return [Marker(List.maybe(replace(expr.inner), expr.ellipsis_indices), expr.ellipsis_indices)]
        elif isinstance(expr, Composition):
            return [Composition(List.maybe(replace(expr.inner), expr.ellipsis_indices), expr.ellipsis_indices)]
        else:
            assert False
    return [List.maybe(replace(expr), ellipsis_indices=[]) for expr in expressions]



def any_parent_is(expr, pred, include_self=True):
    if not include_self:
        if expr.parent is None:
            return False
        expr = expr.parent
    while not expr is None:
        if pred(expr):
            return True
        expr = expr.parent
    return False

def is_at_root(expr):
    return not any_parent_is(expr, lambda expr: isinstance(expr, Composition), include_self=False)