from collections import defaultdict
from .tree import *
from .transform import *
import numpy as np


def cse(expressions, cse_concat=True, cse_in_brackets=False, verbose=False):
    expressions = list(expressions)
    if any(expr is not None and not isinstance(expr, Expression) for expr in expressions):
        raise TypeError("Expected expressions to be of type Expression")

    # Find possible expressions, identified by their string representation
    str_to_common_expr = defaultdict(list)
    for root in expressions:
        if root is not None:
            for expr in root.nodes():
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
    # 1. with at least one axis
    # 2. where axes are not also used outside the expression
    common_exprs = set()
    for str_expr in str_to_common_expr.keys():
        used_axis_ids = set()
        used_axis_names = set()
        for exprlist in str_to_common_expr[str_expr]:
            for expr in exprlist:
                for v in expr.nodes():
                    if isinstance(v, Axis):
                        used_axis_ids.add(id(v))
                        used_axis_names.add(v.name)

        if len(used_axis_ids) == 0:
            continue

        axes_used_only_in_this_subexpression = True
        for root in expressions:
            if root is not None:
                for global_axis in root.nodes():
                    if isinstance(global_axis, Axis) and global_axis.name in used_axis_names:
                        axes_used_only_in_this_subexpression = axes_used_only_in_this_subexpression and id(global_axis) in used_axis_ids

        if axes_used_only_in_this_subexpression:
            common_exprs.add(str_expr)

    common_exprs = [str_to_common_expr[k] for k in common_exprs]  # list of common_expr(=list of exprlist)

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
                    len(exprlist1) == len(exprlist2) and all(id(expr1) == id(expr2) for expr1, expr2 in zip(exprlist1, exprlist2, strict=False))
                )
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
        elif isinstance(expr, Axis):
            return True
        elif isinstance(expr, Brackets):
            return is_singleton(expr.inner)
        else:
            return False

    common_exprs = [common_expr for common_expr in common_exprs if not is_singleton(common_expr[0])]

    if verbose:
        print("CSE: Removed singletons")
        for v in common_exprs:
            print(f"    {[' '.join([str(y) for y in x]) for x in v]}")

    # Remove expressions with/ in markers
    if cse_in_brackets:
        common_exprs = [
            common_expr
            for common_expr in common_exprs
            if not any(isinstance(expr, Brackets) for exprlist in common_expr for expr in exprlist for expr in expr.nodes())
        ]
    else:
        common_exprs = [
            common_expr
            for common_expr in common_exprs
            if not any(is_in_brackets(expr) for exprlist in common_expr for expr in exprlist for expr in expr.nodes())
        ]

    # Remove expressions that contain concatenations
    if not cse_concat:
        common_exprs = [
            common_expr
            for common_expr in common_exprs
            if not any(isinstance(expr, ConcatenatedAxis) for exprlist in common_expr for expr in exprlist for expr in expr.nodes())
        ]

    if verbose:
        print("CSE: Removed expressions with markers")
        for v in common_exprs:
            print(f"    {[' '.join([str(y) for y in x]) for x in v]}")

    # Remove expressions at root level with len > 1
    def get_len(x):
        if isinstance(x, list):
            return len(x)
        else:
            return x.ndim

    common_exprs = [
        common_expr for common_expr in common_exprs if not (is_at_root(common_expr[0][0]) and (get_len(common_expr[0]) > 1 or get_len(common_expr[0][0]) > 1))
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
            return child.parent is not None and (id(child.parent) == id(parent) or any_is_parent_of(parent, child.parent))

    common_exprs = [
        common_expr
        for common_expr in common_exprs
        if not any(id(common_expr) != id(common_expr2) and any_is_parent_of(common_expr2, common_expr) for common_expr2 in common_exprs)
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
                        return [Axis(f"cse.{idx}", expr.value, expr.ellipsis_indices, begin_pos=expr.begin_pos, end_pos=expr.end_pos)]

        if isinstance(expr, list):
            result = []
            i = 0
            while i < len(expr):
                # Check if a subexpression starts at position i
                exprlist_found = None
                for idx, common_expr in enumerate(common_exprs):  # noqa: B007
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
                    values = [e.value for e in exprlist]
                    if any(v is None for v in values):
                        value = None
                    else:
                        value = np.prod(values)
                    result.append(Axis(f"cse.{idx}", value, exprlist[0].ellipsis_indices, begin_pos=exprlist[0].begin_pos, end_pos=exprlist[-1].end_pos))
                    i += len(exprlist)
                else:
                    result.extend(replace(expr[i]))
                    i += 1

            return result
        elif isinstance(expr, Axis):
            return [expr.__deepcopy__()]
        elif isinstance(expr, List):
            return replace(expr.children)
        elif isinstance(expr, ConcatenatedAxis):
            return [
                ConcatenatedAxis.create(
                    [c2 for c1 in expr.children for c2 in replace(c1)], expr.ellipsis_indices, begin_pos=expr.begin_pos, end_pos=expr.end_pos
                )
            ]
        elif isinstance(expr, Brackets):
            return [
                Brackets.create(List.create(replace(expr.inner), expr.ellipsis_indices), expr.ellipsis_indices, begin_pos=expr.begin_pos, end_pos=expr.end_pos)
            ]
        elif isinstance(expr, FlattenedAxis):
            return [
                FlattenedAxis.create(
                    List.create(replace(expr.inner), expr.ellipsis_indices), expr.ellipsis_indices, begin_pos=expr.begin_pos, end_pos=expr.end_pos
                )
            ]
        else:
            raise AssertionError()

    return [List.create(replace(root), ellipsis_indices=[]) if root is not None else None for root in expressions]
