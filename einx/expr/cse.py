from . import stage3
from collections import defaultdict
import numpy as np

class CommonSubexpression(stage3.Node):
    def __init__(self, children, cse_id, value, ellipsis_indices):
        stage3.Node.__init__(self, value, ellipsis_indices)
        self.children = children
        self.cse_id = cse_id
        for child in self.children:
            child.parent = self

    def expand(self):
        return stage3.expand(self.children)

    def __str__(self):
        return " ".join([str(x) for x in self.children])

    def traverse(self):
        yield self
        for x in self.children:
            for y in x.traverse():
                yield y

    def __eq__(self, other):
        return other.__class__ == CommonSubexpression and self.children == other.children

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(str(self))

    def copy(self):
        return CommonSubexpression(stage3.copy(self.children), self.cse_id, self.value, self.ellipsis_indices)

def mark_common_subexpressions(expressions, verbose=False):
    if any(not isinstance(expr, stage3.Root) for expr in expressions):
        raise ValueError("Expected stage3.Root expressions")

    # Find expression parents
    str_to_exprs = defaultdict(list)
    for expr in expressions:
        for expr in expr.traverse():
            if not isinstance(expr, stage3.Root):
                str_expr = str(expr)
                str_to_exprs[str_expr].append([expr])

                if isinstance(expr, stage3.Group):
                    for start_index in range(len(expr.unexpanded_children)):
                        for end_index in range(start_index, len(expr.unexpanded_children)):
                            children = expr.unexpanded_children[start_index:end_index + 1]
                            str_expr = " ".join([str(c) for c in children])
                            str_to_exprs[str_expr].append(children)

    if verbose:
        print("CSE1: All subexpressions")
        for k in str_to_exprs.keys():
            print(f"    {k}")

    # Keep only expressions with variables that are only used within the expression
    common_subexpressions = set()
    for str_expr in str_to_exprs.keys():
        used_variable_ids = set()
        used_variable_names = set()
        for expr in str_to_exprs[str_expr]:
            for expr in expr:
                for v in expr.traverse():
                    if isinstance(v, stage3.Variable):
                        used_variable_ids.add(id(v))
                        used_variable_names.add(v.name)

        variables_used_only_in_this_subexpression = True
        for expr in expressions:
            for global_v in expr.traverse():
                if isinstance(global_v, stage3.Variable) and global_v.name in used_variable_names:
                    variables_used_only_in_this_subexpression = variables_used_only_in_this_subexpression and id(global_v) in used_variable_ids

        if variables_used_only_in_this_subexpression:
            common_subexpressions.add(str_expr)

    common_subexpressions = [str_to_exprs[k] for k in common_subexpressions] # list of list of expr_list

    if verbose:
        print("CSE2: Removed expressions with variables that are used elsewhere")
        for v in common_subexpressions:
            print(f"    {[' '.join([str(y) for y in x]) for x in v]}")

    def remove_duplicates(expr_lists):
        new_expr_lists = []
        for expr_list1 in expr_lists:
            same = False
            for expr_list2 in new_expr_lists:
                same = same or (len(expr_list1) == len(expr_list2) and all(id(expr1) == id(expr2) for expr1, expr2 in zip(expr_list1, expr_list2)))
            if not same:
                new_expr_lists.append(expr_list1)
        return new_expr_lists
    common_subexpressions = [remove_duplicates(expr_lists) for expr_lists in common_subexpressions]

    if verbose:
        print("CSE3: Removed duplicates")
        for v in common_subexpressions:
            print(f"    {[' '.join([str(y) for y in x]) for x in v]} {[[id(y) for y in x] for x in v]}")

    # Remove singleton variables
    def is_singleton(expr):
        return isinstance(expr, list) and len(expr) == 1 and isinstance(expr[0], stage3.Variable)
    common_subexpressions = [exprs for exprs in common_subexpressions if not is_singleton(exprs[0])]

    if verbose:
        print("CSE4: Removed single variables")
        for v in common_subexpressions:
            print(f"    {[' '.join([str(y) for y in x]) for x in v]}")

    # Remove subexpressions of root if value is known
    def parent_is_root(expr_list):
        return any(isinstance(expr.parent, stage3.Root) for expr in expr_list)
    def all_values_known(expr_list):
        return all(not expr.value is None for expr in expr_list)
    def remove(expr_lists):
        return any(parent_is_root(expr_list) for expr_list in expr_lists) and all(all_values_known(expr_list) for expr_list in expr_lists)
    common_subexpressions = [exprs for exprs in common_subexpressions if not remove(exprs)]

    if verbose:
        print("CSE5: Removed subexpressions of root with known value")
        for v in common_subexpressions:
            print(f"    {[' '.join([str(y) for y in x]) for x in v]}")

    # Remove subexpressions of subexpressions
    def any_is_parent_of(parent, child):
        if isinstance(parent, list):
            return any(any_is_parent_of(p, child) for p in parent)
        elif isinstance(child, list):
            return any(any_is_parent_of(parent, c) for c in child)
        else:
            return not child.parent is None and (id(child.parent) == id(parent) or any_is_parent_of(parent, child.parent))
    common_subexpressions = [exprs for exprs in common_subexpressions if not any(any_is_parent_of(expr2, exprs) for expr2 in common_subexpressions)]

    if verbose:
        print("CSE6: Removed subexpressions of subexpressions")
        for v in common_subexpressions:
            print(f"    {[' '.join([str(y) for y in x]) for x in v]}")

    # Modify original expressions
    cse_ids = list(range(len(common_subexpressions)))
    def replace(expr):
        if not isinstance(expr, list):
            for cse_id, expr_lists in zip(cse_ids, common_subexpressions):
                for expr_list in expr_lists:
                    if len(expr_list) == 1 and id(expr) == id(expr_list[0]):
                        return CommonSubexpression(stage3.copy(expr), cse_id, expr.value, expr.ellipsis_indices)

        if isinstance(expr, list):
            result = []
            i = 0
            while i < len(expr):
                # Check if subexpression starts at this position in list
                expr_list_found = None
                for cse_id, expr_lists in zip(cse_ids, common_subexpressions):
                    for expr_list in expr_lists:
                        for j in range(len(expr_list)):
                            if i + j >= len(expr) or id(expr_list[j]) != id(expr[i + j]):
                                break
                        else:
                            expr_list_found = expr_list
                    if not expr_list_found is None:
                        break
                expr_list = expr_list_found

                if not expr_list is None:
                    assert len(expr_list) > 0
                    result.append(CommonSubexpression(stage3.copy(expr_list), cse_id, stage3.value(expr_list), expr_list[0].ellipsis_indices))
                    i += len(expr_list)
                else:
                    result.append(replace(expr[i]))
                    i += 1

            return result
        elif isinstance(expr, stage3.Variable):
            return stage3.copy(expr)
        elif isinstance(expr, stage3.Ellipsis):
            return stage3.Ellipsis(replace(expr.inner), expr.value, expr.ellipsis_indices)
        elif isinstance(expr, stage3.Group):
            return stage3.Group(replace(expr.unexpanded_children), expr.value, expr.ellipsis_indices, expr.front, expr.back)
        else:
            assert False
    return [stage3.Root(replace(expr.unexpanded_children), expr.value) for expr in expressions]
