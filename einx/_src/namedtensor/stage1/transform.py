from .tree import *


def map(expr, _map, include_children):
    expr2 = _map(expr)
    if expr2 is not None:
        if include_children:
            expr = expr2
        else:
            return expr2

    if isinstance(expr, Axis):
        return expr.__deepcopy__()
    elif isinstance(expr, FlattenedAxis):
        inner = map(expr.inner, _map, include_children)
        return FlattenedAxis.create(inner, begin_pos=expr.begin_pos, end_pos=expr.end_pos)
    elif isinstance(expr, List):
        children = [map(c, _map, include_children) for c in expr.children]
        return List.create(children, begin_pos=expr.begin_pos, end_pos=expr.end_pos)
    elif isinstance(expr, ConcatenatedAxis):
        children = [map(c, _map, include_children) for c in expr.children]
        return ConcatenatedAxis.create(children, begin_pos=expr.begin_pos, end_pos=expr.end_pos)
    elif isinstance(expr, Brackets):
        inner = map(expr.inner, _map, include_children)
        return Brackets.create(inner, begin_pos=expr.begin_pos, end_pos=expr.end_pos)
    elif isinstance(expr, Ellipsis):
        inner = map(expr.inner, _map, include_children)
        return Ellipsis.create(inner, begin_pos=expr.begin_pos, end_pos=expr.end_pos, ellipsis_id=expr.ellipsis_id)
    elif isinstance(expr, Args):
        children = [map(c, _map, include_children) for c in expr.children]
        return Args(children, begin_pos=expr.begin_pos, end_pos=expr.end_pos)
    elif isinstance(expr, Op):
        children = [map(c, _map, include_children) for c in expr.children]
        return Op(children, begin_pos=expr.begin_pos, end_pos=expr.end_pos)
    else:
        raise ValueError(f"Invalid expression type {type(expr)}")


def remove(expr, pred, keep_children=False):
    if isinstance(pred, type):
        pred = lambda x, pred=pred: isinstance(x, pred)

    if keep_children:

        def _map(expr):
            if pred(expr):
                # Remove a node but keep its children
                if isinstance(expr, FlattenedAxis | Brackets | Ellipsis):
                    return expr.inner
                else:
                    raise ValueError(f"Cannot remove {type(expr)} with keep_children=True")
            else:
                return None

        return map(expr, _map, include_children=True)
    else:

        def _map(expr):
            if pred(expr):
                return List([])
            else:
                return None

        return map(expr, _map, include_children=False)


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


def is_in_brackets(expr):
    return any_parent_is(expr, lambda expr: isinstance(expr, Brackets))
