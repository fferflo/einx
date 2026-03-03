from .tree import *


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


def is_at_root(expr):
    return not any_parent_is(expr, lambda expr: isinstance(expr, FlattenedAxis), include_self=False)
