_all = all
_map = map


def flatten(x, is_leaf=None):
    if is_leaf is not None and is_leaf(x):
        yield x
    elif isinstance(x, list | tuple):
        for x in x:
            yield from flatten(x, is_leaf=is_leaf)
    elif isinstance(x, dict):
        for x in x.items():
            yield from flatten(x, is_leaf=is_leaf)
    else:
        yield x


def map(func, *trees, is_leaf=None, assert_same_structure=False):
    if len(trees) == 0:
        raise ValueError("At least one tree is required")
    if is_leaf is not None and is_leaf(trees[0]):
        same_structure = _all(is_leaf(x) for x in trees[1:])
        if assert_same_structure:
            assert same_structure
        return func(*trees)
    elif isinstance(trees[0], list):
        same_structure = _all(isinstance(x, list) for x in trees[1:]) and len(set(_map(len, trees))) == 1
        if assert_same_structure:
            assert same_structure
        elif not same_structure:
            return func(*trees)
        return [map(func, *x, is_leaf=is_leaf, assert_same_structure=assert_same_structure) for x in zip(*trees, strict=False)]
    elif isinstance(trees[0], tuple):
        same_structure = _all(isinstance(x, tuple) for x in trees[1:]) and len(set(_map(len, trees))) == 1
        if assert_same_structure:
            assert same_structure
        elif not same_structure:
            return func(*trees)
        return tuple(map(func, *x, is_leaf=is_leaf, assert_same_structure=assert_same_structure) for x in zip(*trees, strict=False))
    elif isinstance(trees[0], dict):
        keys = trees[0].keys()
        same_structure = _all(isinstance(x, dict) for x in trees[1:]) and _all(x.keys() == keys for x in trees[1:])
        if assert_same_structure:
            assert same_structure
        elif not same_structure:
            return func(*trees)
        return {k: map(func, *[x[k] for x in trees], is_leaf=is_leaf, assert_same_structure=assert_same_structure) for k in trees[0]}
    else:
        return func(*trees)


def all(pred, *trees, is_leaf=None):
    bool_tree = map(pred, *trees, is_leaf=is_leaf)
    return _all(flatten(bool_tree))
