# Avoid a hard jax dependency


def tree_map_with_key(func, *trees, key=(), is_leaf=None):
    if is_leaf is not None and is_leaf(key, *trees):
        return func(*trees, key=key)
    elif all(isinstance(tree, list) for tree in trees) and all(
        len(trees[0]) == len(tree) for tree in trees[1:]
    ):
        return [
            tree_map_with_key(func, *elements, key=key + (i,), is_leaf=is_leaf)
            for i, elements in enumerate(zip(*trees))
        ]
    elif all(isinstance(tree, tuple) for tree in trees) and all(
        len(trees[0]) == len(tree) for tree in trees[1:]
    ):
        return tuple(
            tree_map_with_key(func, *elements, key=key + (i,), is_leaf=is_leaf)
            for i, elements in enumerate(zip(*trees))
        )
    elif all(isinstance(tree, dict) for tree in trees) and all(
        trees[0].keys() == tree.keys() for tree in trees[1:]
    ):
        return {
            k: tree_map_with_key(
                func, *[tree[k] for tree in trees], key=key + (k,), is_leaf=is_leaf
            )
            for k in trees[0]
        }
    else:
        return func(*trees, key=key)


def tree_map(func, *trees, is_leaf=None):
    if is_leaf is not None and is_leaf(*trees):
        return func(*trees)
    elif all(isinstance(tree, list) for tree in trees) and all(
        len(trees[0]) == len(tree) for tree in trees[1:]
    ):
        return [
            tree_map(func, *elements, is_leaf=is_leaf) for i, elements in enumerate(zip(*trees))
        ]
    elif all(isinstance(tree, tuple) for tree in trees) and all(
        len(trees[0]) == len(tree) for tree in trees[1:]
    ):
        return tuple(
            tree_map(func, *elements, is_leaf=is_leaf) for i, elements in enumerate(zip(*trees))
        )
    elif all(isinstance(tree, dict) for tree in trees) and all(
        trees[0].keys() == tree.keys() for tree in trees[1:]
    ):
        return {k: tree_map(func, *[tree[k] for tree in trees], is_leaf=is_leaf) for k in trees[0]}
    else:
        return func(*trees)


def tree_flatten(x, is_leaf=None):
    if is_leaf is not None and is_leaf(x):
        yield x
    elif isinstance(x, (list, tuple)):
        for x in x:
            yield from tree_flatten(x, is_leaf=is_leaf)
    elif isinstance(x, dict):
        for x in x.items():
            yield from tree_flatten(x, is_leaf=is_leaf)
    else:
        yield x
