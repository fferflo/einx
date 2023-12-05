# Avoid a hard jax dependency

def tree_map_with_key(func, *trees, key=()):
    if all(isinstance(tree, list) for tree in trees) and all(len(trees[0]) == len(tree) for tree in trees[1:]):
        return [tree_map_with_key(func, *elements, key=key + (i,)) for i, elements in enumerate(zip(*trees))]
    elif all(isinstance(tree, tuple) for tree in trees) and all(len(trees[0]) == len(tree) for tree in trees[1:]):
        return tuple(tree_map_with_key(func, *elements, key=key + (i,)) for i, elements in enumerate(zip(*trees)))
    elif all(isinstance(tree, dict) for tree in trees) and all(trees[0].keys() == tree.keys() for tree in trees[1:]):
        return {k: tree_map_with_key(func, *[tree[k] for tree in trees], key=key + (k,)) for k in trees[0]}
    else:
        return func(*trees, key=key)

def tree_map(func, *trees):
    return tree_map_with_key(lambda *xs, key: func(*xs), *trees)

def tree_flatten(x):
    if isinstance(x, (list, tuple)):
        for x in x:
            yield from tree_flatten(x)
    elif isinstance(x, dict):
        for x in x.items():
            yield from tree_flatten(x)
    else:
        yield x