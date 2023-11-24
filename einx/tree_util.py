# Avoid a hard jax dependency

def tree_map(func, x):
    if isinstance(x, list):
        return [tree_map(func, v) for v in x]
    elif isinstance(x, tuple):
        return tuple(tree_map(func, v) for v in x)
    elif isinstance(x, dict):
        return {k: tree_map(func, v) for k, v in x.items()}
    else:
        return func(x)

def tree_map_with_key(func, x, key=()):
    if isinstance(x, list):
        return [tree_map_with_key(func, v, key + (i,)) for i, v in enumerate(x)]
    elif isinstance(x, tuple):
        return tuple(tree_map_with_key(func, v, key + (i,)) for i, v in enumerate(x))
    elif isinstance(x, dict):
        return {k: tree_map_with_key(func, v, key + (k,)) for k, v in x.items()}
    else:
        return func(x, key)

def tree_flatten(x):
    if isinstance(x, (list, tuple)):
        for x in x:
            yield from tree_flatten(x)
    elif isinstance(x, dict):
        for x in x.items():
            yield from tree_flatten(x)
    else:
        yield x