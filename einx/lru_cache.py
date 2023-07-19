import functools, os, collections
import numpy as np

def _hash(x):
    if isinstance(x, list) or isinstance(x, tuple):
        h = 91724
        for v in x:
            h += _hash(v)
            h *= 18738
        return h
    elif isinstance(x, dict):
        h = 697123
        for v in x.values():
            h += _hash(v)
            h *= 9583
        return h
    else:
        return hash(x)

def lru_cache(inner):
    cache_size = int(os.environ.get("EINX_CACHE_SIZE", 1))
    if cache_size > 0:
        cache = collections.OrderedDict()
        def outer(*args, **kwargs):
            nonlocal cache
            h = _hash((args, kwargs))
            if h in cache:
                candidates = cache[h]
                for k, v in candidates:
                    if k == (args, kwargs):
                        cache.move_to_end(h)
                        return v
            else:
                candidates = []
                cache[h] = candidates
                if len(cache) > cache_size:
                    cache.popitem(False)
            # print("Cache miss")
            value = inner(*args, **kwargs)
            candidates.append(((args, kwargs), value))
            return value
        def cache_clear():
            cache.clear()
        outer.cache_clear = cache_clear
        return outer
    else:
        return inner
