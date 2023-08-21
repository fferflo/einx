import functools, os, collections, sys, threading
from einx.expr import stage3
import numpy as np

def _hash(x):
    if isinstance(x, np.ndarray):
        x = x.reshape([-1]).tolist()
    if "torch" in sys.modules:
        import torch
        if isinstance(x, torch.Size):
            x = list(x)
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
    elif isinstance(x, stage3.Root):
        return stage3.cache_hash(x)
    else:
        return hash(x)

def lru_cache(inner):
    cache_size = int(os.environ.get("EINX_CACHE_SIZE", 1024))
    if cache_size > 0:
        print_cache_miss = str(os.environ.get("EINX_PRINT_CACHE_MISS", "false")).lower() in ["true", "yes", "1"]
        lock = threading.Lock()
        cache = collections.OrderedDict()
        def outer(*args, **kwargs):
            h = _hash((args, kwargs))
            with lock:
                if h in cache:
                    for k, v in cache[h]:
                        if k == (args, kwargs):
                            cache.move_to_end(h)
                            return v
            if print_cache_miss:
                print(f"einx: Cache miss on {inner.__name__} with args={args} kwargs={kwargs} hash={hash(inner)} cache_size={len(cache)}")
            value = inner(*args, **kwargs)
            with lock:
                if h in cache:
                    candidates = cache[h]
                else:
                    candidates = cache[h] = []
                    if len(cache) > cache_size:
                        cache.popitem(False)
                candidates.append(((args, kwargs), value))
            return value
        def cache_clear():
            with lock:
                cache.clear()
        outer.cache_clear = cache_clear
        return outer
    else:
        return inner
