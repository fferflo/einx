import einx._src.tracer as tracer
import einx._src.util.pytree as pytree
from collections import defaultdict
import numpy as np


class Map:
    def __init__(self):
        self.id_to_usagenum = defaultdict(lambda: 0)

    def __getitem__(self, key):
        if id(key) in self.id_to_usagenum:
            return self.id_to_usagenum[id(key)]
        elif isinstance(key, tuple | list):
            return max(self[k] for k in key)
        elif isinstance(key, dict):
            return max(self[k] for k in list(key.keys()) + list(key.values()))
        else:
            raise KeyError(f"Key {key} not found in usage map.")


def get_usages(object):
    map = Map()
    done = set()

    def _recurse(x):
        if isinstance(x, str | int | float | np.integer | np.floating | bool) or x is None:
            return
        if id(x) in done:
            return
        done.add(id(x))

        map.id_to_usagenum[id(x)] += 1
        if isinstance(x, tracer.Tracer):
            if x.origin is not None:
                for input in x.origin.inputs:
                    _recurse(input)
                for output in pytree.flatten(x.origin.output):
                    _recurse(output)
        elif isinstance(x, list | tuple):
            for input in x:
                _recurse(input)
        elif isinstance(x, dict):
            for input in list(x.keys()) + list(x.values()):
                _recurse(input)
        elif isinstance(x, tracer.Graph):
            _recurse(x.output)

    _recurse(object)

    return map
