import einx._src.tracer as tracer
import importlib
import operator
import builtins
import numpy as np

from einx._src.util import pytree

name_to_operator = {"+": operator.add, "*": operator.mul, "-": operator.sub, "==": operator.eq, "!=": operator.ne}


class CompilationCache:
    def __init__(self):
        self.tracerid_to_concrete = {}
        self.enter_num = 0

    def __enter__(self):
        assert self.enter_num >= 0
        if self.enter_num == 0:
            self.tracerid_to_concrete.clear()
        self.enter_num += 1

    def __exit__(self, *args):
        self.enter_num -= 1
        if self.enter_num == 0:
            self.tracerid_to_concrete.clear()
        assert self.enter_num >= 0

    def _get_new(self, x):
        if isinstance(x, tracer.Graph):
            return CompiledGraph(x, self)
        assert isinstance(x, tracer.Tracer) and x.origin is not None

        for input in x.origin.inputs:
            self.__getitem__(input)

        if isinstance(x.origin, tracer.signature.python.GetAttr):
            return getattr(self.__getitem__(x.origin.obj), x.origin.key)
        elif isinstance(x.origin, tracer.signature.python.GetItem):
            return self.__getitem__(x.origin.obj)[self.__getitem__(x.origin.key)]
        elif isinstance(x.origin, tracer.signature.python.Call):
            func = self.__getitem__(x.origin.function)
            args = [self.__getitem__(arg) for arg in x.origin.args]
            kwargs = {k: self.__getitem__(v) for k, v in x.origin.kwargs.items()}
            return func(*args, **kwargs)
        elif isinstance(x.origin, tracer.signature.python.Import):
            if x.origin.from_ is None:
                module = x.origin.import_
                return importlib.import_module(module)
            else:
                module = x.origin.from_
                module = importlib.import_module(module)
                return getattr(module, x.origin.import_)
        elif isinstance(x.origin, tracer.signature.python.OperatorApplication):
            return name_to_operator[x.origin.operator](*[self.__getitem__(arg) for arg in x.origin.operands])
        elif isinstance(x.origin, tracer.Cast):
            concrete_output = self.__getitem__(x.origin.input)
            pytree.map(self.__setitem__, x.origin.output, concrete_output)
            return self[x]
        elif isinstance(x.origin, tracer.signature.python.Assert):
            condition = self.__getitem__(x.origin.condition)
            assert condition, x.origin.message
            return self.__getitem__(x.origin.xs)
        elif isinstance(x.origin, tracer.signature.python.Builtin):
            return getattr(builtins, x.origin.name)
        elif isinstance(x.origin, tracer.signature.python.Constant):
            return x.origin.value
        else:
            raise NotImplementedError(f"Unsupported operation: {type(x.origin)}")

    def __getitem__(self, x):
        if isinstance(x, str | int | float | np.integer | np.floating | bool) or x is None:
            return x
        if isinstance(x, list):
            return [self.__getitem__(item) for item in x]
        elif isinstance(x, tuple):
            return tuple(self.__getitem__(item) for item in x)
        elif isinstance(x, dict):
            return {self.__getitem__(k): self.__getitem__(v) for k, v in x.items()}
        elif id(x) in self.tracerid_to_concrete:
            return self.tracerid_to_concrete[id(x)]
        else:
            concrete = self._get_new(x)
            assert id(concrete) not in self.tracerid_to_concrete
            self.tracerid_to_concrete[id(x)] = concrete
            return concrete

    def __setitem__(self, x, concrete):
        assert isinstance(x, tracer.Tracer)
        assert id(x) not in self.tracerid_to_concrete
        self.tracerid_to_concrete[id(x)] = concrete


class CompiledGraph:
    def __init__(self, graph, cache):
        self.graph = graph
        self.cache = cache

    def __call__(self, *args):
        if len(args) != len(self.graph.inputs):
            raise ValueError(f"Expected {len(self.graph.inputs)} arguments, got {len(args)}")
        with self.cache:
            for input, concrete in zip(self.graph.inputs, args, strict=False):
                self.cache[input] = concrete
            return self.cache[self.graph.output]


def compile(object, return_code=False):
    function = CompilationCache()[object]
    if return_code:
        return function, None
    else:
        return function
