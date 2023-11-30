from functools import partial
import numpy as np
import operator, einx

class Tracer:
    def __init__(self, shape):
        if isinstance(shape, np.ndarray):
            shape = tuple(shape.tolist())
        elif isinstance(shape, list):
            shape = tuple(shape)
        self.shape = shape

    def compute(self, backend, input_values, concrete_values):
        if id(self) in concrete_values:
            concrete_value = concrete_values[id(self)]
        else:
            concrete_value = self._compute(backend, input_values, concrete_values)
            concrete_values[id(self)] = concrete_value
        assert not isinstance(concrete_value, Tracer) or backend == einx.backend.tracer, f"Got tracer value {concrete_value} for backend {backend} with op {self} {self.op}, args {self.args}, kwargs {self.kwargs}"
        return concrete_value

    def __getitem__(self, key):
        if self.shape is None:
            shape = None
        else:
            if isinstance(key, (int, slice)):
                slices = (key,)
            else:
                slices = key
            if len(slices) > len(self.shape):
                raise ValueError(f"Too many indices for tensor of dimension {len(self.shape)}")

            shape = []
            for axis, key2 in zip(self.shape[:len(slices)], slices):
                if isinstance(key2, int):
                    pass
                elif isinstance(key2, slice):
                    start, stop, step = key2.indices(axis)
                    shape.append((stop - start) // step)
                else:
                    raise TypeError(f"Invalid key type: {type(slice)}")

        return Op(operator.getitem, args=[self, key], shape=shape)

    def __add__(self, other):
        return elementwise(self, other, op="add")

    def __radd__(self, other):
        return elementwise(other, self, op="add")

    def __sub__(self, other):
        return elementwise(self, other, op="subtract")

    def __rsub__(self, other):
        return elementwise(other, self, op="subtract")

    def __mul__(self, other):
        return elementwise(self, other, op="multiply")

    def __rmul__(self, other):
        return elementwise(other, self, op="multiply")

    def __truediv__(self, other):
        return elementwise(self, other, op="true_divide")

    def __rtruediv__(self, other):
        return elementwise(other, self, op="true_divide")

    def __floordiv__(self, other):
        return elementwise(self, other, op="floor_divide")

    def __rfloordiv__(self, other):
        return elementwise(other, self, op="floor_divide")

    def __div__(self, other):
        return elementwise(self, other, op="divide")

    def __rdiv__(self, other):
        return elementwise(other, self, op="divide")

    def __and__(self, other):
        return elementwise(self, other, op="logical_and")

    def __rand__(self, other):
        return elementwise(other, self, op="logical_and")

    def __or__(self, other):
        return elementwise(self, other, op="logical_or")

    def __ror__(self, other):
        return elementwise(other, self, op="logical_or")

    def __lt__(self, other):
        return elementwise(self, other, op="less")

    def __le__(self, other):
        return elementwise(self, other, op="less_equal")

    def __gt__(self, other):
        return elementwise(self, other, op="greater")

    def __ge__(self, other):
        return elementwise(self, other, op="greater_equal")

    def __eq__(self, other):
        return elementwise(self, other, op="equal")

    def __ne__(self, other):
        return elementwise(self, other, op="not_equal")

class Input(Tracer):
    def __init__(self, key, shape):
        super().__init__(shape)
        self.key = key

    def _compute(self, backend, input_values, concrete_values):
        return input_values[self.key]

class Constant(Tracer):
    def __init__(self, value):
        value = np.asarray(value)
        super().__init__(value.shape)
        self.value = value

    def _compute(self, backend, input_values, concrete_values):
        return backend.to_tensor(self.value)

class Op(Tracer):
    def __init__(self, op, args, kwargs={}, shape=None, pass_backend=False):
        if op is None:
            raise TypeError("op cannot be None")
        super().__init__(shape)
        self.op = op
        self.args = args
        self.kwargs = kwargs
        self.pass_backend = pass_backend
        assert not "backend" in self.kwargs

    def _compute(self, backend, input_values, concrete_values):
        def map(tensor):
            if isinstance(tensor, Tracer):
                return tensor.compute(backend, input_values, concrete_values)
            else:
                return tensor
        args, kwargs = einx.tree_util.tree_map(map, (self.args, self.kwargs))

        if backend == einx.backend.tracer:
            return Op(self.op, args=args, kwargs=kwargs, shape=self.shape, pass_backend=self.pass_backend)
        else:
            if self.pass_backend:
                assert not "backend" in kwargs
                kwargs["backend"] = backend
            if isinstance(self.op, str):
                op = backend
                for name in self.op.split("."):
                    op = getattr(op, name)
            else:
                op = self.op
            return op(*args, **kwargs)

def _to_str(x, names, lines):
    if isinstance(x, Input):
        if id(x) in names:
            return names[id(x)]
        name = names[id(x)] = f"I{len([n for n in names.values() if n.startswith('I')])}"
        return name
    elif isinstance(x, Constant):
        return _to_str(x.value, names, lines)
    elif isinstance(x, Op):
        if id(x) in names:
            return names[id(x)]
        name = names[id(x)] = f"X{len(names)}"

        args = ", ".join([_to_str(a, names, lines) for a in x.args] + [f"{k}={_to_str(v, names, lines)}" for k, v in x.kwargs.items()])
        op = x.op.__name__ if "__name__" in dir(x.op) else f"{x.op}"
        line = (name, f"{op}(" + args + ")")
        lines.append(line)

        return name
    elif isinstance(x, str):
        return f"\"{x}\""
    elif isinstance(x, tuple):
        return "(" + ", ".join(_to_str(a, names, lines) for a in x) + ")"
    elif isinstance(x, list):
        return "[" + ", ".join(_to_str(a, names, lines) for a in x) + "]"
    elif isinstance(x, dict):
        return "{" + ", ".join(f"{k}: {_to_str(v, names, lines)}" for k, v in x.items()) + "}"
    elif __name__ in dir(x):
        return x.__name__
    else:
        return str(x)

class Graph:
    def __init__(self, output, name, args, kwargs):
        self.output = output
        self.name = name
        self.args = args
        self.kwargs = kwargs

    def __call__(self, *args, backend=None, **kwargs):
        if backend is None:
            backend = einx.backend.get(list(einx.tree_util.tree_flatten((args, kwargs))))

        input_values = {}
        def map(x, key):
            input_values[key] = x
        einx.tree_util.tree_map_with_key(map, args)
        einx.tree_util.tree_map_with_key(map, kwargs)

        concrete_values = {}
        map = lambda x: x.compute(backend, input_values, concrete_values) if isinstance(x, Tracer) else x
        concrete_output = einx.tree_util.tree_map(map, self.output)

        return concrete_output

    def __str__(self):
        names = {}
        lines = []

        string = f"Graph {self.name}(" + ", ".join([_to_str(a, names, lines) for a in self.args] + [f"{k}={_to_str(v, names, lines)}" for k, v in self.kwargs.items()]) + "):"

        _to_str(self.output, names, lines)

        max_name_len = max(len(name) for name, _ in lines)
        lines = [f"{name:<{max_name_len}} := {line}" for name, line in lines]
        for line in lines:
            string += "\n    " + line
        string += f"\n    return {_to_str(self.output, names, lines)}"
        return string




class vmapped_op:
    def __init__(self, op, in_axes, out_axes):
        self.op = op
        self.in_axes = in_axes
        self.out_axes = out_axes

    def to_backend(self, backend):
        if isinstance(self.op, vmapped_op):
            op = self.op.to_backend(backend)
        else:
            op = self.op
        return backend.vmap(op, in_axes=self.in_axes, out_axes=self.out_axes)

    def __call__(self, *args, **kwargs):
        def inner(*args, backend, **kwargs):
            return self.to_backend(backend)(*args, **kwargs)
        inner.__name__ = self.__name__
        outputs = Op(inner, args=args, kwargs=kwargs, shape=None, pass_backend=True)
        return tuple(outputs[i] for i in range(len(self.out_axes)))

    @property
    def __name__(self):
        return f"vmap({self.op.__name__ if '__name__' in dir(self.op) else str(self.op)}, in_axes={self.in_axes}, out_axes={self.out_axes})"


def elementwise(*args, op):
    shape = None
    for a in args:
        if "shape" in dir(a):
            if shape is None:
                shape = a.shape
            else:
                shape2 = a.shape
                while len(shape) < len(shape2):
                    shape = (1,) + shape
                while len(shape2) < len(shape):
                    shape2 = (1,) + shape2
                shape = np.maximum(shape, shape2)
    return Op(op, args=args, shape=shape)

def reduce(tensor, axis, keepdims=False, op=None):
    axes = [axis] if isinstance(axis, int) else axis
    shape = list(tensor.shape)
    if keepdims:
        for a in axes:
            shape[a] = 1
    else:
        for a in reversed(sorted(axes)):
            del shape[a]
    return Op(op, args=[tensor, axis], kwargs={"keepdims": keepdims}, shape=tuple(shape))

class tracer:
    Input = Input
    Constant = Constant
    Op = Op
    Graph = Graph

    @staticmethod
    def to_tensor(tensor):
        if isinstance(tensor, Tracer):
            return tensor
        else:
            return Constant(tensor)

    tensor = Tracer
    name = "tracer"

    def cast(tensor, dtype):
        return Op("cast", args=[tensor], kwargs={"dtype": dtype}, shape=tensor.shape)
    def reshape(tensor, shape):
        return Op("reshape", args=[tensor, shape], shape=shape)

    def transpose(tensor, perm):
        shape = [tensor.shape[i] for i in perm]
        return Op("transpose", args=[tensor, perm], shape=shape)

    def broadcast_to(tensor, shape):
        return Op("broadcast_to", args=[tensor, shape], shape=shape)

    def einsum(eq, *tensors):
        exprs = eq.split("->")[0].split(",")
        if len(exprs) != len(tensors):
            raise ValueError(f"Expected {len(exprs)} tensors, got {len(tensors)}")
        values = {}
        for i, (expr, tensor) in enumerate(zip(exprs, tensors)):
            expr = expr.strip().replace(" ", "")
            if len(expr) != len(tensor.shape):
                raise ValueError(f"Expected {len(expr)} axes, got {len(tensor.shape)} for {i}-th (zero-based) input tensor")
            for axis, value in zip(expr, tensor.shape):
                if axis in values:
                    if values[axis] != value:
                        raise ValueError(f"Got conflicting values for axis {axis}: {values[axis]} and {value}")
                else:
                    values[axis] = value
        expr_out = eq.split("->")[-1].strip().replace(" ", "")
        shape_out = tuple(values[axis] for axis in expr_out)
        return Op("einsum", args=[eq, *tensors], shape=shape_out)

    def dot(a, b):
        raise NotImplementedError()

    def swapaxes(a, axis1, axis2):
        shape = list(a.shape)
        shape[axis1], shape[axis2] = shape[axis2], shape[axis1]
        return Op("swapaxes", args=[a, axis1, axis2], shape=shape)

    def stack(tensors, axis):
        shape = list(tensors[0].shape)
        shape.insert(axis, len(tensors))
        return Op("stack", args=[tensors, axis], shape=shape)

    def concatenate(tensors, axis):
        shape = list(tensors[0].shape)
        shape[axis] = sum(tensor.shape[axis] for tensor in tensors)
        return Op("concatenate", args=[tensors, axis], shape=shape)

    def zeros(shape, dtype="float32"):
        return Op("zeros", args=[shape, dtype], shape=shape)
    def ones(shape, dtype="float32"):
        return Op("ones", args=[shape, dtype], shape=shape)


    elementwise = elementwise
    add = partial(elementwise, op="add")
    subtract = partial(elementwise, op="subtract")
    multiply = partial(elementwise, op="multiply")
    true_divide = partial(elementwise, op="true_divide")
    floor_divide = partial(elementwise, op="floor_divide")
    divide = partial(elementwise, op="divide")
    logical_and = partial(elementwise, op="logical_and")
    logical_or = partial(elementwise, op="logical_or")
    where = partial(elementwise, op="where")
    less = partial(elementwise, op="less")
    less_equal = partial(elementwise, op="less_equal")
    greater = partial(elementwise, op="greater")
    greater_equal = partial(elementwise, op="greater_equal")
    equal = partial(elementwise, op="equal")
    not_equal = partial(elementwise, op="not_equal")
    maximum = partial(elementwise, op="maximum")
    minimum = partial(elementwise, op="minimum")

    reduce = reduce
    sum = partial(reduce, op="sum")
    mean = partial(reduce, op="mean")
    var = partial(reduce, op="var")
    std = partial(reduce, op="std")
    prod = partial(reduce, op="prod")
    count_nonzero = partial(reduce, op="count_nonzero")
    any = partial(reduce, op="any")
    all = partial(reduce, op="all")
    min = partial(reduce, op="min")
    max = partial(reduce, op="max")

    def sqrt(tensor):
        return Op("sqrt", args=[tensor], shape=tensor.shape)
    def rsqrt(tensor):
        return Op("rsqrt", args=[tensor], shape=tensor.shape)
    def square(tensor):
        return Op("square", args=[tensor], shape=tensor.shape)

    def allclose(a, b):
        return Op("allclose", args=[a, b], shape=())

    def vmap(op, in_axes, out_axes):
        return vmapped_op(op, in_axes, out_axes)

    def assert_shape(tensor, shape):
        def inner(x):
            if not x.shape is None:
                assert x.shape == shape, f"Expected shape {shape}, got {x.shape}"
            return x
        inner.__name__ = "id"
        return Op(inner, args=[tensor], shape=shape)

    class random:
        def bernoulli(rng, p, shape):
            return Op("random.bernoulli", args=[rng, p, shape], shape=shape)