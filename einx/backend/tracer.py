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
            shape = np.asarray(shape)

        return Op(operator.getitem, args=[self, key], output_shapes=shape).output_tracers

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
    def __init__(self, shape, index):
        super().__init__(shape)
        self.index = index

    def __str__(self):
        return f"tracer.Input({self.shape})"

    def __hash__(self):
        return 712873 + hash(self.shape)

    def __eq__(self, other):
        return isinstance(other, Input) and self.shape == other.shape

class Constant(Tracer):
    def __init__(self, value):
        value = np.asarray(value)
        super().__init__(value.shape)
        self.value = value

class OpOutput(Tracer):
    def __init__(self, op, shape, key):
        super().__init__(shape)
        self.op = op
        self.key = key

class Op:
    def __init__(self, op, args=[], kwargs={}, output_shapes=None, pass_backend=False):
        if op is None:
            raise TypeError("op cannot be None")
        self.op = op
        self.op_split = op.split(".") if isinstance(op, str) else None
        self.args = args
        self.kwargs = kwargs
        self.pass_backend = pass_backend
        self.output_shapes = output_shapes
        self.output_tracers = einx.tree_util.tree_map_with_key(lambda shape, key: OpOutput(self, shape, key), self.output_shapes)
        assert not "backend" in self.kwargs

class CustomOp:
    def __init__(self, op):
        assert isinstance(op, Op)
        self.op = op

    def __call__(self, *args, backend, **kwargs):
        if backend == einx.backend.tracer:
            return Op(self.op.op, args=args, kwargs=kwargs, output_shapes=self.op.output_shapes, pass_backend=self.op.pass_backend).output_tracers
        else:
            if self.op.pass_backend:
                assert not "backend" in kwargs
                kwargs["backend"] = backend
            return self.op.op(*args, **kwargs)

def to_eval_str(x, names, lines, constants):
    if isinstance(x, Input):
        if id(x) in names:
            return names[id(x)]
        name = names[id(x)] = f"i{x.index}"
        return name
    elif isinstance(x, Constant):
        return to_eval_str(x.value, names, lines, constants)
    elif isinstance(x, Op):
        if id(x) in names:
            return names[id(x)]

        if x.op == operator.getitem:
            assert len(x.args) == 2 and len(x.kwargs) == 0
            tensor = to_eval_str(x.args[0], names, lines, constants)

            slices = x.args[1]
            if not isinstance(slices, tuple):
                slices = (slices,)
            assert isinstance(slices, tuple)
            assert len(slices) > 0

            def slice_to_str(s):
                if isinstance(s, slice):
                    x = ""
                    if not s.start is None:
                        x += str(s.start)
                    x += ":"
                    if not s.stop is None:
                        x += str(s.stop)
                    if not s.step is None:
                        x += ":" + str(s.step)
                    return x
                else:
                    return to_eval_str(s, names, lines, constants)
            slices = ", ".join(slice_to_str(s) for s in slices)

            name = names[id(x)] = f"x{len(names)}"

            line = (name, f"{tensor}[" + slices + "]")
            lines.append(line)

            return name
        else:
            pass_backend = x.pass_backend
            if isinstance(x.op, str):
                op = f"backend.{x.op}"
            else:
                if not callable(x.op):
                    raise TypeError(f"Expected callable, got {type(x.op)}")

                if x.op == einx.param.instantiate:
                    op = "einx.param.instantiate"
                else:
                    num_existing_ops = len([name for name, value, desc in constants if name.startswith("op")])
                    op = f"op{num_existing_ops}"
                    custom_op = CustomOp(x)
                    constants.append((op, custom_op, repr(custom_op)))
                    pass_backend = True

            args = \
                [to_eval_str(a, names, lines, constants) for a in x.args] \
            + [f"{k}={to_eval_str(v, names, lines, constants)}" for k, v in x.kwargs.items()] \
            + (["backend=backend"] if pass_backend else [])
            args = ", ".join(args)

            name = names[id(x)] = f"x{len(names)}"

            line = (name, f"{op}(" + args + ")")
            lines.append(line)

            return name
    elif isinstance(x, OpOutput):
        name = to_eval_str(x.op, names, lines, constants)
        for k in x.key:
            name += f"[{to_eval_str(k, names, lines, constants)}]"
        return name
    elif isinstance(x, str):
        return f"\"{x}\""
    elif isinstance(x, tuple):
        return "(" + ", ".join(to_eval_str(a, names, lines, constants) for a in x) + ("," if len(x) == 1 else "") + ")"
    elif isinstance(x, list):
        return "[" + ", ".join(to_eval_str(a, names, lines, constants) for a in x) + "]"
    elif isinstance(x, dict):
        return "{" + ", ".join(f"{k}: {to_eval_str(v, names, constants)}" for k, v in x.items()) + "}"
    elif isinstance(x, (int, float)):
        return str(x)
    else:
        if id(x) in names:
            return names[id(x)]
        num_existing_consts = len([name for name, value, desc in constants if name.startswith("const")])
        name = names[id(x)] = f"const{num_existing_consts}"
        constants.append((name, x, repr(x)))
        return name

class Graph:
    def __init__(self, output, name, args, kwargs):
        if name.endswith("_stage0"):
            name = name[:-len("_stage0")]
        self.output = output
        self.name = name
        self.args = args
        self.kwargs = kwargs

        # Get input tracers
        self.input_tracers = []
        index = 0
        def map(x):
            nonlocal index
            if isinstance(x, Input):
                self.input_tracers.append(x)
                assert x.index == index
                index += 1
            return x
        einx.tree_util.tree_map(map, args)
        einx.tree_util.tree_map(map, kwargs)

        # Generate Python code for the graph
        names = {}
        lines = []
        constants = []
        to_eval_str(output, names, lines, constants)

        string = ""
        for const_name, value, desc in constants:
            string += f"# {const_name}: {desc}\n"
        string += f"def {name}({', '.join([to_eval_str(x, names, lines, constants) for x in self.input_tracers] + ['backend'])}):\n"
        for var_name, line in lines:
            string += f"    {var_name} = {line}\n"
        string += f"    return {to_eval_str(output, names, lines, constants)}"
        self.op_string = string

        # Just-in-time compile the graph
        scope_globals = {const_name: value for const_name, value, desc in constants}
        scope_globals["einx"] = einx
        scope_locals = {}
        exec(string, scope_globals, scope_locals)
        self.op = scope_locals[name]

    def __call__(self, *tracer_values, backend=None):
        if len(tracer_values) != len(self.input_tracers):
            raise ValueError(f"Expected {len(self.input_tracers)} inputs, got {len(tracer_values)}")
        if backend is None:
            backend = einx.backend.get(tracer_values)
        elif isinstance(backend, str):
            backend = einx.backend.get(backend)

        return self.op(*tracer_values, backend=backend)

    def __str__(self):
        return self.op_string



class vmapped_op:
    def __init__(self, op, in_axes, out_axes):
        self.op = op
        self.in_axes = in_axes
        self.out_axes = out_axes

    def _to_backend(self, backend):
        if isinstance(self.op, vmapped_op):
            op = self.op._to_backend(backend)
        else:
            op = self.op
        return backend.vmap(op, in_axes=self.in_axes, out_axes=self.out_axes)

    def __call__(self, *args, backend, **kwargs):
        return self._to_backend(backend)(*args, **kwargs)

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
    return Op(op, args=args, output_shapes=np.asarray(shape)).output_tracers

def reduce(tensor, axis, *, op=None, **kwargs):
    keepdims = kwargs.get("keepdims", False)
    axes = [axis] if isinstance(axis, int) else axis
    shape = list(tensor.shape)
    if keepdims:
        for a in axes:
            shape[a] = 1
    else:
        for a in reversed(sorted(axes)):
            del shape[a]
    return Op(op, args=[tensor], kwargs=kwargs | {"axis": axis}, output_shapes=np.asarray(shape)).output_tracers

def map(tensor, axis, op, *args, **kwargs):
    return Op(op, args=[tensor], kwargs=kwargs | {"axis": axis}, output_shapes=np.asarray(tensor.shape)).output_tracers

def index(tensor, coordinates, update=None, op=None):
    return Op(op, args=[tensor, coordinates, update], output_shapes=np.asarray(coordinates[0].shape)).output_tracers

class tracer:
    Input = Input
    Constant = Constant
    Op = Op
    Graph = Graph

    @staticmethod
    def to_tensor(tensor):
        if isinstance(tensor, Tracer):
            return Op("to_tensor", args=[tensor], output_shapes=np.asarray(tensor.shape)).output_tracers
        else:
            return Constant(tensor)

    tensor = Tracer
    name = "tracer"

    def apply(op, args, kwargs, output_shapes):
        return Op(op, args=args, kwargs=kwargs, output_shapes=output_shapes, pass_backend=isinstance(op, vmapped_op)).output_tracers

    def cast(tensor, dtype):
        return Op("cast", args=[tensor], kwargs={"dtype": dtype}, output_shapes=np.asarray(tensor.shape)).output_tracers
    def reshape(tensor, shape):
        if isinstance(tensor, OpOutput) and tensor.op.op == "reshape":
            # Merge consecutive reshapes
            return Op("reshape", args=[tensor.op.args[0], shape], output_shapes=np.asarray(shape)).output_tracers
        else:
            return Op("reshape", args=[tensor, shape], output_shapes=np.asarray(shape)).output_tracers

    def transpose(tensor, perm):
        shape = [tensor.shape[i] for i in perm]
        return Op("transpose", args=[tensor, perm], output_shapes=np.asarray(shape)).output_tracers

    def broadcast_to(tensor, shape):
        return Op("broadcast_to", args=[tensor, shape], output_shapes=np.asarray(shape)).output_tracers

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
        return Op("einsum", args=[eq, *tensors], output_shapes=np.asarray(shape_out)).output_tracers

    def dot(a, b):
        raise NotImplementedError()

    def swapaxes(a, axis1, axis2):
        shape = list(a.shape)
        shape[axis1], shape[axis2] = shape[axis2], shape[axis1]
        return Op("swapaxes", args=[a, axis1, axis2], output_shapes=np.asarray(shape)).output_tracers

    def arange(n, dtype="int32"):
        return Op("arange", args=[n], kwargs={"dtype": dtype}, output_shapes=np.asarray([n])).output_tracers

    def stack(tensors, axis):
        shape = list(tensors[0].shape)
        shape.insert(axis, len(tensors))
        return Op("stack", args=[tensors, axis], output_shapes=np.asarray(shape)).output_tracers

    def concatenate(tensors, axis):
        shape = list(tensors[0].shape)
        shape[axis] = sum(tensor.shape[axis] for tensor in tensors)
        return Op("concatenate", args=[tensors, axis], output_shapes=np.asarray(shape)).output_tracers

    def zeros(shape, dtype="float32"):
        return Op("zeros", args=[shape, dtype], output_shapes=np.asarray(shape)).output_tracers
    def ones(shape, dtype="float32"):
        return Op("ones", args=[shape, dtype], output_shapes=np.asarray(shape)).output_tracers


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
    logsumexp = partial(reduce, op="logsumexp")

    get_at = partial(index, op="get_at")
    set_at = partial(index, op="set_at")

    flip = partial(map, op="flip")
    roll = partial(map, op="roll")
    softmax = partial(map, op="softmax")
    log_softmax = partial(map, op="log_softmax")

    def sqrt(tensor):
        return Op("sqrt", args=[tensor], output_shapes=np.asarray(tensor.shape)).output_tracers
    def rsqrt(tensor):
        return Op("rsqrt", args=[tensor], output_shapes=np.asarray(tensor.shape)).output_tracers
    def square(tensor):
        return Op("square", args=[tensor], output_shapes=np.asarray(tensor.shape)).output_tracers

    def allclose(a, b):
        return Op("allclose", args=[a, b], shape=np.asarray([]).astype("int32")).output_tracers

    def vmap(op, in_axes, out_axes):
        return vmapped_op(op, in_axes, out_axes)

    class random:
        def bernoulli(rng, p, shape):
            return Op("random.bernoulli", args=[rng, p, shape], output_shapes=np.asarray(shape)).output_tracers