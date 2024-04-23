import numpy as np
from .tracer import *
from functools import partial


class op:
    def reshape(op: Tracer):
        @trace
        def reshape(tensor, shape):
            if shape == get_shape(tensor):
                return tensor
            else:
                return apply(op, args=[tensor, shape], output=Tensor(shape), signature="reshape")

        return reshape

    def transpose(op: Tracer):
        @trace
        def transpose(tensor, perm):
            if list(perm) == list(range(tensor.ndim)):
                return tensor
            else:
                shape = tuple(tensor.shape[i] for i in perm)
                return apply(op, args=[tensor, perm], output=Tensor(shape), signature="transpose")

        return transpose

    def broadcast_to(op: Tracer):
        @trace
        def broadcast_to(tensor, shape):
            if get_shape(tensor) == shape:
                return tensor
            else:
                return apply(
                    op, args=[tensor, shape], output=Tensor(shape), signature="broadcast_to"
                )

        return broadcast_to

    def einsum(op: Tracer):
        @trace
        def einsum(eq, *tensors, **kwargs):
            exprs = eq.split("->")[0].split(",")
            if len(exprs) != len(tensors):
                raise ValueError(f"Expected {len(exprs)} tensors, got {len(tensors)}")
            values = {}
            for i, (expr, tensor) in enumerate(zip(exprs, tensors)):
                expr = expr.strip().replace(" ", "")
                if len(expr) != len(tensor.shape):
                    raise ValueError(
                        f"Expected {len(expr)} axes, got {len(tensor.shape)} for {i}-th "
                        "(zero-based) input tensor"
                    )
                for axis, value in zip(expr, tensor.shape):
                    if axis in values:
                        if values[axis] != value:
                            raise ValueError(
                                f"Got conflicting values for axis {axis}: {values[axis]} and {value}"
                            )
                    else:
                        values[axis] = value
            expr_out = eq.split("->")[-1].strip().replace(" ", "")
            shape_out = tuple(values[axis] for axis in expr_out)
            return apply(op, args=[eq, *tensors], kwargs=kwargs, output=Tensor(shape_out))

        return einsum

    def arange(op: Tracer):
        @trace
        def arange(n, dtype="int32"):
            return apply(op, args=[n], kwargs={"dtype": dtype}, output=Tensor((n,)))

        return arange

    def stack(op: Tracer):
        @trace
        def stack(tensors, axis=0):
            if axis < 0:
                axis = len(tensors[0].shape) + axis + 1
            shape = list(tensors[0].shape)
            shape.insert(axis, len(tensors))
            return apply(op, args=[tensors], kwargs={"axis": axis}, output=Tensor(shape))

        return stack

    def concatenate(op: Tracer):
        @trace
        def concatenate(tensors, axis=0):
            shape = list(tensors[0].shape)
            shape[axis] = sum(tensor.shape[axis] for tensor in tensors)
            return apply(op, args=[tensors], kwargs={"axis": axis}, output=Tensor(shape))

        return concatenate

    def fill_constant(op: Tracer, value):
        @trace
        def fill_constant(shape, dtype="float32"):
            return apply(op, args=[shape], kwargs={"dtype": dtype}, output=Tensor(shape))

        return fill_constant

    def elementwise(op: Tracer):
        @trace
        def elementwise(*args, **kwargs):
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
            assert not shape is None  # TODO: can this happen?

            return apply(op, args=args, kwargs=kwargs, output=Tensor(shape))

        return elementwise

    def keep_shape(op: Tracer):
        @trace
        def keep_shape(*args, **kwargs):
            return apply(op, args=args, kwargs=kwargs, output=Tensor(args[0].shape))

        return keep_shape

    def reduce(op: Tracer):
        @trace
        def reduce(tensor, axis=None, **kwargs):
            keepdims = kwargs.get("keepdims", False)
            if axis is None:
                shape = ()
            else:
                axes = [axis] if isinstance(axis, int) else axis
                shape = list(tensor.shape)
                if keepdims:
                    for a in axes:
                        shape[a] = 1
                else:
                    for a in sorted(axes, reverse=True):
                        del shape[a]
                kwargs = {**kwargs, **{"axis": axis}}
            return apply(op, args=[tensor], kwargs=kwargs, output=Tensor(shape))

        return reduce

    def get_at(op: Tracer):
        @trace
        def get_at(tensor, coordinates):
            coordinates2 = (coordinates,) if not isinstance(coordinates, tuple) else coordinates
            if len([c for c in coordinates2 if c is not None]) > len(tensor.shape):
                raise ValueError(f"Too many indices for tensor of dimension {len(tensor.shape)}")

            def is_multidim(c):
                if c is None or isinstance(c, (slice, int, np.integer)):
                    return False
                elif isinstance(c, list):
                    return True
                else:
                    return c.ndim > 0

            if any(is_multidim(c) for c in coordinates2):
                # Got multi-dimensional indices
                while len(coordinates2) < len(tensor.shape):
                    coordinates2 = coordinates2 + (slice(None),)

                # Find front and back slices
                front_slices = []
                back_slices = []
                i = 0
                is_front = True
                for i in range(tensor.ndim):
                    if is_front:
                        if isinstance(coordinates2[i], slice):
                            front_slices.append(i)
                        else:
                            is_front = False
                    else:
                        if isinstance(coordinates2[i], slice):
                            back_slices.append(i)

                # Broadcast coordinates expressions
                def broadcast(dims):
                    dims = np.asarray(list({int(i) for i in dims}))
                    assert np.all(dims > 0)
                    if len(dims) > 2 or len(dims) == 2 and np.amin(dims) > 1:
                        raise ValueError("Cannot broadcast coordinates")
                    return np.amax(dims)

                shapes = [c.shape for c in coordinates2 if not isinstance(c, slice)]
                if len({len(s) for s in shapes}) != 1:
                    raise ValueError("Expected all coordinates to have same number of dimensions")
                shapes = np.asarray(shapes)
                shape = [broadcast(shapes[:, i]) for i in range(shapes.shape[1])]

                # Prepend and append slices
                shape = tuple(
                    [tensor.shape[i] for i in front_slices]
                    + shape
                    + [tensor.shape[i] for i in back_slices]
                )
            else:
                output_shape = []
                input_shape = tensor.shape
                for s in coordinates2:
                    if isinstance(s, (int, np.integer)):
                        input_shape = input_shape[1:]
                    elif isinstance(s, slice):
                        start, stop, step = s.indices(input_shape[0])
                        output_shape.append((stop - start) // step)
                        input_shape = input_shape[1:]
                    elif s is None:
                        output_shape.append(1)
                    elif isinstance(s, Tensor) and s.ndim == 0:
                        input_shape = input_shape[1:]
                    else:
                        raise TypeError(f"Invalid coordinate type: {type(s)}")
                shape = tuple(output_shape) + tuple(input_shape)

            return apply(op, args=[tensor, coordinates], output=Tensor(shape))

        return get_at

    def update_at(op: Tracer = None, inplace=False):
        if op is None:
            return partial(einx.tracer.tensor.op.update_at, inplace=inplace)

        @trace
        def update_at(tensor, coordinates, update):
            output = Tensor(tensor.shape)
            return apply(
                op,
                args=[tensor, coordinates, update],
                output=output,
                inplace_updates=[(tensor, output)] if inplace else [],
            )

        return update_at

    def vmap(vmap):
        @trace
        def vmap_with_output_types(op, in_axes, out_axes, input_shapes, output_shapes):
            return apply(
                vmap,
                args=[op],
                kwargs={"in_axes": in_axes, "out_axes": out_axes},
                signature="vmap",
                output=Function(output=[Tensor(shape) for shape in output_shapes]),
            )

        return vmap_with_output_types


class Tensor(Tracer):
    def __init__(self, shape):
        Tracer.__init__(self)

        if isinstance(shape, np.ndarray):
            if shape.ndim != 1:
                raise ValueError(f"Invalid shape: {shape}")
            self.shape = tuple(int(i) for i in shape)
        else:
            try:
                self.shape = tuple(int(i) for i in shape)
            except:
                raise ValueError(f"Invalid shape: {shape}")

    @property
    def ndim(self):
        return len(self.shape)

    def __copy__(self):
        assert type(self) == Tensor
        return Tensor(self.shape)

    def __getitem__(self, key):
        return op.get_at(GetAt())(self, key)

    def __setitem__(self, key, value):
        if (
            not value.origin is None
            and isinstance(value.origin.op, AssignAt)
            and value.origin.op != "="
            and value.origin.args[0] is self
            and value.origin.args[1] is key
        ):
            # Python reformulates operations like 'tensor[key] += update' as follows:
            # 1. x1 = __getitem__(tensor, key)
            # 2. x2 = __iadd__(x1, update)
            # 3. x3 = __setitem__(tensor, key, x2)
            # The output of the second line already returns the results of the AssignAt (see below), so
            # we can skip the third line.
            return value
        return op.update_at(AssignAt("="), inplace=True)(self, key, value)

    def __iadd__(self, value):
        if not isinstance(self.origin.op, GetAt):
            raise ValueError("Inplace operator only supported for get_at outputs")
        return op.update_at(AssignAt("+="), inplace=True)(
            self.origin.args[0], self.origin.args[1], value
        )

    def __isub__(self, value):
        if not isinstance(self.origin.op, GetAt):
            raise ValueError("Inplace operator only supported for get_at outputs")
        return op.update_at(AssignAt("-="), inplace=True)(
            self.origin.args[0], self.origin.args[1], value
        )

    def __imul__(self, value):
        if not isinstance(self.origin.op, GetAt):
            raise ValueError("Inplace operator only supported for get_at outputs")
        return op.update_at(AssignAt("*="), inplace=True)(
            self.origin.args[0], self.origin.args[1], value
        )

    def __itruediv__(self, value):
        if not isinstance(self.origin.op, GetAt):
            raise ValueError("Inplace operator only supported for get_at outputs")
        return op.update_at(AssignAt("/="), inplace=True)(
            self.origin.args[0], self.origin.args[1], value
        )

    def __ifloordiv__(self, value):
        if not isinstance(self.origin.op, GetAt):
            raise ValueError("Inplace operator only supported for get_at outputs")
        return op.update_at(AssignAt("//="), inplace=True)(
            self.origin.args[0], self.origin.args[1], value
        )

    def __add__(self, other):
        return op.elementwise(Operator("+"))(self, other)

    def __radd__(self, other):
        return op.elementwise(Operator("+"))(other, self)

    def __neg__(self):
        return op.elementwise(Operator("-"))(self)

    def __sub__(self, other):
        return op.elementwise(Operator("-"))(self, other)

    def __rsub__(self, other):
        return op.elementwise(Operator("-"))(other, self)

    def __mul__(self, other):
        return op.elementwise(Operator("*"))(self, other)

    def __rmul__(self, other):
        return op.elementwise(Operator("*"))(other, self)

    def __truediv__(self, other):
        return op.elementwise(Operator("/"))(self, other)

    def __rtruediv__(self, other):
        return op.elementwise(Operator("/"))(other, self)

    def __floordiv__(self, other):
        return op.elementwise(Operator("//"))(self, other)

    def __rfloordiv__(self, other):
        return op.elementwise(Operator("//"))(other, self)

    def __div__(self, other):
        return op.elementwise(Operator("/"))(self, other)

    def __rdiv__(self, other):
        return op.elementwise(Operator("/"))(other, self)

    def __mod__(self, other):
        return op.elementwise(Operator("%"))(self, other)

    def __rmod__(self, other):
        return op.elementwise(Operator("%"))(other, self)

    def __lt__(self, other):
        return op.elementwise(Operator("<"))(self, other)

    def __le__(self, other):
        return op.elementwise(Operator("<="))(self, other)

    def __gt__(self, other):
        return op.elementwise(Operator(">"))(self, other)

    def __ge__(self, other):
        return op.elementwise(Operator(">="))(self, other)

    def __eq__(self, other):
        return op.elementwise(Operator("=="))(self, other)

    def __ne__(self, other):
        return op.elementwise(Operator("!="))(self, other)


class Scalar(Tensor):
    def __init__(self):
        Tensor.__init__(self, ())


class TensorRequiringConversion(Tensor):
    def __init__(self, shape):
        Tensor.__init__(self, shape)


class TensorFactory(Tracer):
    def __init__(self, params):
        self.params = params

    def __call__(self, shape, kwargs):
        # Filter kwargs
        if any(param.startswith("**") for param in self.params):
            pass
        else:
            kwargs = {k: v for k, v in kwargs.items() if k in self.params}

        return apply(self, args=[shape], kwargs=kwargs, output=Tensor(shape))


def is_scalar(x):
    return isinstance(x, (int, float, bool, np.integer, np.floating, np.bool_, Scalar))


def is_tensor(x):
    return isinstance(x, (int, float, bool, np.integer, np.floating, np.bool_, Tensor))


def _get_list_shape(x):
    if isinstance(x, (tuple, list)):
        subshapes = {_get_list_shape(y) for y in x}
        if len(subshapes) != 1:
            raise ValueError("Failed to determine shape of input tensor")
        subshape = subshapes.pop()
        return (len(x),) + subshape
    elif is_scalar(x):
        return ()
    else:
        raise ValueError("Failed to determine shape of input tensor")


def get_shape(x):
    if isinstance(x, (tuple, list)):
        return _get_list_shape(x)
    elif is_scalar(x):
        return ()

    try:
        # Concrete tensor
        return tuple(int(i) for i in x.shape)
    except:
        # Cannot determine shape (e.g. tensor factory)
        return None


@trace
def call_factory(x, shape, backend, **kwargs):
    if is_tensor(x):
        return x
    elif isinstance(x, TensorFactory):
        return x(shape, kwargs=kwargs)
    else:
        assert False, f"{type(x)}"
