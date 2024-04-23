import numpy as np
from . import tracer, tensor
import einx
import inspect

# Define classes for different types of inputs that act as cache keys and will
# be converted into the corresponding tracer objects when a graph is constructed


class CacheKey:
    pass


class Scalar(CacheKey):
    def __eq__(self, other):
        return isinstance(other, Scalar)

    def __hash__(self):
        return 1

    def to_tracer(self, backend, virtual_arg):
        x = tensor.Scalar()
        return x, x


class Tensor(CacheKey):
    def __init__(self, shape, type):
        self.shape = shape
        self.type = type

    def __eq__(self, other):
        return isinstance(other, Tensor) and other.shape == self.shape and other.type == self.type

    def __hash__(self):
        return 2 + hash(self.shape) + hash(self.type)

    def to_tracer(self, backend, virtual_arg):
        if any(issubclass(self.type, type) for type in backend.tensor_types):
            x = tensor.Tensor(self.shape)
        else:
            x = tensor.TensorRequiringConversion(self.shape)
        return x, x


class TensorFactory(CacheKey):
    def __init__(self, params):
        self.params = tuple(params)

    def __eq__(self, other):
        return isinstance(other, TensorFactory) and other.params == self.params

    def __hash__(self):
        return 3 + hash(self.params)

    def to_tracer(self, backend, virtual_arg):
        x = tensor.TensorFactory(self.params)
        return x, x


class Input:
    pass


tensor_factories = []


def register_tensor_factory(factory):
    tensor_factories.append(factory)
    return factory


def apply_registered_tensor_factory(x):
    for factory in tensor_factories:
        x2 = factory(x)
        if x2 is not None:
            return x2
    return None


def concrete_to_value_and_key(x):
    if isinstance(x, (float, int, np.floating, np.integer, bool, np.bool_)):
        # Scalar
        return x, Scalar()
    elif isinstance(x, (tuple, list)):
        # Nested list/ tuple of scalars
        shape = einx.tracer.get_shape(x)
        if shape is None:
            raise ValueError("Failed to determine shape of input tensor")
        return x, Tensor(shape, type(x))
    elif isinstance(x, Input):
        # Custom input
        return x.to_value_and_key()
    elif not (x2 := apply_registered_tensor_factory(x)) is None:
        # Registered tensor factory
        return x2
    elif callable(x):
        # Simple callable tensor factory
        params = []
        try:
            for name, param in inspect.signature(x).parameters.items():
                if param.kind == inspect.Parameter.VAR_KEYWORD:
                    name = f"**{name}"
                elif param.kind == inspect.Parameter.VAR_POSITIONAL:
                    name = f"*{name}"
                params.append(name)
        except:
            pass
        return x, TensorFactory(params)
    else:
        # Tensor
        return x, Tensor(tuple(int(i) for i in x.shape), type(x))


def key_to_tracer(x, backend, virtual_arg):
    args = []

    def map(x):
        if isinstance(x, CacheKey):
            arg, x = x.to_tracer(backend, virtual_arg)
            if not arg is None:
                args.append(arg)
            return x
        else:
            return x

    x = einx.tree_util.tree_map(map, x)

    return args, x
