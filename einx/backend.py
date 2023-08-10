import numpy as np
from functools import partial
import sys, einx

backends = []
backend_factories = {}

def to_tuple(x):
    if isinstance(x, tuple):
        return x
    elif isinstance(x, list):
        return tuple(x)
    elif isinstance(x, np.ndarray):
        return tuple(x.tolist())
    else:
        raise ValueError(f"Cannot convert {type(x)} to tuple")

class numpy:
    @staticmethod
    def to_tensor(tensor):
        return np.asarray(tensor)

    tensor = np.ndarray
    name = "numpy"

    cast = lambda tensor, dtype: tensor.astype(dtype)
    reshape = np.reshape
    transpose = np.transpose
    broadcast_to = np.broadcast_to
    einsum = partial(np.einsum, optimize="optimal")
    dot = np.dot
    swapaxes = np.swapaxes

    zeros = np.zeros
    ones = np.ones

    add = np.add
    subtract = np.subtract
    multiply = np.multiply
    true_divide = np.true_divide
    floor_divide = np.floor_divide
    divide = np.divide
    logical_and = np.logical_and
    logical_or = np.logical_or
    where = np.where
    less = np.less
    less_equal = np.less_equal
    greater = np.greater
    greater_equal = np.greater_equal
    equal = np.equal
    not_equal = np.not_equal
    maximum = np.maximum
    minimum = np.minimum

    sum = np.sum
    mean = np.mean
    var = np.var
    std = np.std
    prod = np.prod
    count_nonzero = np.count_nonzero
    any = np.any
    all = np.all
    min = np.amin
    max = np.amax

    sqrt = np.sqrt
    rsqrt = lambda x: 1.0 / np.sqrt(x)

backends.append(numpy)

def make_jax_backend():
    import jax as jax_
    import jax.numpy as jnp
    class jax:
        @staticmethod
        def to_tensor(tensor):
            if not isinstance(tensor, jnp.ndarray) and isinstance(tensor, np.ndarray):
                return tensor
            else:
                return jnp.asarray(tensor)

        tensor = jnp.ndarray
        name = "jax"

        cast = lambda tensor, dtype: tensor.astype(dtype)
        reshape = jnp.reshape
        transpose = jnp.transpose
        broadcast_to = jnp.broadcast_to
        einsum = partial(jnp.einsum, optimize="optimal")
        dot = jnp.dot
        swapaxes = jnp.swapaxes

        zeros = jnp.zeros
        ones = jnp.ones

        add = jnp.add
        subtract = jnp.subtract
        multiply = jnp.multiply
        true_divide = jnp.true_divide
        floor_divide = jnp.floor_divide
        divide = jnp.divide
        logical_and = jnp.logical_and
        logical_or = jnp.logical_or
        where = jnp.where
        less = jnp.less
        less_equal = jnp.less_equal
        greater = jnp.greater
        greater_equal = jnp.greater_equal
        equal = jnp.equal
        not_equal = jnp.not_equal
        maximum = jnp.maximum
        minimum = jnp.minimum

        sum = jnp.sum
        mean = jnp.mean
        var = jnp.var
        std = jnp.std
        prod = jnp.prod
        count_nonzero = jnp.count_nonzero
        any = jnp.any
        all = jnp.all
        min = jnp.amin
        max = jnp.amax

        sqrt = jnp.sqrt
        rsqrt = jax_.lax.rsqrt
        square = jnp.square

    return jax
backend_factories["jax"] = make_jax_backend

def make_torch_backend():
    import torch as torch_
    import torch._dynamo as _dynamo
    class torch:
        @staticmethod
        def to_tensor(tensor):
            if isinstance(tensor, np.ndarray) or torch_.is_tensor(tensor):
                return tensor
            else:
                return torch_.asarray(tensor)

        tensor = torch_.Tensor
        name = "torch"

        cast = lambda tensor, dtype: tensor.type(vars(torch_)[dtype] if isinstance(dtype, str) else dtype)
        reshape = lambda tensor, shape: torch_.reshape(tensor, to_tuple(shape))
        transpose = torch_.permute
        broadcast_to = lambda tensor, shape: torch_.broadcast_to(tensor, to_tuple(shape))
        einsum = torch_.einsum
        dot = torch_.matmul
        swapaxes = torch_.swapaxes

        zeros = lambda shape, dtype="float32": torch_.zeros(*shape, dtype=vars(torch_)[dtype] if isinstance(dtype, str) else dtype)
        ones = lambda shape, dtype="float32": torch_.ones(*shape, dtype=vars(torch_)[dtype] if isinstance(dtype, str) else dtype)

        add = torch_.add
        subtract = torch_.subtract
        multiply = torch_.multiply
        true_divide = torch_.true_divide
        floor_divide = torch_.floor_divide
        divide = torch_.divide
        logical_and = torch_.logical_and
        logical_or = torch_.logical_or
        where = torch_.where
        less = torch_.less
        less_equal = torch_.less_equal
        greater = torch_.greater
        greater_equal = torch_.greater_equal
        equal = torch_.equal
        not_equal = torch_.not_equal
        maximum = lambda a, b: torch_.maximum(torch.to_tensor(a), torch.to_tensor(b))
        minimum = lambda a, b: torch_.minimum(torch.to_tensor(a), torch.to_tensor(b)) # TODO: add support for python scalars everywhere

        sum = torch_.sum
        mean = torch_.mean
        var = torch_.var
        std = torch_.std
        prod = torch_.prod
        count_nonzero = torch_.count_nonzero
        any = torch_.any
        all = torch_.all
        min = torch_.min
        max = torch_.max

        sqrt = torch_.sqrt
        rsqrt = torch_.rsqrt
        square = torch_.square

    _dynamo.allow_in_graph(einx.dot)
    _dynamo.allow_in_graph(einx.rearrange)
    _dynamo.allow_in_graph(einx.elementwise)
    _dynamo.allow_in_graph(einx.reduce)
    _dynamo.allow_in_graph(einx.dl.meanvar_norm)
    _dynamo.allow_in_graph(einx.dl.linear)

    for op_name in einx.elementwise.op_names + einx.reduce.op_names:
        _dynamo.allow_in_graph(getattr(einx, op_name))

    return torch
backend_factories["torch"] = make_torch_backend

def make_tensorflow_backend():
    import tensorflow as tf
    import tensorflow.experimental.numpy as tnp
    class tensorflow:
        @staticmethod
        def to_tensor(tensor):
            if isinstance(tensor, np.ndarray):
                return tensor
            else:
                tensor = tf.convert_to_tensor(tensor)
                if any(s is None for s in tensor.shape):
                    raise ValueError("Tensorflow tensors with dynamic shape are not supported")
                return tensor

        tensor = tf.Tensor
        name = "tensorflow"

        cast = tf.cast
        reshape = tf.reshape
        transpose = tf.transpose
        broadcast_to = tf.broadcast_to
        einsum = partial(tf.einsum, optimize="optimal")
        dot = tnp.dot
        swapaxes = tnp.swapaxes

        zeros = lambda shape, dtype="float32": tf.zeros(shape, dtype=dtype)
        ones = lambda shape, dtype="float32": tf.ones(shape, dtype=dtype)

        add = tnp.add
        subtract = tnp.subtract
        multiply = tnp.multiply
        true_divide = tnp.true_divide
        floor_divide = tnp.floor_divide
        divide = tnp.divide
        logical_and = tnp.logical_and
        logical_or = tnp.logical_or
        where = tnp.where
        less = tnp.less
        less_equal = tnp.less_equal
        greater = tnp.greater
        greater_equal = tnp.greater_equal
        equal = tnp.equal
        not_equal = tnp.not_equal
        maximum = tnp.maximum
        minimum = tnp.minimum

        sum = tnp.sum
        mean = tnp.mean
        var = tnp.var
        var = tnp.std
        prod = tnp.prod
        count_nonzero = tnp.count_nonzero
        any = tnp.any
        all = tnp.all
        min = tnp.min
        max = tnp.max

        sqrt = tf.math.sqrt
        rsqrt = tf.math.rsqrt
        square = tnp.square

    return tensorflow
backend_factories["tensorflow"] = make_tensorflow_backend

def update():
    for backend_name in list(backend_factories.keys()):
        if backend_name in sys.modules:
            backends.append(backend_factories[backend_name]())
            del backend_factories[backend_name]



type_to_backend = {}

def _get1(tensor):
    tensor_backend = type_to_backend.get(type(tensor), None)
    if tensor_backend is None:
        update()
        
        # Find matching backend
        for tensor_backend in backends:
            if isinstance(tensor, tensor_backend.tensor) and not isinstance(tensor, numpy.tensor):
                break
        else:
            # Default backend is numpy
            tensor_backend = numpy
        type_to_backend[type(tensor)] = tensor_backend
    return tensor_backend

def get(tensors):
    if len(tensors) == 1:
        return _get1(tensors[0])
    backend = None
    for tensor in tensors:
        backend2 = _get1(tensor)
        if backend2 != numpy:
            if not backend is None and backend != backend2:
                raise ValueError(f"Got tensors with conflicting backends: {backend.__name__} and {backend2.__name__}")
            backend = backend2
    if backend is None:
        return numpy
    else:
        return backend

def get_by_name(name):
    update()
    for backend in backends:
        if backend.name == name:
            return backend
    raise ValueError(f"Backend {name} not found")