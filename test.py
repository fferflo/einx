import timeit, sys, os
import numpy as np

import jax, torch
# import jax.numpy as jnp

# os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
# import tensorflow

import einx

backend = einx.backend.get("numpy")

x = np.arange(3)[np.newaxis]

# y = einx.vmap_with_axis("a [b] -> a [b]", x, op=np.flip)
# print(str(y))

# y = einx.vmap_with_axis("a b, a b -> a b", x, x, op=np.add)
# print(str(y))

y = einx.vmap("a [b] -> a [b]", x, op=np.flip, graph=True)
print(str(y))

y = einx.vmap_with_axis("a [b] -> a", x, op=np.sum, graph=True)
print(str(y))

# y = einx.sum("a [b]", x)
# print(str(y))

# y = einx.add("a b, a b -> a b", x, x)
# print(str(y))

# y = einx.roll("a [b] -> a [b]", x, shift=1)
# print(str(y))

# x = backend.zeros((10, 10), "float32")
# y = backend.zeros((10,), "float32")

# g = einx.add("b, -> b 3", y, 1, graph=True)
# print(str(g))

sys.exit(0)


exprs_in = ["(a + b + (c1 + c2) + (d e))"]
exprs_out = ["(e d) + c1 + c2", "(b + a) z"]
tensor_shapes = [(10,)]
parameters = {
    "a": 2,
    "b": 3,
    "c1": 2,
    "c2": 1,
    "d": 1,
    "z": 2,
}

# exprs_in = ["a [b]", "[b]", "..."]
# exprs_out = ["([b] + [1]) c a ..."]
# tensor_shapes = [(3, 2,), (2,), (3,)]
# parameters = {
#     "c": 2,
# }

op_string = ", ".join(exprs_in) + " -> " + ", ".join(exprs_out)
print(f"OP " + op_string)

tensors_in = [np.ones(shape) for shape in tensor_shapes]
output_ndims = None
output_shape = None

exprs = einx.expr.solve(
      [einx.expr.Condition(expr=expr_in, value=shape, depth=0) for expr_in, shape in zip(exprs_in, tensor_shapes)] \
    + [einx.expr.Condition(expr=expr_out, value=output_shape, shape=(output_ndims,) if not output_ndims is None else None, depth=0) for expr_out in exprs_out] \
    + [einx.expr.Condition(expr=k, value=np.asarray(v)[..., np.newaxis]) for k, v in parameters.items()],
    verbose=True,
)

exprs_in, exprs_out = exprs[:len(exprs_in)], exprs[len(exprs_in):len(exprs_in) + len(exprs_out)]

for expr, shape in zip(exprs_in, tensor_shapes):
    print("IN", expr, shape)

tensors_out = einx.op.rearrange(op_string, tensors_in, **parameters)#
# tensor_out, expr_out = einx.op.add(exprs_in, tensors_in, exprs_out[0], einx.backend.numpy)

# def func(a, b, c): # b, b -> b, 1
#     return jnp.concatenate([a + b, np.asarray([2])], axis=0)
# tensors_out, exprs_out = einx.op.vmap(exprs_in, tensors_in, exprs_out, backend=einx.backend.get_by_name("jax"), op=func)

for expr_out, tensor_out in zip(exprs_out, tensors_out):
    print("OUT", expr_out, tensor_out.shape, tensor_out)


# for expr, shape in zip(exprs_in, tensor_shapes):
#     print("IN", expr, shape)

# tensors_out, exprs_out2 = einx.op.sum(exprs_in, tensors_in, exprs_out, einx.backend.numpy)
# assert exprs_out2 == exprs_out

# for expr_out, tensor_out in zip(exprs_out, tensors_out):
#     print("OUT", expr_out, tensor_out.shape, tensor_out)


print("DONE")
import sys
sys.exit(0)

# x = jnp.ones((16, 128, 128, 64))
# # w1 = jnp.ones((128, 128, 512))
# # w2 = jnp.ones((512, 128, 128))

# # @jax.jit
# # def func1(x, w1, w2):
# #     x = einx.dot("b [s...|s2] c", x, w1)
# #     x = jax.nn.relu(x)
# #     x = einx.dot("b [s2|s...] c", x, w2)
# #     return x

# # @jax.jit
# # def func2(x, w1, w2):
# #     x = einx.rearrange("b s... c -> b c s...", x)

# #     x = einx.dot("b c [s...|s2]", x, w1)
# #     x = jax.nn.relu(x)
# #     x = einx.dot("b c [s2|s...]", x, w2)

# #     x = einx.rearrange("b c s... -> b s... c", x)
# #     return x

# # @jax.jit
# # def func3(x, w1, w2):
# #     x = einx.rearrange("b s... c -> s... b c", x)
# #     x = einx.dot("[s...|s2] b c", x, w1)
# #     x = jax.nn.relu(x)
# #     x = einx.dot("[s2|s...] b c", x, w2)
# #     x = einx.rearrange("s... b c -> b s... c", x)
# #     return x

# # func1(x, w1, w2)
# # func2(x, w1, w2)
# # func3(x, w1, w2)

# # print(timeit.timeit(lambda: func1(x, w1, w2).block_until_ready(), number=1000))
# # print(timeit.timeit(lambda: func2(x, w1, w2).block_until_ready(), number=1000))
# # print(timeit.timeit(lambda: func3(x, w1, w2).block_until_ready(), number=1000))


print("Devices:", jax.devices())


c = 500
n = 1
n2 = 10

if True:
    x = jnp.ones((16, 128, 128, 64)).astype("float32")
    scale = jnp.ones((64,)).astype("float32")
    offset = jnp.ones((64,)).astype("float32")

    def layernorm0(x):
        for _ in range(n):
            mean = einx.mean("b... [c]", x)
            var = einx.var("b... [c]", x)

            inv = einx.multiply("b..., c -> b... c", jax.lax.rsqrt(var), scale)
            x = einx.subtract("[b...] c", x, mean)
            x = inv * x
            x = einx.add("b... [c]", x, offset)
        return x

    def layernorm1(x):
        for _ in range(n):
            mean = einx.mean("b... ([c])", x)
            var = einx.var("b... ([c])", x)
            x = (x - mean) * jax.lax.rsqrt(var)

            x = einx.multiply("b... [c]", x, scale)
            x = einx.add("b... [c]", x, offset)
        return x

    def layernorm2(x):
        for _ in range(n):
            mean = jnp.mean(x, axis=-1, keepdims=True)
            var = jnp.var(x, axis=-1, keepdims=True)

            inv = scale * jax.lax.rsqrt(var)
            x = inv * (x - mean) + offset
        return x

    def layernorm3(x):
        for _ in range(n):
            mean = einx.mean("b... ([c])", x)
            var = einx.var("b... ([c])", x)
            x = (x - mean) / jax.lax.sqrt(var)

            x = einx.multiply("b... [c]", x, scale)
            x = einx.add("b... [c]", x, offset)
        return x

    def layernorm4(x,):
        for _ in range(n):
            x = einx.dl.meanvar_norm(x, "b... [c]", scale=scale, bias=offset)
        return x

    methods = [layernorm0, layernorm1, layernorm2, layernorm3, layernorm4]

    for method in methods:
        print(method.__name__)
        #print(jax.make_jaxpr(method)(x))
        print(jax.jit(method).lower(x).compile().as_text())
        print()
    assert False

    print("Initializing")
    methods = [jax.jit(m) for m in methods]
    import random
    random.shuffle(methods)
    for _ in range(2):
        for method in methods:
            method(x)

    print("Benchmarking")
    random.shuffle(methods)
    from collections import defaultdict
    method_to_time = defaultdict(lambda: 0)
    for _ in range(c):
        random.shuffle(methods)
        for method in methods:
            sec = timeit.timeit(lambda: method(x).block_until_ready(), number=n2)
            method_to_time[method] += sec
    for method, sec in sorted(method_to_time.items(), key=lambda t: str(t[0])):
        print(method, sec / c)



    assert False

if True:
    import torch, einops, torch._dynamo

    x = torch.ones((16, 128, 128, 64))
    eps = 1e-5

    layer1 = torch.nn.LayerNorm((64,), eps=eps, elementwise_affine=True)

    layer2 = einx.dl.torch.Norm("b... [c]", epsilon=eps)

    def layernorm0(x):
        with torch.no_grad():
            for _ in range(n):
                mean = x.mean(dim=-1, keepdims=True)
                var = x.var(dim=-1, keepdims=True)

                inv = layer1.weight * torch.rsqrt(var + eps)
                x = (x - mean) * inv
                x = x + layer1.bias
        return x

    def layernorm1(x):
        with torch.no_grad():
            for _ in range(n):
                x = layer1(x)
        return x

    def layernorm2(x):
        with torch.no_grad():
            for _ in range(n):
                mean = einops.reduce(x, "b h w c -> b h w 1", "mean")
                var = x.var(dim=-1, keepdims=True)

                inv = layer1.weight * torch.rsqrt(var + eps)
                x = (x - mean) * inv
                x = x + layer1.bias
        return x

    def layernorm3(x):
        with torch.no_grad():
            for _ in range(n):
                mean = einx.mean("b... [c]", x, keepdims=True)
                var = x.var(dim=-1, keepdims=True)

                inv = layer1.weight * torch.rsqrt(var + eps)
                x = (x - mean) * inv
                x = x + layer1.bias
        return x

    # import torch
    # einx.backend.update_available_backends()

    # meanvar_norm = torch.compile(lambda *args, **kwargs: einx.dl.meanvar_norm(*args, **kwargs))

    def layernorm4(x):
        with torch.no_grad():
            for _ in range(n):
                x = einx.dl.meanvar_norm(x, "b... [c]", scale=layer1.weight, bias=layer1.bias)
        return x

    # def layernorm5(x):
    #     with torch.no_grad():
    #         for _ in range(3):
    #             x = layer2(x)
    #     return x

    methods = [layernorm0, layernorm1, layernorm2, layernorm3, layernorm4] #, layernorm5]

    print("Initializing")
    import random
    random.shuffle(methods)
    for _ in range(2):
        for method in methods:
            method(x)

    # layer2 = torch.compile(layer2)

    layer1 = layer1.to(torch.float32).cuda()
    layer2 = layer2.to(torch.float32).cuda()
    x = x.cuda()

    random.shuffle(methods)
    for _ in range(2):
        for method in methods:
            method(x)


    # from torch._dynamo.utils import CompileProfiler
    # prof = CompileProfiler()

    methods = [torch.compile(method) for method in methods] #, backend=prof
    for _ in range(2):
        for method in methods:
            method(x)
        # print(prof.report())
        # assert False

    print("Benchmarking")
    random.shuffle(methods)
    from collections import defaultdict
    method_to_time = defaultdict(lambda: 0)
    for _ in range(c):
        random.shuffle(methods)
        for method in methods:
            sec = timeit.timeit(lambda: method(x).to(torch.device("cuda"), non_blocking=False), number=n2)
            method_to_time[method] += sec
    for method, sec in sorted(method_to_time.items(), key=lambda t: str(t[0])):
        print(method, sec / c)

    assert False

assert False

import einops, einx
import numpy as np

x = np.ones((10, 10, 10))

expr = "b... (g [c])"

mean = einx.mean(expr, x)

assert False


import jax.numpy as np
x = jnp.ones((10, 10))

def func():
    for _ in range(2):
        x = jnp.zeros((10, 5, 1), "float32")
        y = jnp.zeros((13,), "float32")
        assert einx.elementwise("a b 1, l -> b l a 1", x, y, op=jnp.add).shape == (5, 13, 10, 1)
        assert einx.add("a b 1, l -> b l a 1", x, y).shape == (5, 13, 10, 1)
        assert einx.add("a b 1, l -> a b l", x, y).shape == (10, 5, 13)

        x = jnp.zeros((10, 10), "float32")
        y = jnp.zeros((10,), "float32")
        assert einx.add("a, a b", y, x).shape == (10, 10)
        assert einx.add("a b, a", x, y).shape == (10, 10)
        assert einx.add("a b, b", x, y).shape == (10, 10)
        assert einx.add("a [b]", x, y).shape == (10, 10)
        assert einx.add("a b, a b", x, x).shape == (10, 10)
        assert einx.add("a b, ", x, 1).shape == (10, 10)
        assert einx.add(", a b", 1, x).shape == (10, 10)
        assert einx.add("a b, 1", x, [1]).shape == (10, 10)
        assert einx.add("1, a b", [1], x).shape == (10, 10)

print(timeit.timeit(func, number=50))

import sys
sys.exit(0)




import torch

class Linear(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.weight = (torch.nn.parameter.UninitializedParameter(), torch.nn.init.normal_)
        self.bias = (torch.nn.parameter.UninitializedParameter(), torch.nn.init.zeros_)

    def forward(self, x):
        x = einx.dot("b... [c1|c2]", x, self.weight, c2=self.channels)
        x = einx.add("b... [c]", x, self.bias)
        return x

x = torch.zeros((4, 128, 128, 3))
layer = Linear(32)
print("AAAAAAAAAA")
print(layer.forward(x).shape)
print(layer)
assert False


import haiku as hk
def hk_param(name, init, is_param=True):
    get = hk.get_parameter if is_param else hk.get_state
    return lambda shape, dtype: get(name, shape, dtype=dtype, init=init)

from typing import (Any, Callable, Tuple)
import flax
class Param(flax.linen.module.Module):
    name: str
    shape: Tuple
    dtype: Any
    init: Callable
    is_param: bool = True

    @flax.linen.compact
    def __call__(self, inputs):
        return self.param(self.name, self.kernel_init, self.shape, self.dtype)

def flax_param(name, init, is_param=True):
    return lambda shape, dtype: Param(name, shape, dtype, init, is_param)


# Grouped




x = np.ones((64, 128))
y = np.arange(128)

x = einx.elementwise("a [b]", x, y, op=np.add)
print(x.shape)