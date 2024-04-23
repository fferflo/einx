import torch
import jax
import einx
import timeit
import einops
import random
import argparse
import math
import gc
import types
import jax.numpy as jnp
import numpy as np
from functools import partial
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument("--n", type=int, default=100)
args = parser.parse_args()

k = 1
n = args.n // k
rows = []

envs = [
    types.SimpleNamespace(
        name="torch-eager",
        backend=einx.backend.get("torch"),
        jit=lambda x: x,
        block_until_ready=lambda x: torch.cuda.synchronize(),
        to_numpy=lambda x: np.asarray(x.cpu()),
        ones=lambda shape, dtype="float32": torch.ones(shape, dtype=vars(torch)[dtype]).cuda(),
        transpose=torch.permute,
        mean=torch.mean,
        var=torch.var,
        square=torch.square,
        einsum=torch.einsum,
        swapaxes=torch.swapaxes,
        rsqrt=torch.rsqrt,
        where=torch.where,
        dot=torch.matmul,
        softmax=lambda x, axis: torch.nn.functional.softmax(x, axis),
        native_transposed=True,
    ),
    types.SimpleNamespace(
        name="torch-compile",
        backend=einx.backend.get("torch"),
        jit=torch.compile,
        block_until_ready=lambda x: torch.cuda.synchronize(),
        to_numpy=lambda x: np.asarray(x.cpu()),
        ones=lambda shape, dtype="float32": torch.ones(shape, dtype=vars(torch)[dtype]).cuda(),
        transpose=torch.permute,
        mean=torch.mean,
        var=torch.var,
        square=torch.square,
        einsum=torch.einsum,
        swapaxes=torch.swapaxes,
        rsqrt=torch.rsqrt,
        where=torch.where,
        dot=torch.matmul,
        softmax=lambda x, axis: torch.nn.functional.softmax(x, axis),
        native_transposed=True,
    ),
    types.SimpleNamespace(
        name="jax-jit",
        backend=einx.backend.get("jax"),
        jit=jax.jit,
        block_until_ready=lambda x: x.block_until_ready(),
        to_numpy=lambda x: np.asarray(x),
        ones=lambda shape, dtype="float32": jnp.ones(shape, dtype=dtype),
        transpose=jnp.transpose,
        mean=jnp.mean,
        var=jnp.var,
        square=jnp.square,
        einsum=jnp.einsum,
        swapaxes=jnp.swapaxes,
        rsqrt=jnp.sqrt,
        where=jnp.where,
        dot=jnp.dot,
        softmax=jax.nn.softmax,
        native_transposed=False,
    ),
]

k = int(math.sqrt(args.n))

for env in envs:
    experiments = []

    f = 1

    x = env.ones((16 // f, 512 // f, 512 // f, 64 // f), "float32")
    if "torch" in env.name:
        x_transposed = env.ones((16 // f, 64 // f, 512 // f, 512 // f), "float32")
        x_transposed[:] = einx.rearrange("b s... c -> b c s...", x)
    y = (env.ones((512 // f, 512 // f)))
    z1 = (env.ones((64 // f,), "float32"))
    z2 = (env.ones((64 // f,), "float32"))
    w = (env.ones((64 // f, 128 // f), "float32"))
    if "torch" in env.name:
        w_transposed = (env.ones((128 // f, 64 // f), "float32"))
        w_transposed[:] = w.T
    w1 = (env.ones((512 // f, 512 // f, 128 // f)))
    w2 = (env.ones((128 // f, 512 // f, 512 // f)))
    b128 = (env.ones((128 // f,), "float32"))
    epsilon = 1e-5

    query = (env.ones((16 // f, 512 // f, 512 // f), "float32"))
    key = (env.ones((16 // f, 512 // f, 512 // f), "float32"))
    value = (env.ones((16 // f, 512 // f, 512 // f), "float32"))

    def benchmark_einx(x, bias, scale):
        return einx.nn.norm(x, "b... [c]", bias=bias, scale=scale, epsilon=epsilon, fastvar=False)[
            0
        ]

    def benchmark_idx(x, bias, scale):
        mean = env.mean(x, axis=-1, keepdims=True)
        var = env.var(x, axis=-1, keepdims=True)

        inv = scale * env.rsqrt(var + epsilon)
        x = inv * (x - mean) + bias

        return x

    if "torch" in env.name:

        def benchmark_native(x, bias, scale):
            return torch.nn.functional.layer_norm(
                x, (x.shape[-1],), weight=scale, bias=bias, eps=epsilon
            )
    else:
        benchmark_native = None
    experiments.append((
        "layernorm",
        (benchmark_einx, benchmark_native, benchmark_idx),
        lambda m: (x, z1, z2),
        3.0,
    ))

    def benchmark_einx(x, bias, scale):
        return einx.nn.norm(x, "b... [c]", bias=bias, scale=scale, epsilon=epsilon, fastvar=True)[0]

    def benchmark_idx(x, bias, scale):
        # https://github.com/deepmind/dm-haiku/blob/main/haiku/_src/layer_norm.py
        mean = env.mean(x, axis=-1, keepdims=True)
        mean_of_squares = env.mean(env.square(x), axis=-1, keepdims=True)
        var = mean_of_squares - env.square(mean)

        inv = scale * env.rsqrt(var + epsilon)
        x = inv * (x - mean) + bias

        return x

    if "torch" in env.name:

        def benchmark_native(x, bias, scale):
            return torch.nn.functional.layer_norm(
                x, (x.shape[-1],), weight=scale, bias=bias, eps=epsilon
            )
    else:
        benchmark_native = None
    experiments.append((
        "layernorm_fastvar",
        (benchmark_einx, benchmark_native, benchmark_idx),
        lambda m: (x, z1, z2),
        3.0,
    ))

    def benchmark_einx(x, bias, scale):
        return einx.nn.norm(x, "[b...] c", bias=bias, scale=scale, epsilon=epsilon, fastvar=False)[
            0
        ]

    def benchmark_idx(x, bias, scale):
        mean = env.mean(x, axis=(1, 2), keepdims=True)
        var = env.var(x, axis=(1, 2), keepdims=True)

        inv = scale * env.rsqrt(var + epsilon)
        x = inv * (x - mean) + bias

        return x

    if "torch" in env.name:

        def benchmark_native(x, bias, scale):
            return torch.nn.functional.batch_norm(
                x, None, None, weight=scale, bias=bias, eps=epsilon, training=True
            )
    else:
        benchmark_native = None
    experiments.append((
        "batchnorm",
        (benchmark_einx, benchmark_native, benchmark_idx),
        lambda m: (x_transposed if env.native_transposed and "native" in m.__name__ else x, z1, z2),
        3.0,
    ))

    def benchmark_einx(x, bias, scale):
        return einx.nn.norm(x, "[b...] c", bias=bias, scale=scale, epsilon=epsilon, fastvar=True)[0]

    def benchmark_idx(x, bias, scale):
        # https://github.com/deepmind/dm-haiku/blob/main/haiku/_src/batch_norm.py
        mean = env.mean(x, axis=(1, 2), keepdims=True)
        mean_of_squares = env.mean(env.square(x), axis=(1, 2), keepdims=True)
        var = mean_of_squares - env.square(mean)

        inv = scale * env.rsqrt(var + epsilon)
        x = inv * (x - mean) + bias

        return x

    if "torch" in env.name:

        def benchmark_native(x, bias, scale):
            return torch.nn.functional.batch_norm(
                x, None, None, weight=scale, bias=bias, eps=epsilon, training=True
            )
    else:
        benchmark_native = None
    experiments.append((
        "batchnorm_fastvar",
        (benchmark_einx, benchmark_native, benchmark_idx),
        lambda m: (x_transposed if env.native_transposed and "native" in m.__name__ else x, z1, z2),
        3.0,
    ))

    def benchmark_einx(x, bias, weight):
        return einx.nn.linear(x, "b... [c1->c2]", bias=bias, weight=weight)

    def benchmark_idx(x, bias, weight):
        # https://github.com/deepmind/dm-haiku/blob/main/haiku/_src/basic.py
        x = env.dot(x, weight)
        x = x + bias
        return x

    if "torch" in env.name:

        def benchmark_native(x, bias, weight):
            return torch.nn.functional.linear(x, weight=weight, bias=bias)
    else:
        benchmark_native = None
    experiments.append((
        "channel_linear",
        (benchmark_einx, benchmark_native, benchmark_idx),
        lambda m: (x, b128, w_transposed if env.native_transposed and "native" in m.__name__ else w),
        1.0,
    ))

    def benchmark_einx(x, w1, b1, w2, b2):
        x0 = x
        x = einx.nn.linear(x, "b [s...->s2] c", weight=w1, bias=b1)
        x = env.where(x < 0, 0, x)
        x = einx.nn.linear(x, "b [s2->s...] c", weight=w2, bias=b2)
        x = x + x0
        return x

    def benchmark_idx(x, w1, b1, w2, b2):
        # https://github.com/lucidrains/mlp-mixer-pytorch/blob/main/mlp_mixer_pytorch/mlp_mixer_pytorch.py
        # https://github.com/google-research/vision_transformer/blob/main/vit_jax/models_mixer.py
        x0 = x
        shape = x.shape
        x = x.reshape([x.shape[0], -1, x.shape[-1]])
        x = env.swapaxes(x, 1, 2)
        x = env.dot(x, w1.reshape([-1, w1.shape[-1]]))
        x = x + b1
        x = env.where(x < 0, 0, x)
        x = env.dot(x, w2.reshape([w2.shape[0], -1]))
        x = x + b2.reshape([-1])
        x = env.swapaxes(x, 1, 2)
        x = x.reshape(shape)
        x = x + x0
        return x

    experiments.append((
        "spatial_mlp",
        (benchmark_einx, None, benchmark_idx),
        lambda m: (x, w1, b128, w2, y),
        1.0,
    ))

    heads = 8

    def benchmark_einx(q, k, v, heads=heads):
        attn = einx.dot("b q (h c), b k (h c) -> b q k h", q, k, h=heads)
        attn = einx.softmax("b q [k] h", attn)
        x = einx.dot("b q k h, b k (h c) -> b q (h c)", attn, v)
        return x

    def benchmark_idx(q, k, v, heads=heads):
        q = einops.rearrange(q, "b l (h k) -> b h l k", h=heads)
        k = einops.rearrange(k, "b t (h k) -> b h t k", h=heads)
        v = einops.rearrange(v, "b t (h v) -> b h t v", h=heads)
        attn = env.einsum("bhlk,bhtk->bhlt", q, k)
        attn = env.softmax(attn, axis=3)
        x = env.einsum("bhlt,bhtv->bhlv", attn, v)
        x = einops.rearrange(x, "b h l v -> b l (h v)")
        return x

    experiments.append((
        "multihead-attention",
        (benchmark_einx, None, benchmark_idx),
        lambda m: (query, key, value),
        1.0,
    ))

    for name, methods, inputs, mul in experiments:
        name = env.name + " " + name
        print(name)

        results = []
        for method in methods:
            if method is not None:
                r = method(*inputs(method))
                if "batchnorm" in name and "torch" in env.name and "native" in method.__name__:
                    r = einx.rearrange("b c s... -> b s... c", r)
                results.append(r)
        results = [env.to_numpy(r) for r in results]
        for r2 in results[1:]:
            assert np.allclose(results[0], r2)

        for _ in range(5):
            for method in methods:
                if method is not None:
                    env.block_until_ready(method(*inputs(method)))
        methods = [env.jit(m) if m is not None else None for m in methods]
        for _ in range(5):
            for method in methods:
                if method is not None:
                    env.block_until_ready(method(*inputs(method)))

        times = defaultdict(list)
        order = "random"
        if order == "random":
            methods2 = list(methods)
            for _ in range(max(1, int(n * mul))):
                random.shuffle(methods2)
                for method in methods2:
                    if method is not None:
                        inputs2 = inputs(method)
                        times[method.__name__].append(
                            timeit.repeat(
                                lambda: env.block_until_ready(method(*inputs2)), repeat=1, number=k
                            )[0]
                            / k
                        )
        elif order == "sequential":
            for method in methods:
                if method is not None:
                    inputs2 = inputs(method)
                    for _ in range(max(1, int(n * mul))):
                        times[method.__name__].append(
                            timeit.repeat(
                                lambda: env.block_until_ready(method(*inputs2)), repeat=1, number=k
                            )[0]
                            / k
                        )
        else:
            raise AssertionError()

        for key2 in list(times.keys()):
            p = int(len(times[key2]) * 0.2)
            times[key2] = sorted(times[key2])[p:-p]

        # if "benchmark_native" not in times:
        #     times["benchmark_native"] = times["benchmark_idx"]

        for method in methods:
            if method is not None:
                print(
                    f"{method.__name__:>25}: {1000.0 * np.mean(times[method.__name__]):0.6f} "
                    f"+- {1000.0 * np.std(times[method.__name__]):0.6f}"
                )
        rows.append((name, times))
        print()

    del x, y, z1, z2, w, w1, w2, b128, query, key, value
    gc.collect()

import tabulate

table = []


def tostr(times):
    if len(times) == 0 or times is None:
        return ""
    m = f"{np.mean(times):0.3f}"
    s = f"{np.std(times):0.3f}"
    return f"{m:>7} +- {s:>7}"


for name, times in rows:
    times = {k: np.asarray(v) for k, v in times.items()}
    table.append([
        name,
        1000000.0
        * (
            np.mean(times["benchmark_einx"])
            - np.mean(
                times["benchmark_native"] if "benchmark_native" in times else times["benchmark_idx"]
            )
        ),
        tostr(1000.0 * times["benchmark_einx"]),
        tostr(1000.0 * times["benchmark_native"]) if "benchmark_native" in times else "",
        tostr(1000.0 * times["benchmark_idx"]),
    ])
print(
    tabulate.tabulate(
        table,
        headers=["Method", "einx overhead (us)", "einx (ms)", "native (ms)", "index-based (ms)"],
        tablefmt="github",
    )
)
