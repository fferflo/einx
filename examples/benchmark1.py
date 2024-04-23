import torch
import jax
import einx
import timeit
import types
import einops
import random
import argparse
import math
import jax.numpy as jnp
import numpy as np
from functools import partial
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument("--n", type=int, default=1000)
args = parser.parse_args()

k = 1
n = args.n // k
rows = []

envs = [
    types.SimpleNamespace(
        name="numpy",
        backend=einx.backend.get("numpy"),
        jit=lambda x: x,
        block_until_ready=lambda x: x,
        to_numpy=np.asarray,
        ones=lambda shape, dtype: np.ones(shape, dtype=dtype),
        transpose=np.transpose,
        mean=np.mean,
        einsum=np.einsum,
    ),
    types.SimpleNamespace(
        name="torch-eager",
        backend=einx.backend.get("torch"),
        jit=lambda x: x,
        block_until_ready=lambda x: torch.cuda.synchronize(),
        to_numpy=lambda x: np.asarray(x.cpu()),
        ones=lambda shape, dtype: torch.ones(shape, dtype=vars(torch)[dtype]).cuda(),
        transpose=torch.permute,
        mean=torch.mean,
        einsum=torch.einsum,
    ),
    types.SimpleNamespace(
        name="torch-compile",
        backend=einx.backend.get("torch"),
        jit=torch.compile,
        block_until_ready=lambda x: torch.cuda.synchronize(),
        to_numpy=lambda x: np.asarray(x.cpu()),
        ones=lambda shape, dtype: torch.ones(shape, dtype=vars(torch)[dtype]).cuda(),
        transpose=torch.permute,
        mean=torch.mean,
        einsum=torch.einsum,
    ),
    types.SimpleNamespace(
        name="jax-jit",
        backend=einx.backend.get("jax"),
        jit=jax.jit,
        block_until_ready=lambda x: x.block_until_ready(),
        to_numpy=lambda x: np.asarray(x),
        ones=lambda shape, dtype: jnp.ones(shape, dtype=dtype),
        transpose=jnp.transpose,
        mean=jnp.mean,
        einsum=jnp.einsum,
    ),
]

for env in envs:
    experiments = []

    f = 4 if env.name == "numpy" else 1

    x = env.ones((16 // f, 512 // f, 512 // f, 64 // f), "float32")
    x2 = env.ones((16 // f, 256 // f, 256 // f, 64 // f), "float32")
    y = env.ones((512 // f, 512 // f), "float32")
    z1 = env.ones((64 // f,), "float32")
    w = env.ones((64 // f, 128 // f), "float32")

    def benchmark_einx(x):
        return einx.rearrange("b h w c -> b c h w", x, backend=env.backend)

    def benchmark_einops(x):
        return einops.rearrange(x, "b h w c -> b c h w")

    def benchmark_idx(x):
        return env.transpose(x, (0, 3, 1, 2))

    experiments.append(("rearrange", (benchmark_einx, benchmark_einops, benchmark_idx), (x,), 5.0))

    def benchmark_einx(x):
        return einx.mean("b [s...] c", x)

    def benchmark_einops(x):
        return einops.reduce(x, "b h w c -> b c", reduction="mean")

    def benchmark_idx(x):
        return env.mean(x, axis=(1, 2))

    experiments.append((
        "spatial_mean",
        (benchmark_einx, benchmark_einops, benchmark_idx),
        (x,),
        5.0,
    ))

    def benchmark_einx(x):
        return einx.mean("b s... [c]", x)

    def benchmark_einops(x):
        return einops.reduce(x, "b h w c -> b h w", reduction="mean")

    def benchmark_idx(x):
        return env.mean(x, axis=3)

    experiments.append((
        "channel_mean",
        (benchmark_einx, benchmark_einops, benchmark_idx),
        (x,),
        5.0,
    ))

    def benchmark_einx(x, y):
        return einx.add("b [s...] c", x, y)

    def benchmark_idx(x, y):
        return x + y[None, ..., None]

    experiments.append(("spatial_add", (benchmark_einx, None, benchmark_idx), (x, y), 5.0))

    def benchmark_einx(x, y):
        return einx.add("b s... [c]", x, y)

    def benchmark_idx(x, y):
        return x + y

    experiments.append(("channel_add", (benchmark_einx, None, benchmark_idx), (x, z1), 5.0))

    def benchmark_einx(x, w):
        return einx.dot("b... [c1->c2]", x, w)

    def benchmark_einops(x, w):
        return einops.einsum(x, w, "... c1, c1 c2 -> ... c2")

    def benchmark_idx(x, w):
        return env.einsum("b h w c, c d -> b h w d", x, w)

    experiments.append(("einsum", (benchmark_einx, benchmark_einops, benchmark_idx), (x2, w), 5.0))

    for name, methods, inputs, mul in experiments:
        name = env.name + " " + name
        print(name)

        # Assert correctness
        results = []
        for method in methods:
            if method is not None:
                results.append(method(*inputs))
        results = [env.to_numpy(r) for r in results]
        for r2 in results[1:]:
            assert np.allclose(results[0], r2)

        # Initialization
        for _ in range(5):
            for method in methods:
                if method is not None:
                    env.block_until_ready(method(*inputs))
        methods = [env.jit(m) if m is not None else None for m in methods]
        for _ in range(5):
            for method in methods:
                if method is not None:
                    env.block_until_ready(method(*inputs))

        # Benchmark
        times = defaultdict(list)
        order = "random"
        if order == "random":
            methods2 = list(methods)
            for _ in range(max(1, int(n * mul))):
                random.shuffle(methods2)
                for method in methods2:
                    if method is not None:
                        times[method.__name__].append(
                            timeit.repeat(
                                lambda: env.block_until_ready(method(*inputs)), repeat=1, number=k
                            )[0]
                            / k
                        )
        elif order == "sequential":
            for method in methods:
                if method is not None:
                    for _ in range(max(1, int(n * mul))):
                        times[method.__name__].append(
                            timeit.repeat(
                                lambda: env.block_until_ready(method(*inputs)), repeat=1, number=k
                            )[0]
                            / k
                        )
        else:
            raise AssertionError()

        # Store and print results
        for key in list(times.keys()):
            p = int(len(times[key]) * 0.2)
            times[key] = sorted(times[key])[p:-p]
        for method in methods:
            if method is not None:
                print(
                    f"{method.__name__:>25}: {1000.0 * np.mean(times[method.__name__]):0.6f} "
                    f"+- {1000.0 * np.std(times[method.__name__]):0.6f}"
                )
        rows.append((name, times))
        print()

# Print markup table
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
        1000000.0 * (np.mean(times["benchmark_einx"]) - np.mean(times["benchmark_idx"])),
        1000000.0 * (np.mean(times["benchmark_einops"]) - np.mean(times["benchmark_idx"]))
        if "benchmark_einops" in times
        else "",
        tostr(1000.0 * times["benchmark_einx"]),
        tostr(1000.0 * times["benchmark_einops"]) if "benchmark_einops" in times else "",
        tostr(1000.0 * times["benchmark_idx"]),
    ])
print(
    tabulate.tabulate(
        table,
        headers=[
            "Method",
            "einx overhead (us)",
            "einops overhead (us)",
            "einx (ms)",
            "einops (ms)",
            "index-based (ms)",
        ],
        tablefmt="github",
    )
)
