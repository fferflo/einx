import flax.linen as nn
import einx
from functools import partial
import jax.numpy as jnp

# Using _ prefix on classes and a separater constructor, since dataclass/nn.Module does not support **kwargs parameter.

class _Norm(nn.Module):
    stats: str
    params: str = "b... [c]"
    mean: bool = True
    var: bool = True
    scale: bool = True
    bias: bool = True
    decay_rate: float = None
    epsilon: float = 1e-5
    dtype: str = "float32"
    kwargs: dict = None

    def moving_average(self, f, name, training):
        if self.decay_rate is None:
            return f()
        else:
            if training is None:
                raise ValueError("training must be specified when decay_rate is not None")
            if training:
                x = f()

                if name == "mean":
                    assert self.mean
                    ema = self.variable("stats", "mean", lambda: jnp.zeros(x.shape, self.dtype))
                elif name == "var":
                    assert self.var
                    ema = self.variable("stats", "var", lambda: jnp.ones(x.shape, self.dtype))
                else:
                    assert False

                if not self.is_initializing():
                    ema.value = self.decay_rate * ema.value + (1 - self.decay_rate) * x
                return x
            else:
                if name == "mean":
                    assert self.mean
                    ema = self.variable("stats", "mean", None)
                elif name == "var":
                    assert self.var
                    ema = self.variable("stats", "var", None)
                else:
                    assert False

                return ema.value

    @nn.compact
    def __call__(self, x, training=None):
        return einx.nn.meanvar_norm(
            x,
            self.stats,
            self.params,
            moving_average=partial(self.moving_average, training=training),
            mean=self.mean,
            var=self.var,
            scale=lambda shape: self.param("scale", nn.initializers.ones_init(), shape, self.dtype) if self.scale else None,
            bias=lambda shape: self.param("bias", nn.initializers.zeros_init(), shape, self.dtype) if self.bias else None,
            epsilon=self.epsilon,
            **(self.kwargs if not self.kwargs is None else {}),
        )

def Norm(stats, params="b... [c]", mean=True, var=True, scale=True, bias=True, decay_rate=None, epsilon=1e-5, dtype="float32", **kwargs):
    return _Norm(stats, params=params, mean=mean, var=var, scale=scale, bias=bias, decay_rate=decay_rate, epsilon=epsilon, dtype=dtype, kwargs=kwargs)

class _Linear(nn.Module):
    expr: str
    bias: bool = True
    dtype: str = "float32"
    kwargs: dict = None

    @nn.compact
    def __call__(self, x):
        return einx.nn.linear(
            x,
            self.expr,
            bias=lambda shape: self.param("bias", nn.initializers.zeros_init(), shape, self.dtype) if self.bias else None,
            weight=lambda shape, in_axis, out_axis, batch_axis: self.param("weight", nn.initializers.lecun_normal(in_axis, out_axis, batch_axis), shape, self.dtype),
            **(self.kwargs if not self.kwargs is None else {}),
        )

def Linear(expr, bias=True, dtype="float32", **kwargs):
    return _Linear(expr, bias=bias, dtype=dtype, kwargs=kwargs)

class _Dropout(nn.Module):
    expr: str
    drop_rate: float
    rng_collection: str = "dropout"
    kwargs: dict = None

    @nn.compact
    def __call__(self, x, training):
        if training:
            return einx.nn.dropout(
                x,
                self.expr,
                drop_rate=self.drop_rate,
                rng=self.make_rng(self.rng_collection),
                **(self.kwargs if not self.kwargs is None else {}),
            )
        else:
            return x

def Dropout(expr, drop_rate, rng_collection="dropout", **kwargs):
    return _Dropout(expr, drop_rate, rng_collection=rng_collection, kwargs=kwargs)