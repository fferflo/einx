import flax.linen as nn
import einx
from functools import partial
import jax.numpy as jnp

class Norm(nn.Module):
    stats: str
    params: str = "b... [c]"
    mean: bool = True
    var: bool = True
    scale: bool = True
    bias: bool = True
    decay_rate: float = None
    epsilon: float = 1e-5

    def moving_average(self, f, name, is_training):
        if self.decay_rate is None:
            return f()
        else:
            if is_training is None:
                raise ValueError("is_training must be specified when decay_rate is not None")
            if is_training:
                x = f()

                if name == "mean":
                    assert self.mean
                    ema = self.variable("stats", "mean", lambda: jnp.zeros(x.shape, "float32"))
                elif name == "var":
                    assert self.var
                    ema = self.variable("stats", "var", lambda: jnp.ones(x.shape, "float32"))
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
    def __call__(self, x, is_training=None):
        return einx.nn.meanvar_norm(
            x,
            self.stats,
            self.params,
            moving_average=partial(self.moving_average, is_training=is_training),
            mean=self.mean,
            var=self.var,
            scale=lambda shape: self.param("scale", nn.initializers.ones_init(), shape, "float32") if self.scale else None,
            bias=lambda shape: self.param("bias", nn.initializers.zeros_init(), shape, "float32") if self.bias else None,
            epsilon=self.epsilon,
        )

class Linear(nn.Module):
    expr: str
    bias: bool = True

    @nn.compact
    def __call__(self, x, **kwargs):
        return einx.nn.linear(
            x,
            self.expr,
            bias=lambda shape: self.param("bias", nn.initializers.zeros_init(), shape, "float32") if self.bias else None,
            weight=lambda shape: self.param("weight", nn.initializers.lecun_normal(), shape, "float32"),
            **kwargs,
        )
