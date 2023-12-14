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
    fastvar: bool = True
    dtype: str = "float32"
    kwargs: dict = None

    @nn.compact
    def __call__(self, x, training=None):
        if not self.decay_rate is None and training is None:
            raise ValueError("training must be specified when decay_rate is used")

        use_ema = not self.decay_rate is None and (not training or self.is_initializing())
        x, mean, var = einx.nn.norm(
            x,
            self.stats,
            self.params,
            mean=(lambda shape: self.variable("stats", "mean", lambda: jnp.zeros(shape, self.dtype)).value) if use_ema else self.mean,
            var=(lambda shape: self.variable("stats", "var", lambda: jnp.ones(shape, self.dtype)).value) if use_ema else self.var,
            scale=(lambda shape: self.param("scale", nn.initializers.ones_init(), shape, self.dtype)) if self.scale else None,
            bias=(lambda shape: self.param("bias", nn.initializers.zeros_init(), shape, self.dtype)) if self.bias else None,
            epsilon=self.epsilon,
            fastvar=self.fastvar,
            **(self.kwargs if not self.kwargs is None else {}),
        )

        update_ema = not self.decay_rate is None and training and not self.is_initializing()
        if update_ema:
            if self.mean:
                mean_ema = self.variable("stats", "mean", None)
                mean_ema.value = self.decay_rate * mean_ema.value + (1 - self.decay_rate) * mean
            if self.var:
                var_ema = self.variable("stats", "var", None)
                var_ema.value = self.decay_rate * var_ema.value + (1 - self.decay_rate) * var

        return x

def Norm(stats, params="b... [c]", mean=True, var=True, scale=True, bias=True, decay_rate=None, epsilon=1e-5, fastvar=True, dtype="float32", name=None, **kwargs):
    """Normalization layer.

    Args:
        stats: Einstein string determining the axes along which mean and variance are computed. Will be passed to ``einx.reduce``.
        params: Einstein string determining the axes along which learnable parameters are applied. Will be passed to ``einx.elementwise``. Defaults to ``"b... [c]"``.
        mean: Whether to apply mean normalization. Defaults to ``True``.
        var: Whether to apply variance normalization. Defaults to ``True``.
        scale: Whether to apply a learnable scale according to ``params``. Defaults to ``True``.
        bias: Whether to apply a learnable bias according to ``params``. Defaults to ``True``.
        epsilon: A small float added to the variance to avoid division by zero. Defaults to ``1e-5``.
        fastvar: Whether to use a fast variance computation. Defaults to ``True``.
        dtype: Data type of the weights. Defaults to ``"float32"``.
        decay_rate: Decay rate for exponential moving average of mean and variance. If ``None``, no moving average is applied. Defaults to ``None``.
        name: Name of the module. Defaults to ``None``.
        **kwargs: Additional parameters that specify values for single axes, e.g. ``a=4``.
    """
    
    return _Norm(stats, params=params, mean=mean, var=var, scale=scale, bias=bias, decay_rate=decay_rate, epsilon=epsilon, fastvar=fastvar, dtype=dtype, name=name, kwargs=kwargs)

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
            bias=(lambda shape: self.param("bias", nn.initializers.zeros_init(), shape, self.dtype)) if self.bias else None,
            weight=lambda shape, in_axis, out_axis, batch_axis: self.param("weight", nn.initializers.lecun_normal(in_axis, out_axis, batch_axis), shape, self.dtype),
            **(self.kwargs if not self.kwargs is None else {}),
        )

def Linear(expr, bias=True, dtype="float32", name=None, **kwargs):
    """Linear layer.

    Args:
        expr: Einstein string determining the axes along which the weight matrix is multiplied. Will be passed to ``einx.dot``.
        bias: Whether to apply a learnable bias. Defaults to ``True``.
        dtype: Data type of the weights. Defaults to ``"float32"``.
        name: Name of the module. Defaults to ``None``.
        **kwargs: Additional parameters that specify values for single axes, e.g. ``a=4``.
    """

    return _Linear(expr, bias=bias, dtype=dtype, name=name, kwargs=kwargs)

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

def Dropout(expr, drop_rate, rng_collection="dropout", name=None, **kwargs):
    """Dropout layer.

    Args:
        expr: Einstein string determining the axes along which dropout is applied. Will be passed to ``einx.elementwise``.
        drop_rate: Drop rate.
        rng_collection: the rng collection name to use when requesting an rng key. Defaults to ``"dropout"``.
        name: Name of the module. Defaults to ``None``.
        **kwargs: Additional parameters that specify values for single axes, e.g. ``a=4``.
    """

    return _Dropout(expr, drop_rate, rng_collection=rng_collection, name=name, kwargs=kwargs)