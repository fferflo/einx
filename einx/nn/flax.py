import flax.linen as nn
import einx, flax
from functools import partial
import jax.numpy as jnp

def param(bound_method, shape=None, name=None, init=None, dtype=None, col=None, **kwargs):
    if isinstance(bound_method, nn.Module):
        bound_method = bound_method.param

    if shape is None:
        kwargs = dict(kwargs)
        if not name is None:
            kwargs["name"] = name
        if not init is None:
            kwargs["init"] = init
        if not dtype is None:
            kwargs["dtype"] = dtype
        if not col is None:
            kwargs["col"] = col
        return partial(param, bound_method, **kwargs)

    if name is None:
        raise ValueError("Must specify name for tensor factory flax.linen.Module.{param|variable}")

    if init is None:
        raise ValueError("Must specify init for tensor factory flax.linen.Module.{param|variable}")
    elif isinstance(init, str):
        if init == "get_at" or init == "rearrange":
            init = nn.initializers.normal(stddev=0.02)
        elif init == "add":
            init = nn.initializers.zeros_init()
        elif init == "multiply":
            init = nn.initializers.ones_init()
        elif init == "dot":
            init = nn.initializers.lecun_normal(kwargs["in_axis"], kwargs["out_axis"], kwargs["batch_axis"])
        else:
            raise ValueError(f"Don't know which initializer to use for operation '{init}'")
    elif isinstance(init, (int, float)):
        init = nn.initializers.constant(init, dtype=dtype)

    if bound_method.__func__ == nn.Module.param:
        if not col is None:
            raise ValueError("col is not accepted for flax.linen.Module.param")
        return bound_method(name, init, shape, dtype)
    elif bound_method.__func__ == nn.Module.variable:
        if col is None:
            raise ValueError("col must be specified for flax.linen.Module.variable")
        # Assume that variable initialization does not need an rng key by passing None:
        return bound_method(col, name, init, None, shape, dtype).value
    else:
        raise ValueError(f"Unknown tensor factory flax.linen.Module.{bound_method.__func__.__name__}")

def to_tensor_factory(x):
    if isinstance(x, nn.Module) or (hasattr(x, "__func__") and x.__func__ == nn.Module.param):
        return param(x)
    else:
        return None



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
            mean=param(self.variable, col="stats", name="mean", dtype=self.dtype) if use_ema else self.mean,
            var=param(self.variable, col="stats", name="var", dtype=self.dtype) if use_ema else self.var,
            scale=param(self.param, name="scale", dtype=self.dtype) if self.scale else None,
            bias=param(self.param, name="bias", dtype=self.dtype) if self.bias else None,
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
            bias=param(self.param, name="bias", dtype=self.dtype) if self.bias else None,
            weight=param(self.param, name="weight", dtype=self.dtype),
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