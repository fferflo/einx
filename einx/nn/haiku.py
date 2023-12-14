import haiku as hk
import einx
from functools import partial

class Norm(hk.Module):
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

    def __init__(self, stats, params="b... [c]", mean=True, var=True, scale=True, bias=True, epsilon=1e-5, fastvar=True, dtype="float32", decay_rate=None, name=None, **kwargs):
        super().__init__(name=name)
        self.stats = stats
        self.params = params
        self.mean = mean
        self.var = var
        self.scale = scale
        self.bias = bias
        self.epsilon = epsilon
        self.fastvar = fastvar
        self.dtype = dtype
        self.decay_rate = decay_rate
        self.kwargs = kwargs

    def __call__(self, x, training=None):
        if not self.decay_rate is None and training is None:
            raise ValueError("training must be specified when decay_rate is used")

        get_ema_mean = lambda shape: hk.get_state(name="mean", shape=shape, dtype=self.dtype, init=hk.initializers.Constant(0.0))
        get_ema_var = lambda shape: hk.get_state(name="var", shape=shape, dtype=self.dtype, init=hk.initializers.Constant(1.0))

        use_ema = not self.decay_rate is None and not training
        x, mean, var = einx.nn.norm(
            x,
            self.stats,
            self.params,
            mean=get_ema_mean if use_ema else self.mean,
            var=get_ema_var if use_ema else self.var,
            scale=(lambda shape: hk.get_parameter(name="scale", shape=shape, dtype=self.dtype, init=hk.initializers.Constant(1.0))) if self.scale else None,
            bias=(lambda shape: hk.get_parameter(name="bias", shape=shape, dtype=self.dtype, init=hk.initializers.Constant(0.0))) if self.bias else None,
            epsilon=self.epsilon,
            fastvar=self.fastvar,
            **self.kwargs,
        )

        update_ema = not self.decay_rate is None and training
        if update_ema:
            if self.mean:
                hk.set_state("mean", get_ema_mean(mean.shape) * self.decay_rate + mean * (1 - self.decay_rate))
            if self.var:
                hk.set_state("var", get_ema_var(var.shape) * self.decay_rate + var * (1 - self.decay_rate))

        return x

class Linear(hk.Module):
    """Linear layer.

    Args:
        expr: Einstein string determining the axes along which the weight matrix is multiplied. Will be passed to ``einx.dot``.
        bias: Whether to apply a learnable bias. Defaults to ``True``.
        dtype: Data type of the weights. Defaults to ``"float32"``.
        name: Name of the module. Defaults to ``None``.
        **kwargs: Additional parameters that specify values for single axes, e.g. ``a=4``.
    """

    def __init__(self, expr, bias=True, dtype="float32", name=None, **kwargs):
        super().__init__(name=name)
        self.expr = expr
        self.bias = bias
        self.dtype = dtype
        self.kwargs = kwargs

    def __call__(self, x):
        return einx.nn.linear(
            x,
            self.expr,
            bias=(lambda shape: hk.get_parameter(name="bias", shape=shape, dtype=self.dtype, init=hk.initializers.Constant(0.0))) if self.bias else None,
            weight=lambda shape: hk.get_parameter(name="weight", shape=shape, dtype=self.dtype, init=hk.initializers.VarianceScaling(1.0, "fan_in", "truncated_normal")),
            **self.kwargs,
        )

class Dropout(hk.Module):
    """Dropout layer.

    Args:
        expr: Einstein string determining the axes along which dropout is applied. Will be passed to ``einx.elementwise``.
        drop_rate: Drop rate.
        name: Name of the module. Defaults to ``None``.
        **kwargs: Additional parameters that specify values for single axes, e.g. ``a=4``.
    """

    def __init__(self, expr, drop_rate, name=None, **kwargs):
        super().__init__(name=name)
        self.expr = expr
        self.drop_rate = drop_rate
        self.kwargs = kwargs

    def __call__(self, x, training):
        if training:
            return einx.nn.dropout(
                x,
                self.expr,
                drop_rate=self.drop_rate,
                rng=hk.next_rng_key(),
                **self.kwargs,
            )
        else:
            return x