import torch, einx, math
from functools import partial
import numpy as np

class Parameter(torch.nn.parameter.UninitializedParameter):
    def __init__(self, init, dtype):
        self.init = init

    def __new__(cls, init, dtype):
        return super().__new__(cls, dtype=vars(torch)[dtype])

    def __call__(self, shape, **kwargs):
        super().materialize(shape)
        with torch.no_grad():
            self.init(self.data, **kwargs)
        return self



class Norm(torch.nn.Module):
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
        **kwargs: Additional parameters that specify values for single axes, e.g. ``a=4``.
    """

    def __init__(self, stats, params="b... [c]", mean=True, var=True, scale=True, bias=True, epsilon=1e-5, fastvar=True, dtype="float32", decay_rate=None, **kwargs):
        super().__init__()
        self.stats = stats
        self.params = params
        self.use_mean = mean
        self.use_var = var
        self.epsilon = epsilon
        self.fastvar = fastvar
        self.decay_rate = decay_rate
        self.kwargs = kwargs

        if mean and not decay_rate is None:
            self.register_buffer("mean", torch.nn.parameter.UninitializedBuffer(dtype=vars(torch)[dtype]))
            def get_mean(shape):
                if isinstance(self.mean, torch.nn.parameter.UninitializedBuffer):
                    with torch.no_grad():
                        self.mean.materialize(shape)
                        torch.nn.init.zeros_(self.mean.data)
                return self.mean
            self.get_mean = get_mean
        else:
            self.get_mean = None
        if var and not decay_rate is None:
            self.register_buffer("var", torch.nn.parameter.UninitializedBuffer(dtype=vars(torch)[dtype]))
            def get_var(shape):
                if isinstance(self.var, torch.nn.parameter.UninitializedBuffer):
                    with torch.no_grad():
                        self.var.materialize(shape)
                        torch.nn.init.ones_(self.var.data)
                return self.var
            self.get_var = get_var
        else:
            self.get_var = None
        self.scale = Parameter(torch.nn.init.ones_, dtype) if scale else None
        self.bias = Parameter(torch.nn.init.zeros_, dtype) if bias else None

    def forward(self, x):
        with x.device:
            use_ema = not self.decay_rate is None and not self.training
            x, mean, var = einx.nn.norm(
                x,
                self.stats,
                self.params,
                mean=self.get_mean if use_ema else self.use_mean,
                var=self.get_var if use_ema else self.use_var,
                scale=self.scale,
                bias=self.bias,
                epsilon=self.epsilon,
                fastvar=self.fastvar,
                backend=einx.backend.get("torch"),
                **self.kwargs,
            )
            update_ema = not self.decay_rate is None and self.training
            if update_ema:
                with torch.no_grad():
                    if not mean is None:
                        self.mean = self.decay_rate * self.get_mean(mean.shape) + (1 - self.decay_rate) * mean
                    if not var is None:
                        self.var = self.decay_rate * self.get_var(var.shape) + (1 - self.decay_rate) * var
            return x

class Linear(torch.nn.Module):
    """Linear layer.

    Args:
        expr: Einstein string determining the axes along which the weight matrix is multiplied. Will be passed to ``einx.dot``.
        bias: Whether to apply a learnable bias. Defaults to ``True``.
        dtype: Data type of the weights. Defaults to ``"float32"``.
        **kwargs: Additional parameters that specify values for single axes, e.g. ``a=4``.
    """

    def __init__(self, expr, bias=True, dtype="float32", **kwargs):
        super().__init__()

        self.fan_in = None
        def init_weight(x, in_axis, out_axis, batch_axis):
            self.fan_in = np.prod([x.shape[i] for i in in_axis])
            bound = math.sqrt(3.0) / math.sqrt(self.fan_in)
            torch.nn.init.uniform_(x, -bound, bound)
        self.weight = Parameter(init_weight, dtype)
        if bias:
            def init_bias(x):
                bound = 1 / math.sqrt(self.fan_in)
                torch.nn.init.uniform_(x, -bound, bound)
            self.bias = Parameter(init_bias, dtype)
        else:
            self.bias = None

        self.expr = expr
        self.kwargs = kwargs

    def forward(self, x):
        with x.device:
            return einx.nn.linear(
                x,
                self.expr,
                self.weight,
                self.bias,
                backend=einx.backend.get("torch"),
                **self.kwargs,
            )

class Dropout(torch.nn.Module):
    """Dropout layer.

    Args:
        expr: Einstein string determining the axes along which dropout is applied. Will be passed to ``einx.elementwise``.
        drop_rate: Drop rate.
        **kwargs: Additional parameters that specify values for single axes, e.g. ``a=4``.
    """

    def __init__(self, expr, drop_rate, **kwargs):
        super().__init__()

        self.expr = expr
        self.drop_rate = drop_rate
        self.kwargs = kwargs

    def forward(self, x):
        with x.device:
            if self.training:
                return einx.nn.dropout(
                    x,
                    self.expr,
                    drop_rate=self.drop_rate,
                    backend=einx.backend.get("torch"),
                    **self.kwargs,
                )
            else:
                return x