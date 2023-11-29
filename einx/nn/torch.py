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

class Buffer(torch.nn.parameter.UninitializedBuffer):
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
        dtype: Data type of the weights. Defaults to ``"float32"``.
        decay_rate: Decay rate for exponential moving average of mean and variance. If ``None``, no moving average is applied. Defaults to ``None``.
        **kwargs: Additional parameters that specify values for single axes, e.g. ``a=4``.
    """

    def __init__(self, stats, params="b... [c]", mean=True, var=True, scale=True, bias=True, epsilon=1e-5, dtype="float32", decay_rate=None, **kwargs):
        super().__init__()
        self.stats = stats
        self.params = params
        self.kwargs = kwargs
        self.epsilon = epsilon

        self.mean = Buffer(torch.nn.init.zeros_, dtype) if mean else None
        self.var = Buffer(torch.nn.init.ones_, dtype) if var else None
        self.scale = Parameter(torch.nn.init.ones_, dtype) if scale else None
        self.bias = Parameter(torch.nn.init.zeros_, dtype) if bias else None

        if decay_rate is None:
            self.moving_average = None
        else:
            def moving_average(f, name):
                if decay_rate is None:
                    return f()
                elif self.training:
                    x = f()
                    if (isinstance(vars(self)[name], torch.nn.parameter.UninitializedParameter) or isinstance(vars(self)[name], torch.nn.parameter.UninitializedBuffer)) \
                        and not isinstance(vars(self)[name].data, torch._subclasses.FakeTensor):
                        vars(self)[name](x.shape)
                    vars(self)[name] = decay_rate * vars(self)[name] + (1 - decay_rate) * x
                    return x
                else:
                    return vars(self)[name]
            self.moving_average = moving_average

    def forward(self, x):
        return einx.nn.meanvar_norm(
            x,
            self.stats,
            self.params,
            moving_average=self.moving_average,
            mean=not self.mean is None,
            var=not self.var is None,
            scale=self.scale,
            bias=self.bias,
            epsilon=self.epsilon,
            **self.kwargs,
        )

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

    def forward(self, x, **kwargs):
        return einx.nn.linear(x, self.expr, self.weight, self.bias, **self.kwargs, **kwargs)

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

    def forward(self, x, **kwargs):
        if self.training:
            return einx.nn.dropout(x, self.expr, drop_rate=self.drop_rate, **self.kwargs, **kwargs)
        else:
            return x