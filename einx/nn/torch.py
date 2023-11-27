import torch, einx, math
from functools import partial
import numpy as np

class Parameter(torch.nn.parameter.UninitializedParameter):
    def __init__(self, init):
        super().__init__()
        self.init = init

    def __new__(cls, init):
        return super().__new__(cls)

    def __call__(self, shape, **kwargs):
        super().materialize(shape)
        with torch.no_grad():
            self.init(self.data, **kwargs)
        return self

class Buffer(torch.nn.parameter.UninitializedBuffer):
    def __init__(self, init):
        super().__init__()
        self.init = init

    def __new__(cls, init):
        return super().__new__(cls)

    def __call__(self, shape, **kwargs):
        super().materialize(shape)
        with torch.no_grad():
            self.init(self.data, **kwargs)
        return self



class Norm(torch.nn.Module):
    def __init__(self, stats, params="b... [c]", mean=True, var=True, scale=True, bias=True, decay_rate=None, **kwargs):
        super().__init__()
        self.stats = stats
        self.params = params
        self.kwargs = kwargs

        self.mean = Buffer(torch.nn.init.zeros_) if mean else None
        self.var = Buffer(torch.nn.init.ones_) if var else None
        self.scale = Parameter(torch.nn.init.ones_) if scale else None
        self.bias = Parameter(torch.nn.init.zeros_) if bias else None

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
            **self.kwargs,
        )

class Linear(torch.nn.Module):
    def __init__(self, expr, bias=True, **kwargs):
        super().__init__()

        self.fan_in = None
        def init_weight(x, in_axis, out_axis, batch_axis):
            self.fan_in = np.prod([x.shape[i] for i in in_axis])
            bound = math.sqrt(3.0) / math.sqrt(self.fan_in)
            torch.nn.init.uniform_(x, -bound, bound)
        self.weight = Parameter(init_weight)
        if bias:
            def init_bias(x):
                bound = 1 / math.sqrt(self.fan_in)
                torch.nn.init.uniform_(x, -bound, bound)
            self.bias = Parameter(init_bias)
        else:
            self.bias = None

        self.expr = expr
        self.kwargs = kwargs

    def forward(self, x, **kwargs):
        return einx.nn.linear(x, self.expr, self.weight, self.bias, **self.kwargs, **kwargs)

class Dropout(torch.nn.Module):
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