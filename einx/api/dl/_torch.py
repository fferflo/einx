import torch, einx, math
from functools import partial
import numpy as np

class Parameter(torch.nn.parameter.UninitializedParameter):
    def __init__(self, init):
        super().__init__()
        self.init = init

    def __new__(cls, init):
        return super().__new__(cls)

    def materialize(self, shape):
        super().materialize(shape)
        with torch.no_grad():
            self.init(self.data)

class Buffer(torch.nn.parameter.UninitializedBuffer):
    def __init__(self, init):
        super().__init__()
        self.init = init

    def __new__(cls, init):
        return super().__new__(cls)

    def materialize(self, shape):
        super().materialize(shape)
        with torch.no_grad():
            self.init(self.data)



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
                        vars(self)[name].materialize(x.shape)
                    vars(self)[name] = decay_rate * vars(self)[name] + (1 - decay_rate) * x
                    return x
                else:
                    return vars(self)[name]
            self.moving_average = moving_average

    def forward(self, x):
        return einx.dl.meanvar_norm(
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


        def init_weight(x):
            # TODO: Add proper handling of multiple input+output axes and transposed axes
            # torch.nn.init.kaiming_uniform_(x, a=math.sqrt(5))
            fan = np.prod(x.shape[:-1])
            scale = 1.0 / max(1.0, fan)
            bound = np.sqrt(3.0 * scale)
            torch.nn.init.uniform_(x, -bound, bound)
        self.weight = Parameter(init_weight)
        if bias:
            def init_bias(x):
                fan = np.prod(self.weight.shape[:-1])
                u = 1.0 / math.sqrt(fan)
                torch.nn.init.uniform_(x, -u, u)
            self.bias = Parameter(init_bias)
        else:
            self.bias = None

        self.expr = expr
        self.kwargs = kwargs

    def forward(self, x, **kwargs):
        return einx.dl.linear(x, self.expr, self.weight, self.bias, **self.kwargs, **kwargs)