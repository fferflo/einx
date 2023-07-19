import torch, einx, math
from functools import partial

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
    def __init__(self, expr, bias=True):
        super().__init__()

        def init_weight(x):
            torch.nn.init.kaiming_uniform_(x, a=math.sqrt(5))
        self.weight = Parameter(init_weight)
        if bias:
            def init_bias(x):
                u = 1.0 / math.sqrt(self.weight.shape[0])
                torch.nn.init.uniform_(x, -u, u)
            self.bias = Parameter(init_bias)
        else:
            self.bias = None

        self.expr = expr

    def forward(self, x):
        return einx.dl.linear(x, self.expr, self.weight, self.bias)