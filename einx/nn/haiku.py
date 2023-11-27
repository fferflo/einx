import haiku as hk
import einx
from functools import partial

class Norm(hk.Module):
    def __init__(self, stats, params="b... [c]", mean=True, var=True, scale=True, bias=True, decay_rate=None, name=None, **kwargs):
        super().__init__(name=name)
        self.stats = stats
        self.params = params
        self.kwargs = kwargs

        if decay_rate is None:
            self.mean = mean
            self.var = var
        else:
            self.mean = hk.ExponentialMovingAverage(decay_rate, name="mean") if not mean is None else None
            self.var = hk.ExponentialMovingAverage(decay_rate, name="var") if not var is None else None
        self.scale = scale
        self.bias = bias
        self.decay_rate = decay_rate

    def moving_average(self, f, name, training):
        if self.decay_rate is None:
            return f()
        else:
            if training is None:
                raise ValueError("training must be specified when decay_rate is not None")
            if training:
                x = f()
                vars(self)[name](x)
                return x
            else:
                return vars(self)[name].average

    def __call__(self, x, training=None):
        return einx.nn.meanvar_norm(
            x,
            self.stats,
            self.params,
            moving_average=partial(self.moving_average, training=training),
            mean=self.mean,
            var=self.var,
            scale=lambda shape: hk.get_parameter(name="scale", shape=shape, dtype="float32", init=hk.initializers.Constant(1.0)) if self.scale else None,
            bias=lambda shape: hk.get_parameter(name="bias", shape=shape, dtype="float32", init=hk.initializers.Constant(0.0)) if self.bias else None,
            **self.kwargs,
        )

class Linear(hk.Module):
    def __init__(self, expr, bias=True, name=None, **kwargs):
        super().__init__(name=name)
        self.expr = expr
        self.bias = bias
        self.kwargs = kwargs

    def __call__(self, x):
        return einx.nn.linear(
            x,
            self.expr,
            bias=lambda shape: hk.get_parameter(name="bias", shape=shape, dtype="float32", init=hk.initializers.Constant(0.0)) if self.bias else None,
            weight=lambda shape: hk.get_parameter(name="weight", shape=shape, dtype="float32", init=hk.initializers.VarianceScaling(1.0, "fan_in", "truncated_normal")),
            **self.kwargs,
        )

class Dropout(hk.Module):
    def __init__(self, expr, drop_rate, name=None):
        super().__init__(name=name)
        self.expr = expr
        self.drop_rate = drop_rate

    def __call__(self, x, training):
        if training:
            return einx.nn.dropout(
                x,
                self.expr,
                drop_rate=self.drop_rate,
                rng=hk.next_rng_key(),
            )
        else:
            return x