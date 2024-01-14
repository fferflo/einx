import torch, einx, math
from functools import partial
import numpy as np

def param(uninitialized, init=None):
    """Create a tensor factory for an uninitialized PyTorch parameter or buffer.

    When the tensor factory is invoked, it calls the ``materialize`` method of ``uninitialized`` with the given shape and returns ``uninitialized``.

    Args:
        uninitialized: An instance of either ``torch.nn.parameter.UninitializedParameter`` or ``torch.nn.parameter.UninitializedBuffer``.
        init: Initializer for the parameter. If ``None``, uses a default init method determined from the calling operation. Defaults to ``None``.

    Returns:
        A tensor factory with the given default parameters.
    """

    def torch_param_factory(shape, init=init, **kwargs):
        if init is None:
            raise ValueError("Must specify init for tensor factory torch.nn.parameter.Uninitialized*")
        elif isinstance(init, str):
            if init == "get_at" or init == "rearrange":
                init = partial(torch.nn.init.normal_, std=0.02)
            elif init == "add":
                init = torch.nn.init.zeros_
            elif init == "multiply":
                init = torch.nn.init.ones_
            elif init == "dot":
                fan_in = np.prod([shape[i] for i in kwargs["in_axis"]])
                std = np.sqrt(1.0 / fan_in) / .87962566103423978
                init = partial(torch.nn.init.trunc_normal_, mean=0.0, std=std, a=-2.0, b=2.0)
            else:
                raise ValueError(f"Don't know which initializer to use for operation '{init}'")
        elif isinstance(init, (int, float)):
            init = partial(torch.nn.init.constant_, val=init)

        with torch.no_grad():
            uninitialized.materialize(shape)
            init(uninitialized)

        return uninitialized
    return torch_param_factory

def to_tensor_factory(x):
    if isinstance(x, (torch.nn.parameter.UninitializedParameter, torch.nn.parameter.UninitializedBuffer)) and not isinstance(x, torch._subclasses.FakeTensor):
        return param(x)
    else:
        return None

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
        else:
            self.mean = None
        if var and not decay_rate is None:
            self.register_buffer("var", torch.nn.parameter.UninitializedBuffer(dtype=vars(torch)[dtype]))
        else:
            self.var = None
        self.scale = torch.nn.parameter.UninitializedParameter(dtype=vars(torch)[dtype]) if scale else None
        self.bias = torch.nn.parameter.UninitializedParameter(dtype=vars(torch)[dtype]) if bias else None

    def forward(self, x):
        with x.device:
            use_ema = not self.decay_rate is None and (not self.training or isinstance(self.mean, torch.nn.parameter.UninitializedBuffer) or isinstance(self.var, torch.nn.parameter.UninitializedBuffer))
            x, mean, var = einx.nn.norm(
                x,
                self.stats,
                self.params,
                mean=self.mean if use_ema else self.use_mean,
                var=self.var if use_ema else self.use_var,
                scale=self.scale if not self.scale is None else None,
                bias=self.bias if not self.bias is None else None,
                epsilon=self.epsilon,
                fastvar=self.fastvar,
                backend=einx.backend.get("torch"),
                **self.kwargs,
            )
            update_ema = not self.decay_rate is None and self.training
            if update_ema:
                with torch.no_grad():
                    if not mean is None:
                        if isinstance(self.mean, torch.nn.parameter.UninitializedBuffer):
                            # self.mean has not been initialized in einx.nn.norm
                            param(self.mean, init=torch.nn.init.zeros_)(mean.shape)
                        self.mean = self.decay_rate * self.mean + (1 - self.decay_rate) * mean
                    if not var is None:
                        if isinstance(self.var, torch.nn.parameter.UninitializedBuffer):
                            # self.var has not been initialized in einx.nn.norm
                            param(self.var, init=torch.nn.init.ones_)(var.shape)
                        self.var = self.decay_rate * self.var + (1 - self.decay_rate) * var
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

        self.weight = torch.nn.parameter.UninitializedParameter(dtype=vars(torch)[dtype])
        if bias:
            self.bias = torch.nn.parameter.UninitializedParameter(dtype=vars(torch)[dtype])
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