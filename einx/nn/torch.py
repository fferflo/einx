import torch
import einx
import math
from functools import partial
import numpy as np
from typing import Callable, Union, Optional, Any

_version = tuple(int(i) for i in torch.__version__.split(".")[:2])
if _version < (2, 0):
    raise ImportError(f"einx.nn.torch requires PyTorch version >= 2, but found {torch.__version__}")


def _allow_in_graph(func):
    if "compiler" in dir(torch):
        return torch.compiler.allow_in_graph(func)
    else:
        import torch._dynamo as _dynamo

        return _dynamo.allow_in_graph(func)


ttorch = einx.tracer.import_("torch")


class ParamFactory:
    class Concrete(einx.tracer.input.Input):
        def __init__(self, param, init):
            self.param = param
            self.init = init

        def to_value_and_key(self):
            return self.param, ParamFactory.CacheKey(self.init)

    class CacheKey(einx.tracer.input.CacheKey):
        def __init__(self, init):
            self.init = init

        def __hash__(self):
            return hash(self.init)

        def __eq__(self, other):
            return isinstance(other, ParamFactory.CacheKey) and self.init == other.init

        def to_tracer(self, backend, virtual_arg):
            x = ParamFactory.Tracer(self.init)
            return x, x

    class Tracer(einx.tracer.TensorFactory):
        def __init__(self, init):
            self.init = init

        def __call__(self, shape, kwargs):
            init = self.init if not self.init is None else kwargs.get("init", None)

            x = self

            output = einx.tracer.Tensor(shape)
            x = einx.tracer.apply(
                x.materialize,
                args=[shape],
                output=output,
                inplace_updates=[(x, output)],
            )

            if init is None:
                raise ValueError(
                    "Must specify init for tensor factory torch.nn.parameter.Uninitialized*"
                )
            elif isinstance(init, str):
                if init == "get_at" or init == "rearrange":
                    init = partial(ttorch.nn.init.normal_, std=0.02)
                elif init == "add":
                    init = ttorch.nn.init.zeros_
                elif init == "multiply":
                    init = ttorch.nn.init.ones_
                elif init == "dot":
                    fan_in = np.prod([shape[i] for i in kwargs["in_axis"]])
                    std = np.sqrt(1.0 / fan_in) / 0.87962566103423978
                    init = partial(ttorch.nn.init.trunc_normal_, mean=0.0, std=std, a=-2.0, b=2.0)
                else:
                    raise ValueError(f"Don't know which initializer to use for operation '{init}'")
            elif isinstance(init, (int, float)):
                init = partial(ttorch.nn.init.constant_, val=init)

            output = einx.tracer.Tensor(shape)
            x = einx.tracer.apply(
                init,
                args=[x],
                output=output,
                inplace_updates=[(x, output)],
            )

            return x


def param(
    x: Union[
        torch.nn.parameter.UninitializedParameter,
        torch.nn.parameter.UninitializedBuffer,
        torch.nn.parameter.Parameter,
    ],
    init: Optional[Any] = None,
):
    """Create a tensor factory for an uninitialized PyTorch parameter or buffer.

    If the given parameter is not initialized, this returns a tensor factory that calls
    the ``materialize`` method of ``uninitialized`` with the given shape and returns
    ``uninitialized``. Otherwise, the parameter is returned as is.

    Args:
        x: An instance of ``torch.nn.parameter.UninitializedParameter``,
            ``torch.nn.parameter.UninitializedBuffer`` or ``torch.nn.parameter.Parameter``.
        init: Initializer for the parameter. If ``None``, uses a default init method determined
            from the calling operation. Defaults to ``None``.

    Returns:
        A tensor factory with the given default parameters, or the parameter itself if it is
        already materialized.
    """
    if isinstance(
        x, (torch.nn.parameter.UninitializedParameter, torch.nn.parameter.UninitializedBuffer)
    ) and not isinstance(x, torch._subclasses.FakeTensor):
        # Return
        return ParamFactory.Concrete(x, init)
    else:
        # If parameter is already materialized, return it as is
        return x


# Allow passing UninitializedParameter and UninitializedBuffer as tensor factory:
@einx.tracer.input.register_tensor_factory
def tensor_factory(x):
    if isinstance(
        x, (torch.nn.parameter.UninitializedParameter, torch.nn.parameter.UninitializedBuffer)
    ) and not isinstance(x, torch._subclasses.FakeTensor):
        return param(x).to_value_and_key()
    else:
        return None


class Norm(torch.nn.Module):
    """Normalization layer.

    Args:
        stats: Einstein string determining the axes along which mean and variance are computed.
            Will be passed to ``einx.reduce``.
        params: Einstein string determining the axes along which learnable parameters are applied.
            Will be passed to ``einx.elementwise``. Defaults to ``"b... [c]"``.
        mean: Whether to apply mean normalization. Defaults to ``True``.
        var: Whether to apply variance normalization. Defaults to ``True``.
        scale: Whether to apply a learnable scale according to ``params``. Defaults to ``True``.
        bias: Whether to apply a learnable bias according to ``params``. Defaults to ``True``.
        epsilon: A small float added to the variance to avoid division by zero. Defaults
            to ``1e-5``.
        fastvar: Whether to use a fast variance computation. Defaults to ``True``.
        dtype: Data type of the weights. Defaults to ``"float32"``.
        decay_rate: Decay rate for exponential moving average of mean and variance. If ``None``,
            no moving average is applied. Defaults to ``None``.
        **kwargs: Additional parameters that specify values for single axes, e.g. ``a=4``.
    """

    def __init__(
        self,
        stats: str,
        params: str = "b... [c]",
        mean: bool = True,
        var: bool = True,
        scale: bool = True,
        bias: bool = True,
        epsilon: float = 1e-5,
        fastvar: bool = True,
        dtype: Union[torch.dtype, str] = "float32",
        decay_rate: Optional[float] = None,
        **kwargs: Any,
    ):
        super().__init__()
        self.stats = stats
        self.params = params
        self.use_mean = mean
        self.use_var = var
        self.epsilon = epsilon
        self.fastvar = fastvar
        self.decay_rate = decay_rate
        self.kwargs = kwargs

        if mean and decay_rate is not None:
            self.register_buffer(
                "mean", torch.nn.parameter.UninitializedBuffer(dtype=vars(torch)[dtype])
            )
        else:
            self.mean = None
        if var and decay_rate is not None:
            self.register_buffer(
                "var", torch.nn.parameter.UninitializedBuffer(dtype=vars(torch)[dtype])
            )
        else:
            self.var = None
        self.scale = (
            torch.nn.parameter.UninitializedParameter(dtype=vars(torch)[dtype]) if scale else None
        )
        self.bias = (
            torch.nn.parameter.UninitializedParameter(dtype=vars(torch)[dtype]) if bias else None
        )

        self.initialized = False

    def forward(self, x):
        use_ema = self.decay_rate is not None and (not self.training or not self.initialized)
        x, mean, var = _norm(
            x,
            self.stats,
            self.params,
            mean=self.mean if use_ema and self.use_mean else self.use_mean,
            var=self.var if use_ema and self.use_var else self.use_var,
            scale=self.scale if self.scale is not None else None,
            bias=self.bias if self.bias is not None else None,
            epsilon=self.epsilon,
            fastvar=self.fastvar,
            kwargs=self.kwargs,
        )
        update_ema = self.decay_rate is not None and self.training
        if update_ema:
            with torch.no_grad():
                if mean is not None:
                    self.mean.copy_(self.decay_rate * self.mean + (1 - self.decay_rate) * mean)
                if var is not None:
                    self.var.copy_(self.decay_rate * self.var + (1 - self.decay_rate) * var)
        if not self.initialized:
            self.initialized = True
        return x


@_allow_in_graph
def _norm(x, stats, params, mean, var, scale, bias, epsilon, fastvar, kwargs):
    with x.device:
        return einx.nn.norm(
            x,
            stats,
            params,
            mean=mean,
            var=var,
            scale=scale,
            bias=bias,
            epsilon=epsilon,
            fastvar=fastvar,
            backend="torch",
            **kwargs,
        )


class Linear(torch.nn.Module):
    """Linear layer.

    Args:
        expr: Einstein string determining the axes along which the weight matrix is multiplied.
            Will be passed to ``einx.dot``.
        bias: Whether to apply a learnable bias. Defaults to ``True``.
        dtype: Data type of the weights. Defaults to ``"float32"``.
        **kwargs: Additional parameters that specify values for single axes, e.g. ``a=4``.
    """

    def __init__(
        self,
        expr: str,
        bias: bool = True,
        dtype: Union[torch.dtype, str] = "float32",
        **kwargs: Any,
    ):
        super().__init__()

        self.weight = torch.nn.parameter.UninitializedParameter(dtype=vars(torch)[dtype])
        if bias:
            self.bias = torch.nn.parameter.UninitializedParameter(dtype=vars(torch)[dtype])
        else:
            self.bias = None

        self.expr = expr
        self.kwargs = kwargs

    def forward(self, x):
        return _linear(x, self.expr, self.weight, self.bias, self.kwargs)


@_allow_in_graph
def _linear(x, expr, weight, bias, kwargs):
    with x.device:
        return einx.nn.linear(
            x,
            expr,
            weight,
            bias,
            backend="torch",
            **kwargs,
        )


class Dropout(torch.nn.Module):
    """Dropout layer.

    Args:
        expr: Einstein string determining the axes along which dropout is applied. Will be
            passed to ``einx.elementwise``.
        drop_rate: Drop rate.
        **kwargs: Additional parameters that specify values for single axes, e.g. ``a=4``.
    """

    def __init__(self, expr: str, drop_rate: float, **kwargs: Any):
        super().__init__()

        self.expr = expr
        self.drop_rate = drop_rate
        self.kwargs = kwargs

    def forward(self, x):
        if self.training:
            return _dropout(x, self.expr, self.drop_rate, self.kwargs)
        else:
            return x


@_allow_in_graph
def _dropout(x, expr, drop_rate, kwargs):
    with x.device:
        return einx.nn.dropout(
            x,
            expr,
            drop_rate=drop_rate,
            backend="torch",
            **kwargs,
        )
