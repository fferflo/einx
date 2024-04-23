import flax.linen as nn
import einx
import flax
from functools import partial
import jax.numpy as jnp
from typing import Callable, Union, Optional, Any

tnn = einx.tracer.import_("flax.linen", "nn")


class ParamFactory:
    class Concrete(einx.tracer.input.Input):
        def __init__(self, module, name, init, dtype, col, param_type):
            self.module = module
            self.name = name
            self.init = init

            if dtype is None:
                if hasattr(module, "dtype"):
                    dtype = module.dtype
                else:
                    dtype = "float32"
            self.dtype = dtype

            self.col = col

            if param_type == "param":
                if col is not None:
                    raise ValueError("col is not accepted for flax.linen.Module.param")
            elif param_type == "variable":
                if col is None:
                    raise ValueError("col must be specified for flax.linen.Module.variable")
            else:
                raise ValueError(f"Unknown tensor factory flax.linen.Module.{param_type}")
            self.param_type = param_type

        def to_value_and_key(self):
            return self.module, ParamFactory.CacheKey(
                self.name, self.init, self.dtype, self.col, self.param_type
            )

    class CacheKey(einx.tracer.input.CacheKey):
        def __init__(self, name, init, dtype, col, param_type):
            self.name = name
            self.init = init
            self.dtype = dtype
            self.col = col
            self.param_type = param_type

        def __hash__(self):
            return hash((self.name, self.init, self.dtype, self.col, self.param_type))

        def __eq__(self, other):
            return (
                isinstance(other, ParamFactory.CacheKey)
                and self.name == other.name
                and self.init == other.init
                and self.dtype == other.dtype
                and self.col == other.col
                and self.param_type == other.param_type
            )

        def to_tracer(self, backend, virtual_arg):
            x = ParamFactory.Tracer(self.name, self.init, self.dtype, self.col, self.param_type)
            return x, x

    class Tracer(einx.tracer.TensorFactory):
        def __init__(self, name, init, dtype, col, param_type):
            self.name = name
            self.init = init
            self.dtype = dtype
            self.col = col
            self.param_type = param_type

        def __call__(self, shape, kwargs):
            name = self.name if not self.name is None else kwargs.get("name", None)
            init = self.init if not self.init is None else kwargs.get("init", None)
            dtype = self.dtype if not self.dtype is None else kwargs.get("dtype", None)
            col = self.col

            if name is None:
                raise ValueError(
                    "Must specify name for tensor factory flax.linen.Module.{param|variable}"
                )

            if init is None:
                raise ValueError(
                    "Must specify init for tensor factory flax.linen.Module.{param|variable}"
                )
            elif isinstance(init, str):
                if init == "get_at" or init == "rearrange":
                    init = tnn.initializers.normal(stddev=0.02)
                elif init == "add":
                    init = tnn.initializers.zeros_init()
                elif init == "multiply":
                    init = tnn.initializers.ones_init()
                elif init == "dot":
                    init = tnn.initializers.lecun_normal(
                        kwargs["in_axis"], kwargs["out_axis"], kwargs["batch_axis"]
                    )
                else:
                    raise ValueError(f"Don't know which initializer to use for operation '{init}'")
            elif isinstance(init, (int, float)):
                init = tnn.initializers.constant(init, dtype=dtype)

            if self.param_type == "param":
                x = einx.tracer.apply(
                    self.param, args=[name, init, shape, dtype], output=einx.tracer.Tensor(shape)
                )
            else:
                assert self.param_type == "variable"
                # Assume that variable initialization does not need an rng key by passing None
                x = einx.tracer.apply(
                    self.variable,
                    args=[col, name, init, None, shape, dtype],
                )
                x = einx.tracer.apply(
                    einx.tracer.MemberAccess(), args=[x, "value"], output=einx.tracer.Tensor(shape)
                )
            return x


def param(
    x: Union[Callable, nn.Module],
    name: Optional[str] = None,
    init: Optional[Any] = None,
    dtype: Optional[nn.dtypes.Dtype] = None,
    col: Optional[str] = None,
):
    """Create a tensor factory for Flax parameters.

    Args:
        x: The bound method of a Flax module, i.e. ``nn.Module.param`` or
            ``nn.Module.variable``, or a module instance in which case its ``param`` method
            is used.
        name: Name of the parameter. If ``None``, uses a default name determined from the calling
            operation. Defaults to ``None``.
        init: Initializer for the parameter. If ``None``, uses a default init method determined
            from the calling operation. Defaults to ``None``.
        dtype: Data type of the parameter. If ``None``, uses the ``dtype`` member of the calling
            module or ``float32`` if it does not exist. Defaults to ``None``.
        col: The collection name to use when ``bound_method`` is ``nn.Module.variable``.

    Returns:
        A tensor factory with the given default parameters.
    """
    if hasattr(x, "__func__") and x.__func__ == nn.Module.param:
        module = x.__self__
        param_type = "param"
    elif hasattr(x, "__func__") and x.__func__ == nn.Module.variable:
        module = x.__self__
        param_type = "variable"
    elif isinstance(x, nn.Module):
        module = x
        param_type = "param"
    else:
        raise ValueError("x must be a bound method of a Flax module or a Flax module instance")

    return ParamFactory.Concrete(module, name, init, dtype, col, param_type)


# Allow passing nn.Module, nn.Module.param, nn.Module.variable as tensor factory:
@einx.tracer.input.register_tensor_factory
def tensor_factory(x):
    if isinstance(x, nn.Module) or (
        hasattr(x, "__func__")
        and (x.__func__ == nn.Module.param or x.__func__ == nn.Module.variable)
    ):
        return param(x).to_value_and_key()
    else:
        return None


# Using _ prefix on classes and a separater constructor, since dataclass/nn.Module does
# not support **kwargs parameter.


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
    dtype: nn.dtypes.Dtype = "float32"
    kwargs: dict = None

    @nn.compact
    def __call__(self, x, training=None):
        if self.decay_rate is not None and training is None:
            raise ValueError("training must be specified when decay_rate is used")

        use_ema = self.decay_rate is not None and (not training or self.is_initializing())
        x, mean, var = einx.nn.norm(
            x,
            self.stats,
            self.params,
            mean=param(self.variable, col="stats", name="mean", dtype=self.dtype)
            if use_ema and self.mean
            else self.mean,
            var=param(self.variable, col="stats", name="var", dtype=self.dtype)
            if use_ema and self.var
            else self.var,
            scale=param(self.param, name="scale", dtype=self.dtype) if self.scale else None,
            bias=param(self.param, name="bias", dtype=self.dtype) if self.bias else None,
            epsilon=self.epsilon,
            fastvar=self.fastvar,
            **(self.kwargs if self.kwargs is not None else {}),
        )

        update_ema = self.decay_rate is not None and training and not self.is_initializing()
        if update_ema:
            if self.mean:
                mean_ema = self.variable("stats", "mean", None)
                mean_ema.value = self.decay_rate * mean_ema.value + (1 - self.decay_rate) * mean
            if self.var:
                var_ema = self.variable("stats", "var", None)
                var_ema.value = self.decay_rate * var_ema.value + (1 - self.decay_rate) * var

        return x


def Norm(
    stats: str,
    params: str = "b... [c]",
    mean: bool = True,
    var: bool = True,
    scale: bool = True,
    bias: bool = True,
    decay_rate: Optional[float] = None,
    epsilon: float = 1e-5,
    fastvar: bool = True,
    dtype: nn.dtypes.Dtype = "float32",
    name: Optional[str] = None,
    **kwargs: Any,
):
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
        name: Name of the module. Defaults to ``None``.
        **kwargs: Additional parameters that specify values for single axes, e.g. ``a=4``.
    """

    return _Norm(
        stats,
        params=params,
        mean=mean,
        var=var,
        scale=scale,
        bias=bias,
        decay_rate=decay_rate,
        epsilon=epsilon,
        fastvar=fastvar,
        dtype=dtype,
        name=name,
        kwargs=kwargs,
    )


class _Linear(nn.Module):
    expr: str
    bias: bool = True
    dtype: nn.dtypes.Dtype = "float32"
    kwargs: dict = None

    @nn.compact
    def __call__(self, x):
        return einx.nn.linear(
            x,
            self.expr,
            bias=param(self.param, name="bias", dtype=self.dtype) if self.bias else None,
            weight=param(self.param, name="weight", dtype=self.dtype),
            **(self.kwargs if self.kwargs is not None else {}),
        )


def Linear(
    expr: str,
    bias: bool = True,
    dtype: nn.dtypes.Dtype = "float32",
    name: Optional[str] = None,
    **kwargs: Any,
):
    """Linear layer.

    Args:
        expr: Einstein string determining the axes along which the weight matrix is
            multiplied. Will be passed to ``einx.dot``.
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
                **(self.kwargs if self.kwargs is not None else {}),
            )
        else:
            return x


def Dropout(
    expr: str,
    drop_rate: float,
    rng_collection: str = "dropout",
    name: Optional[str] = None,
    **kwargs: Any,
):
    """Dropout layer.

    Args:
        expr: Einstein string determining the axes along which dropout is applied. Will be passed
            to ``einx.elementwise``.
        drop_rate: Drop rate.
        rng_collection: the rng collection name to use when requesting an rng key. Defaults
            to ``"dropout"``.
        name: Name of the module. Defaults to ``None``.
        **kwargs: Additional parameters that specify values for single axes, e.g. ``a=4``.
    """

    return _Dropout(expr, drop_rate, rng_collection=rng_collection, name=name, kwargs=kwargs)
