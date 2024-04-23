import haiku as hk
import einx
from functools import partial
from haiku._src.base import current_module
from typing import Any, Callable, Literal, Optional

thk = einx.tracer.import_("haiku", "hk")


class ParamFactory:
    class Concrete(einx.tracer.input.Input):
        def __init__(self, name, init, dtype, param_type):
            self.name = name
            self.init = init

            if dtype is None:
                module = current_module()
                if hasattr(module, "dtype"):
                    dtype = module.dtype
                else:
                    dtype = "float32"
            self.dtype = dtype

            self.param_type = param_type

        def to_value_and_key(self):
            return None, ParamFactory.CacheKey(self.name, self.init, self.dtype, self.param_type)

    class CacheKey(einx.tracer.input.CacheKey):
        def __init__(self, name, init, dtype, param_type):
            self.name = name
            self.init = init
            self.dtype = dtype
            self.param_type = param_type

        def __hash__(self):
            return hash((self.name, self.init, self.dtype, self.param_type))

        def __eq__(self, other):
            return (
                isinstance(other, ParamFactory.CacheKey)
                and self.name == other.name
                and self.init == other.init
                and self.dtype == other.dtype
                and self.param_type == other.param_type
            )

        def to_tracer(self, backend, virtual_arg):
            return (
                None,
                ParamFactory.Tracer(self.name, self.init, self.dtype, self.param_type, virtual_arg),
            )

    class Tracer(einx.tracer.TensorFactory):
        def __init__(self, name, init, dtype, param_type, depend_on):
            self.name = name
            self.init = init
            self.dtype = dtype
            self.param_type = param_type
            self.depend_on = depend_on

        def __call__(self, shape, kwargs):
            name = self.name if not self.name is None else kwargs.get("name", None)
            init = self.init if not self.init is None else kwargs.get("init", None)
            dtype = self.dtype if not self.dtype is None else kwargs.get("dtype", None)

            if name is None:
                raise ValueError("Must specify name for tensor factory hk.get_{parameter|state}")

            if init is None:
                raise ValueError("Must specify init for tensor factory hk.get_{parameter|state}")
            elif isinstance(init, str):
                if init in "get_at" or init == "rearrange":
                    init = thk.initializers.RandomNormal(stddev=0.02)
                elif init == "add":
                    init = thk.initializers.Constant(0.0)
                elif init == "multiply":
                    init = thk.initializers.Constant(1.0)
                elif init == "dot":
                    init = thk.initializers.VarianceScaling(
                        1.0, "fan_in", "truncated_normal", fan_in_axes=kwargs["in_axis"]
                    )
                else:
                    raise ValueError(f"Don't know which initializer to use for operation '{init}'")
            elif isinstance(init, (int, float)):
                init = thk.initializers.Constant(init)

            if self.param_type == "parameter":
                func = thk.get_parameter
            elif self.param_type == "state":
                func = thk.get_state
            else:
                assert False

            return einx.tracer.apply(
                func,
                kwargs={"shape": shape, "name": name, "dtype": dtype, "init": init},
                output=einx.tracer.Tensor(shape),
                depend_on=[self.depend_on],
            )


def param(
    func: Literal[hk.get_parameter, hk.get_state] = hk.get_parameter,
    name: Optional[str] = None,
    init: Optional[Any] = None,
    dtype: Optional[Any] = None,
):
    """Create a tensor factory for Haiku parameters.

    Args:
        func: Either ``hk.get_parameter`` or ``hk.get_state``. Defaults to ``hk.get_parameter``.
        name: Name of the parameter. If ``None``, uses a default name determined from the calling
            operation. Defaults to ``None``.
        init: Initializer for the parameter. If ``None``, uses a default init method determined
            from the calling operation. Defaults to ``None``.
        dtype: Data type of the parameter. If ``None``, uses the ``dtype`` member of the calling
            module or ``float32`` if it does not exist. Defaults to ``None``.

    Returns:
        A tensor factory with the given default parameters.
    """
    if func == hk.get_parameter:
        param_type = "parameter"
    elif func == hk.get_state:
        param_type = "state"
    else:
        raise ValueError(f"Unknown parameter function '{func}'")
    return ParamFactory.Concrete(name, init, dtype, param_type)


# Allow passing hk.get_parameter and hk.get_state as tensor factory:
@einx.tracer.input.register_tensor_factory
def tensor_factory(x):
    if id(x) == id(hk.get_parameter) or id(x) == id(hk.get_state):
        return param(x).to_value_and_key()
    else:
        return None


class Norm(hk.Module):
    """Normalization layer.

    Args:
        stats: Einstein string determining the axes along which mean and variance are computed.
            Will be passed to ``einx.reduce``.
        params: Einstein string determining the axes along which learnable parameters are
            applied. Will be passed to ``einx.elementwise``. Defaults to ``"b... [c]"``.
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
        dtype: Any = "float32",
        decay_rate: Optional[float] = None,
        name: Optional[str] = None,
        **kwargs: Any,
    ):
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
        if self.decay_rate is not None and training is None:
            raise ValueError("training must be specified when decay_rate is used")

        use_ema = self.decay_rate is not None and (not training or hk.running_init())
        x, mean, var = einx.nn.norm(
            x,
            self.stats,
            self.params,
            mean=param(hk.get_state, name="mean") if use_ema and self.mean else self.mean,
            var=param(hk.get_state, name="var") if use_ema and self.var else self.var,
            scale=param(hk.get_parameter, name="scale") if self.scale else None,
            bias=param(hk.get_parameter, name="bias") if self.bias else None,
            epsilon=self.epsilon,
            fastvar=self.fastvar,
            **self.kwargs,
        )

        update_ema = self.decay_rate is not None and training and not hk.running_init()
        if update_ema:
            if self.mean:
                hk.set_state(
                    "mean", hk.get_state("mean") * self.decay_rate + mean * (1 - self.decay_rate)
                )
            if self.var:
                hk.set_state(
                    "var", hk.get_state("var") * self.decay_rate + var * (1 - self.decay_rate)
                )

        return x


class Linear(hk.Module):
    """Linear layer.

    Args:
        expr: Einstein string determining the axes along which the weight matrix is multiplied.
            Will be passed to ``einx.dot``.
        bias: Whether to apply a learnable bias. Defaults to ``True``.
        dtype: Data type of the weights. Defaults to ``"float32"``.
        name: Name of the module. Defaults to ``None``.
        **kwargs: Additional parameters that specify values for single axes, e.g. ``a=4``.
    """

    def __init__(
        self,
        expr: str,
        bias: bool = True,
        dtype: Any = "float32",
        name: Optional[str] = None,
        **kwargs: Any,
    ):
        super().__init__(name=name)
        self.expr = expr
        self.bias = bias
        self.dtype = dtype
        self.kwargs = kwargs

    def __call__(self, x):
        return einx.nn.linear(
            x,
            self.expr,
            bias=param(hk.get_parameter, name="bias") if self.bias else None,
            weight=param(hk.get_parameter, name="weight"),
            **self.kwargs,
        )


class Dropout(hk.Module):
    """Dropout layer.

    Args:
        expr: Einstein string determining the axes along which dropout is applied. Will be
            passed to ``einx.elementwise``.
        drop_rate: Drop rate.
        name: Name of the module. Defaults to ``None``.
        **kwargs: Additional parameters that specify values for single axes, e.g. ``a=4``.
    """

    def __init__(self, expr: str, drop_rate: float, name: Optional[str] = None, **kwargs: Any):
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
