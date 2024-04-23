import keras
import einx
import inspect
import numpy as np
from typing import Any, Callable, Optional

_version = tuple(int(i) for i in keras.__version__.split(".")[:2])
if _version < (3, 0):
    raise ImportError(f"einx.nn.keras requires Keras version >= 3, but found {keras.__version__}")


tkeras = einx.tracer.import_("keras")


def create_or_retrieve(layer, name, shape, dtype, init, trainable):
    if name in vars(layer):
        tensor = vars(layer)[name]
    else:
        tensor = vars(layer)[name] = layer.add_weight(
            shape=shape,
            dtype=dtype,
            initializer=init,
            name=name,
            trainable=trainable,
        )
    return tensor


class ParamFactory:
    class Concrete(einx.tracer.input.Input):
        def __init__(self, layer, name, init, dtype, trainable):
            self.layer = layer
            self.name = name
            self.init = init

            if dtype is None:
                if hasattr(layer, "dtype"):
                    dtype = layer.dtype
                else:
                    dtype = "float32"
            self.dtype = dtype

            self.trainable = trainable

        def to_value_and_key(self):
            return self.layer, ParamFactory.CacheKey(
                self.name, self.init, self.dtype, self.trainable
            )

    class CacheKey(einx.tracer.input.CacheKey):
        def __init__(self, name, init, dtype, trainable):
            self.name = name
            self.init = init
            self.dtype = dtype
            self.trainable = trainable

        def __hash__(self):
            return hash((self.name, self.init, self.dtype, self.trainable))

        def __eq__(self, other):
            return (
                isinstance(other, ParamFactory.CacheKey)
                and self.name == other.name
                and self.init == other.init
                and self.dtype == other.dtype
                and self.trainable == other.trainable
            )

        def to_tracer(self, backend, virtual_arg):
            x = ParamFactory.Tracer(self.name, self.init, self.dtype, self.trainable)
            return x, x

    class Tracer(einx.tracer.TensorFactory):
        def __init__(self, name, init, dtype, trainable):
            self.name = name
            self.init = init
            self.dtype = dtype
            self.trainable = trainable

        def __call__(self, shape, kwargs):
            name = self.name if not self.name is None else kwargs.get("name", None)
            init = self.init if not self.init is None else kwargs.get("init", None)
            dtype = self.dtype if not self.dtype is None else kwargs.get("dtype", None)

            if name is None:
                raise ValueError("Must specify name for tensor factory keras.layers.Layer")

            if init is None:
                raise ValueError("Must specify init for tensor factory keras.layers.Layer")
            elif isinstance(init, str):
                if init == "get_at" or init == "rearrange":
                    init = tkeras.initializers.TruncatedNormal(stddev=0.02)
                elif init == "add":
                    init = tkeras.initializers.Constant(0.0)
                elif init == "multiply":
                    init = tkeras.initializers.Constant(1.0)
                elif init == "dot":
                    fan_in = np.prod([shape[i] for i in kwargs["in_axis"]])
                    std = np.sqrt(1.0 / fan_in) / 0.87962566103423978
                    init = tkeras.initializers.TruncatedNormal(mean=0.0, stddev=std)
                else:
                    raise ValueError(f"Don't know which initializer to use for operation '{init}'")
            elif isinstance(init, (int, float)):
                init = tkeras.initializers.Constant(init)

            return einx.tracer.apply(
                create_or_retrieve,  # TODO: make tracable
                args=[self, name, shape, dtype, init, self.trainable],
                output=einx.tracer.Tensor(shape),
            )


def param(
    layer: keras.layers.Layer,
    name: Optional[str] = None,
    init: Optional[Any] = None,
    dtype: Optional[Any] = None,
    trainable: bool = True,
):
    """Create a tensor factory for Keras parameters.

    Args:
        layer: The layer to create the parameter in. Must be an instance of ``keras.layers.Layer``.
        name: Name of the parameter. If ``None``, uses a default name determined from the
            calling operation. Defaults to ``None``.
        init: Initializer for the parameter. If ``None``, uses a default init method determined
            from the calling operation. Defaults to ``None``.
        dtype: Data type of the parameter. If ``None``, uses the ``dtype`` member of the calling
            module or ``float32`` if it does not exist. Defaults to ``None``.
        trainable: Whether the parameter is trainable. Defaults to ``True``.

    Returns:
        A tensor factory with the given default parameters.
    """
    return ParamFactory.Concrete(layer, name, init, dtype, trainable)


def is_leaf(x):
    return isinstance(x, tuple) and all(isinstance(y, int) for y in x)


class Layer(keras.layers.Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def build(self, inputs_shape):
        tracers = einx.tree_util.tree_map(
            lambda shape: keras.ops.zeros(shape, dtype="float32"), inputs_shape, is_leaf=is_leaf
        )

        if "is_initializing" in inspect.signature(self.call).parameters:
            self.call(tracers, is_initializing=True)
        else:
            self.call(tracers)


class Norm(Layer):
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
        dtype: Any = "float32",
        decay_rate: Optional[float] = None,
        **kwargs: Any,
    ):
        super().__init__(dtype=dtype)
        self.stats = stats
        self.params = params
        self.use_mean = mean
        self.use_var = var
        self.use_scale = scale
        self.use_bias = bias
        self.epsilon = epsilon
        self.fastvar = fastvar
        self.decay_rate = decay_rate
        self.kwargs = kwargs

    def call(self, x, training=None, is_initializing=False):
        use_ema = self.decay_rate is not None and (not training or is_initializing)
        x, mean, var = einx.nn.norm(
            x,
            self.stats,
            self.params,
            mean=param(self, name="mean", trainable=False)
            if use_ema and self.use_mean
            else self.use_mean,
            var=param(self, name="var", trainable=False)
            if use_ema and self.use_var
            else self.use_var,
            scale=param(self, name="scale", trainable=True) if self.use_scale else None,
            bias=param(self, name="bias", trainable=True) if self.use_bias else None,
            epsilon=self.epsilon,
            fastvar=self.fastvar,
            **(self.kwargs if self.kwargs is not None else {}),
        )

        update_ema = self.decay_rate is not None and training and not is_initializing
        if update_ema:
            if self.use_mean:
                self.mean.assign(
                    keras.ops.cast(
                        self.decay_rate * self.mean.value + (1 - self.decay_rate) * mean,
                        self.mean.dtype,
                    )
                )
            if self.use_var:
                self.var.assign(
                    keras.ops.cast(
                        self.decay_rate * self.var.value + (1 - self.decay_rate) * var,
                        self.var.dtype,
                    )
                )

        return x


class Linear(Layer):
    """Linear layer.

    Args:
        expr: Einstein string determining the axes along which the weight matrix is multiplied.
            Will be passed to ``einx.dot``.
        bias: Whether to apply a learnable bias. Defaults to ``True``.
        dtype: Data type of the weights. Defaults to ``"float32"``.
        **kwargs: Additional parameters that specify values for single axes, e.g. ``a=4``.
    """

    def __init__(self, expr: str, bias: bool = True, dtype: Any = "float32", **kwargs: Any):
        super().__init__(dtype=dtype)

        self.expr = expr
        self.use_bias = bias
        self.kwargs = kwargs

    def call(self, x):
        return einx.nn.linear(
            x,
            self.expr,
            bias=param(self, name="bias") if self.use_bias else None,
            weight=param(self, name="weight"),
            **self.kwargs,
        )


class Dropout(Layer):
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

    def call(self, x, training=None):
        if training:
            return einx.nn.dropout(
                x,
                self.expr,
                drop_rate=self.drop_rate,
                **self.kwargs,
            )
        else:
            return x
