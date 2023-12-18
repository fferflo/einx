Tutorial: Neural networks
#########################

einx provides several neural network layer types for deep learning frameworks (`PyTorch <https://pytorch.org/>`_, `Flax <https://github.com/google/flax>`_,
`Haiku <https://github.com/google-deepmind/dm-haiku>`_) in the ``einx.nn.*`` namespace 
based on the functions in ``einx.*``. These layers provide abstractions that can implement a wide variety of deep learning operations using Einstein notation.
The ``einx.nn.*`` namespace is entirely optional, and is imported as follows:

..  code::

    import einx.nn.{torch|flax|haiku} as einn

Motivation
----------

The main idea for implementing layers in einx is to exploit :ref:`tensor factories <lazytensorconstruction>` to initialize the weights of a layer.
For example, consider the following linear layer:

..  code::

    x = einx.dot("... [c1|c2]", x, w) # x * w
    x = einx.add("... [c2]", x, b)    # x + b

The arguments ``w`` and ``b`` represent the layer weights. Instead of determining the shapes of ``w`` and ``b`` in advance to create the weights manually,
we define ``w`` and ``b`` as tensor factories that
are called inside the einx functions once the shapes are determined. For example, in the Haiku framework ``hk.get_parameter`` is used to create new weights
in the current module and can be defined as a tensor factory as follows:

..  code::

    import haiku as hk

    class Linear(hk.Module):
        def __call__(self, x):
            w = lambda shape: hk.get_parameter(name="weight", shape=shape, dtype="float32", init=hk.initializers.VarianceScaling(1.0, "fan_in", "truncated_normal"))
            b = lambda shape: hk.get_parameter(name="bias", shape=shape, dtype="float32", init=hk.initializers.Constant(0.0))

            x = einx.dot("b... [c1|c2]", x, w, c2=64)
            x = einx.add("b... [c2]", x, b)
            return x

Unlike a tensor, the tensor factory does not provide shape constraints to the expression solver and requires that we define the missing axes (``c2``) manually. Here,
this corresponds to specifying the number of output channels of the linear layer. All other axis values are determined implicitly from the input shapes.

The weights are created once a layer is run on the first input batch. This is common practice in jax-based frameworks like Flax and Haiku where a model
is typically first invoked with a dummy batch to instantiate all weights.

In PyTorch, we rely on `lazy modules <https://pytorch.org/docs/stable/generated/torch.nn.modules.lazy.LazyModuleMixin.html#torch.nn.modules.lazy.LazyModuleMixin>`_
by creating weights as ``torch.nn.parameter.UninitializedParameter`` in the constructor and calling their ``materialize`` method on the first input batch. This is
handled automatically by einx (see below).

Parameter definition with ``einn.param``
----------------------------------------

einx provides the function ``einn.param`` to create *parameter factories* for the respective deep learning framework. ``einn.param`` is simply a convenience wrapper for
the ``lambda shape: ...`` syntax that is used in the example above:

..  code:: python

    # w1 and w2 give the same result when used as tensor factories in einx functions:

    w1 = lambda shape: hk.get_parameter(name="weight", shape=shape, dtype="float32", init=...)

    w2 = einn.param(name="weight", dtype="float32", init=...)

The utility of ``einn.param`` comes from providing several useful default arguments that simplify the definition of parameters.

*   **Default argument for** ``init``

    The type of (random) initialization that is used for a parameter in neural networks typically depends on the operation that the parameter is used in, for example:

    * A bias parameter is used in an ``add`` operation and often initialized with zeros.
    * A scale parameter is used in a ``multiply`` operation and e.g. initialized with ones in normalization layers.
    * A weight parameter in linear layers is used in a ``dot`` operation and the initialization typically depends on the fan-in or fan-out of the layer (e.g. as in
      `Lecun normal initialization <https://jax.readthedocs.io/en/latest/_autosummary/jax.nn.initializers.lecun_normal.html>`_).

    To allow ``einn.param`` to use a default initialization method based on the operation that it is used in, einx functions like :func:`einx.dot` and :func:`einx.add`
    forward their name as optional arguments to tensor factories. ``einn.param`` then defines a corresponding initializer in the respective framework and
    uses it as a default argument for ``init``. E.g. in Flax:

    ..  code:: python

        from flax import linen as nn

        if init == "get_at" or init == "rearrange":
            init = nn.initializers.normal(stddev=0.02)
        elif init == "add":
            init = nn.initializers.zeros_init()
        elif init == "multiply":
            init = nn.initializers.ones_init()
        elif init == "dot":
            init = nn.initializers.lecun_normal(kwargs["in_axis"], kwargs["out_axis"], kwargs["batch_axis"])

    :func:`einx.dot` additionally determines ``in_axis``, ``out_axis`` and ``batch_axis`` from the Einstein expression and forwards them as optional arguments
    to tensor factories. In this case, they allow ``nn.initializers.lecun_normal`` to determine the fan-in of the layer and choose the initialization accordingly.

*   **Default argument for** ``name``

    A default name is determined implicitly from the operation that the parameter is used in, for example:

    .. list-table:: 
       :widths: 30 30
       :header-rows: 0

       * - Operation
         - Name
       * - :func:`einx.add`
         - ``bias``
       * - :func:`einx.multiply`
         - ``scale``
       * - :func:`einx.dot`
         - ``weight``
       * - :func:`einx.get_at`
         - ``embedding``
       * - :func:`einx.rearrange`
         - ``embedding``

*   **Default argument for** ``dtype``

    The default data type of the parameter is determined from the ``dtype`` member variable of the respective module if it exists, and chosen as ``float32`` otherwise.

Any default argument in ``einn.param`` can be overridden by simply passing the respective argument explicitly:

..  code::

    # Initialize bias with non-zero values
    einx.add("b... [c]", x, einn.param(init=nn.initializers.normal(stddev=0.02)))

    # Initialize layerscale with small value
    einx.multiply("b... [c]", x, einn.param(init=1e-5, name="layerscale"))

If no default argument can be determined (e.g. because there is no default initialization for an operation, or the module does not have a ``dtype`` member) and the
argument is not specified explicitly in ``einn.param``, an exception is raised.

Module definition using ``einn.param``
--------------------------------------

Our definition of a linear layer above that used the ``lambda shape: ...`` syntax can be simplified using ``einn.param`` as shown below.

**Haiku**

..  code:: python

    import haiku as hk

    class Linear(hk.Module):
        dtype: str = "float32"
        def __call__(self, x):
            x = einx.dot("... [c1|c2]", x, einn.param(), c2=64)
            x = einx.add("... [c2]", x, einn.param())
            return x

In Haiku, ``hk.get_parameter`` and ``hk.get_state`` can be passed as the first parameter of ``einn.param`` to determine whether to create a parameter or state variable,
or simply be passed as a tensor factory directly:

..  code:: python

    einx.add("... [c]", x, einn.param(hk.get_parameter))  # calls einn.param(hk.get_parameter)
    einx.add("... [c]", x, einn.param())                  # calls einn.param(hk.get_parameter)
    einx.add("... [c]", x, hk.get_parameter)              # calls einn.param(hk.get_parameter)
    einx.add("... [c]", x, einn.param(hk.get_state))      # calls einn.param(hk.get_state)
    einx.add("... [c]", x, hk.get_state)                  # calls einn.param(hk.get_state)

**Flax**

..  code:: python

    from flax import linen as nn

    class Linear(nn.Module):
        dtype: str = "float32"
        def __call__(self, x):
            x = einx.dot("... [c1|c2]", x, einn.param(self), c2=64)
            x = einx.add("... [c2]", x, einn.param(self))
            return x

In Flax, parameters are created by calling the ``self.param`` (for learnable parameters) or ``self.variable`` (for state variables) method of the current module. For
convenience, einx provides several options to determine which one is used:

..  code:: python

    einx.add("... [c]", x, einn.param(self.param))                  # calls einn.param(self.param)
    einx.add("... [c]", x, einn.param(self))                        # calls einn.param(self.param)
    einx.add("... [c]", x, self.param)                              # calls einn.param(self.param)
    einx.add("... [c]", x, self)                                    # calls einn.param(self.param)
    einx.add("... [c]", x, einn.param(self.variable, col="stats"))  # calls einn.param(self.variable, col="stats")

**PyTorch**

..  code::

    import torch.nn as nn

    class Linear(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.parameter.UninitializedParameter(dtype=torch.float32)
            self.b = nn.parameter.UninitializedParameter(dtype=torch.float32)

        def forward(self, x):
            x = einx.dot("b... [c1|c2]", x, self.w, c2=64)
            x = einx.add("b... [c2]", x, self.b)
            return x

In PyTorch, parameters have to be created in the constructor of the module. Since the shape is only determined once the module is called, einx supports using
``nn.parameter.UninitializedParameter`` and ``nn.parameter.UninitializedBuffer`` as a tensor factories with ``einn.param`` (see
`lazy modules <https://pytorch.org/docs/stable/generated/torch.nn.modules.lazy.LazyModuleMixin.html#torch.nn.modules.lazy.LazyModuleMixin>`_). This also allows defining
the type of initialization which cannot trivially be done using ``lambda shape: ...`` syntax. einx also allows passing uninitialized parameters directly:

..  code:: python

    einx.add("... [c]", x, einn.param(self.w))        # calls einn.param(self.w)
    einx.add("... [c]", x, self.w)                    # calls einn.param(self.w)

For PyTorch, ``einn.param`` does not support a ``dtype`` argument since it has to be specified in the constructor.

Layers
------

einx provides the layer types ``einn.{Linear|Norm|Dropout}`` that are implemented as outlined above.

**einn.Norm** implements a normalization layer with optional exponential moving average (EMA) over the computed statistics. The first parameter is an Einstein expression for
the axes along which the statistics for normalization are computed. The second parameter is an Einstein expression for the axes corresponding to the bias and scale terms, and
defaults to ``b... [c]``. The different sub-steps can be toggled by passing ``True`` or ``False`` for the ``mean``, ``var``, ``scale`` and ``bias`` parameters. The EMA is used only if 
``decay_rate`` is passed.

A variety of normalization layers can be implemented using this abstraction:

..  code::

    layernorm       = einn.Norm("b... [c]")
    instancenorm    = einn.Norm("b [s...] c")
    groupnorm       = einn.Norm("b [s...] (g [c])", g=8)
    batchnorm       = einn.Norm("[b...] c", decay_rate=0.9)
    rmsnorm         = einn.Norm("b... [c]", mean=False, bias=False)

**einn.Linear** implements a linear layer with optional bias term. The first parameter is an operation string that is forwarded to :func:`einx.dot` to multiply the weight matrix.
A bias is added corresponding to the marked output expressions, and is disabled by passing ``bias=False``.

..  code::

    channel_mix     = einn.Linear("b... [c1|c2]", c2=64)
    spatial_mix1    = einn.Linear("b [s...|s2] c", s2=64)
    spatial_mix2    = einn.Linear("b [s2|s...] c", s=(64, 64))
    patch_embed     = einn.Linear("b (s [s2|])... [c1|c2]", s2=4, c2=64)

**einn.Dropout** implements a stochastic dropout. The first parameter specifies the shape of the mask in Einstein notation that is applied to the input tensor.

..  code::

    dropout         = einn.Dropout("[...]",       drop_rate=0.2)
    spatial_dropout = einn.Dropout("[b] ... [c]", drop_rate=0.2)
    droppath        = einn.Dropout("[b] ...",     drop_rate=0.2)

The following is an example of a simple fully-connected network for image classification using ``einn`` in Flax:

..  code::

    from flax import linen as nn
    import einx.nn.flax as einn

    class Net(nn.Module):
        @nn.compact
        def __call__(self, x, training):
            for c in [1024, 512, 256]:
                x = einn.Linear("b [...|c]", c=c)(x)
                x = einn.Norm("[b] c", decay_rate=0.99)(x, training=training)
                x = nn.gelu(x)
                x = einn.Dropout("[...]", drop_rate=0.2)(x, training=training)
            x = einn.Linear("b [...|c]", c=10)(x) # 10 classes
            return x

Example trainings on CIFAR10 are provided in ``examples/train_{torch|flax|haiku}.py`` for models implemented using ``einn``. ``einn`` layers can be combined
with other layers or used as submodules in the respective framework seamlessly.
