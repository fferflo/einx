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

    x = einx.dot("b... [c1|c2]", x, w) # x * w
    x = einx.add("b... [c2]", x, b)    # x + b

The arguments ``w`` and ``b`` represent the layer weights. Instead of determining the shapes of ``w`` and ``b`` in advance to create the weights manually, we define ``w`` and ``b`` as tensor factories that
are called inside the einx functions once the shapes are determined. For example, in the Haiku framework ``hk.get_parameter`` is used to create new weights
in the current module and can be defined as a tensor factory as follows:

..  code::

    import hiaku as hk

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
handled automatically by einx, and ``torch.nn.parameter.UninitializedParameter`` can simply be passed to einx as a tensor factory:

..  code::

    import torch

    class Linear(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.parameter.UninitializedParameter()
            self.b = torch.nn.parameter.UninitializedParameter()

        def forward(self, x):
            x = einx.dot("b... [c1|c2]", x, self.w, c2=64)
            x = einx.add("b... [c2]", x, self.b)
            return x

``einx.dot`` passes the ``in_axes``, ``out_axes`` and ``batch_axes`` arguments to the respective tensor factory (if it includes corresponding parameters) that can be
used to determine the fan-in and fan-out of the layer and initialize the weights accordingly.

Layers
------

einx provides the layer types ``einn.{Linear|Norm|Dropout}`` that are implemented as outlined above.

**einn.Norm** implements a normalization layer with optional exponential moving average (EMA) over the computed statistics. The first parameter is an Einstein expression for
the axes along which the statistics for normalization are computed. The second parameter is an Einstein expression for the axes corresponding to the bias and scale terms, and
defaults to ``b... [c]``. The different sub-steps can be toggled by passing ``True`` or ``False`` for the ``mean``, ``variance``, ``scale`` and ``bias`` parameters. The EMA is used only if 
``decay_rate`` is passed. Additional parameters can be specified as constraints for the Einstein expression.

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

Example trainings on CIFAR10 are provided in ``scripts/train_{torch|flax|haiku}.py`` for models implemented using ``einn``. ``einn`` layers can be combined
with other layers or used as submodules in the respective framework seamlessly.
