Neural networks
###############

einx provides several neural network layer types for deep learning frameworks (`Torch <https://pytorch.org/>`_, `Flax <https://github.com/google/flax>`_,
`Haiku <https://github.com/google-deepmind/dm-haiku>`_) in the ``einx.nn.*`` namespace 
based on the functions in ``einx.*``. These layers provide abstractions that can implement a wide variety of deep learning operations using Einstein notation.
The ``einx.nn.*`` namespace is entirely optional, and is imported as follows:

..  code::

    import einx.nn.{torch|flax|haiku} as einn

Motivation
----------

The main idea for implementing layers in einx is to exploit :ref:`tensor factories <lazytensorconstruction>` to initialize the weights of a layer.
As an example, in the following linear layer the parameters ``w`` and ``b`` represent the layer weights:

..  code::

    x = einx.dot("b... [c1|c2]", x, w) # x * w
    x = einx.add("b... [c]", x, b)     # x + b

Instead of determining the shapes of ``w`` and ``b`` in advance to create the weights manually, we define ``w`` and ``b`` as tensor factories that
are called inside the einx functions once the shapes are determined. For example, in the Haiku framework ``hk.get_parameter`` is used to create new weights
and can be defined as a tensor factory as follows:

..  code::

    import hiaku as hk

    class Linear(hk.Module):
        def __call__(self, x):
            w = lambda shape: hk.get_parameter(name="weight", shape=shape, dtype="float32", init=hk.initializers.VarianceScaling(1.0, "fan_in", "truncated_normal"))
            b = lambda shape: hk.get_parameter(name="bias", shape=shape, dtype="float32", init=hk.initializers.Constant(0.0))

            x = einx.dot("b... [c1|c2]", x, w, c2=64)
            x = einx.add("b... [c]", x, b)

Unlike a tensor, the tensor factory does not provide shape constraints to the expression solver and requires that we define the missing axes (``c2``) manually.

The weights are created once a layer is run on the first input batch. This is standard practice in jax-based frameworks like Flax and Haiku where a model
is typically first invoked with a dummy batch to instantiate all weights before the training loop.

In Torch, we have to rely on `lazy modules <https://pytorch.org/docs/stable/generated/torch.nn.modules.lazy.LazyModuleMixin.html#torch.nn.modules.lazy.LazyModuleMixin>`_
by creating weights as ``torch.nn.parameter.UninitializedParameter`` in the constructor and calling their ``materialize`` method on the first input batch. This is
handled automatically by einx, and ``torch.nn.parameter.UninitializedParameter`` can simply be passed to einx like a regular tensor factory:

..  code::

    import torch

    class Linear(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.parameter.UninitializedParameter()
            self.b = torch.nn.parameter.UninitializedParameter()

        def forward(self, x):
            x = einx.dot("b... [c1|c2]", x, self.w, c2=64)
            x = einx.add("b... [c]", x, self.b)
            return x

``einx.dot`` passes the ``in_axes``, ``out_axes`` and ``batch_axes`` to the respective tensor factory (if it accepts these parameters) that can be used to determine the
fan-in and fan-out of the layer and initialize the weights accordingly.

Layers
------

einx provides the layer types ``einn.{Linear|Norm|Dropout}`` that are implemented as outlined above. In all cases, the constructor accepts Einstein expressions that
describe the desired operation and optionally ``**parameters`` providing additional constraints to the expression solver.

The abstractions can be used to implement a wide variety of different layers:

..  code::

    layernorm       = einn.Norm("b... [c]")
    instancenorm    = einn.Norm("b [s...] c")
    groupnorm       = einn.Norm("b [s...] (g [c])", g=8)
    batchnorm       = einn.Norm("[b...] c", decay_rate=0.9)
    rmsnorm         = einn.Norm("b... [c]", mean=False, bias=False)

    channel_mix     = einn.Linear("b... [c1|c2]", c2=64)
    spatial_mix1    = einn.Linear("b [s...|s2] c", s2=64)
    spatial_mix2    = einn.Linear("b [s2|s...] c", s=(64, 64))
    patch_embed     = einn.Linear("b (s [s2|])... [c1|c2]", s2=4, c2=64)

    dropout         = einn.Dropout("[...]",       drop_rate=0.2)
    spatial_dropout = einn.Dropout("[b] ... [c]", drop_rate=0.2)
    droppath        = einn.Dropout("[b] ...",     drop_rate=0.2)

The scripts ``scripts/train_{torch|flax|haiku}.py`` provide example trainings for models implemented using ``einn`` on CIFAR10. ``einn`` layers can be combined
with other layers or used as submodules in the respective framework seemlessly.