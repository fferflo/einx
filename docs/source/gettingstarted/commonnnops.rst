Example: Common neural network operations
#########################################

einx allows formulating many common operations of deep learning models as concise expressions. This page provides a few examples.

..  code-block:: python

    import einx
    import einx.nn.{torch|flax|haiku|equinox|keras} as einn

LayerScale
----------

Multiply the input tensor ``x`` with a learnable parameter per channel that is initialized with a small value:

..  code-block:: python

    x = einx.multiply("... [c]", x, einn.param(init=1e-5))

Reference: `LayerScale explained <https://paperswithcode.com/method/layerscale>`_

Prepend class-token
-------------------

Flatten the spatial axes of an n-dimensional input tensor ``x`` and prepend a learnable class token:

..  code-block:: python

    x = einx.rearrange("b s... c, c -> b (1 + (s...)) c", x, einn.param(name="class_token"))

Reference: `Classification token in Vision Transformer <https://paperswithcode.com/method/vision-transformer>`_

Positional embedding
--------------------

Add a learnable positional embedding onto all tokens of the input ``x``. Works with n-dimensional inputs (text, image, video, ...):

..  code-block:: python

    x = einx.add("b [s... c]", x, einn.param(name="pos_embed", init=nn.initializers.normal(stddev=0.02)))

Reference: `Position embeddings in Vision Transformer <https://paperswithcode.com/method/vision-transformer>`_

Word embedding
--------------

Retrieve a learnable embedding vector for each token in the input sequence ``x``:

..  code-block:: python

    x = einx.get_at("[v] c, b t -> b t c", einn.param(name="vocab_embed"), x, v=50257, c=1024)

Reference: `Torch tutorial on word embeddings <https://pytorch.org/tutorials/beginner/nlp/word_embeddings_tutorial.html>`_

Layer normalization
-------------------

Compute the mean and variance along the channel axis, and normalize the tensor by subtracting the mean and dividing by the standard deviation.
Apply learnable scale and bias:

..  code-block:: python

    mean = einx.mean("... [c]", x, keepdims=True)
    var = einx.var("... [c]", x, keepdims=True)
    x = (x - mean) * torch.rsqrt(var + epsilon)

    x = einx.add("... [c]", x, einn.param(name="bias"))
    x = einx.multiply("... [c]", x, einn.param(name="scale"))

This can similarly be achieved using the ``einn.Norm`` layer:

..  code-block:: python

    import einx.nn.{torch|flax|haiku|...} as einn
    x = einn.Norm("... [c]")(x)

Reference: `Layer normalization explained <https://paperswithcode.com/method/layer-normalization>`_

Multihead attention
-------------------

Compute multihead attention for the queries ``q``, keys ``k`` and values ``v`` with ``h = 8`` heads:

..  code-block:: python

    a = einx.dot("b q (h c), b k (h c) -> b q k h", q, k, h=8)
    a = einx.softmax("b q [k] h", a)
    x = einx.dot("b q k h, b k (h c) -> b q (h c)", a, v)

Reference: `Multi-Head Attention <https://paperswithcode.com/method/multi-head-attention>`_

Shifted window attention
------------------------

Shift and partition the input tensor ``x`` into windows with sidelength ``w``, compute self-attention in each window, and unshift and merge windows again. Works with
n-dimensional inputs (text, image, video, ...):

..  code-block:: python

    # Compute axis values so we don't have to specify s and w manually later
    consts = einx.solve("b (s w)... c", x, w=16) 

    # Shift and partition windows
    x = einx.roll("b [...] c", x, shift=-shift)
    x = einx.rearrange("b (s w)... c -> (b s...) (w...) c", x, **consts)

    # Compute attention
    ...

    # Unshift and merge windows
    x = einx.rearrange("(b s...) (w...) c -> b (s w)... c", x, **consts)
    x = einx.roll("b [...] c", x, shift=shift)

Reference: `Swin Transformer <https://paperswithcode.com/method/swin-transformer>`_

Multilayer Perceptron along spatial axes (MLP-Mixer)
----------------------------------------------------

Apply a weight matrix multiplication along the spatial axes of the input tensor:

..  code-block:: python

    x = einx.dot("b [s...->s2] c", x, einn.param(name="weight1"))
    ...
    x = einx.dot("b [s2->s...] c", x, einn.param(name="weight2"), s=(256, 256))

Or with the ``einn.Linear`` layer that includes a bias term:

..  code-block:: python

    x = einn.Linear("b [s...->s2] c")(x)
    ...
    x = einn.Linear("b [s2->s...] c", s=(256, 256))(x)

Reference: `MLP-Mixer <https://paperswithcode.com/method/mlp-mixer>`_

The following page provides an example implementation of GPT-2 with ``einx`` and ``einn`` using many of these operations and validates
their correctness by loading pretrained weights and generating some example text.