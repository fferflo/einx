Example operations
##################

einx allows expressing many commonly used operations in a concise and readable manner. This page shows some examples.

Miscellaneous
=============

Outer product, sum, etc
-----------------------

The outer product between two vectors is expressed as follows:

..  code-block:: python

    z = einx.multiply("i, j -> i j", x, y)

The same vectorization pattern of a scalar elementary operation may be used to express any other "outer" operation:

..  code-block:: python

    z = einx.add     ("i, j -> i j", x, y)          # Outer sum
    z = einx.subtract("i, j -> i j", x, y)          # Outer difference
    z = einx.maximum ("i, j -> i j", x, y)          # Outer maximum
    z = einx.id      ("i, j -> i j (1 + 1)", x, y)  # Outer stacking (*i.e.* mesh-grid)

Gather pixel colors from a batch of images
------------------------------------------

The elementary ``get_at`` operation may be used to gather pixel colors from a batch of images at specified pixel coordinates:

..  code-block:: python

    # Same coordinate for all images in the batch
    pixel_colors = einx.get_at("b [h w] c,   p [2] -> b p c", images, pixel_coords)

    # Different coordinates for each image in the batch
    pixel_colors = einx.get_at("b [h w] c, b p [2] -> b p c", images, pixel_coords)

The operation may analogously be expressed as an n-dimensional gather operation that works for inputs with arbitrary spatial dimensions, such as 
sequences, images and volumes:

..  code-block:: python

    colors = einx.get_at("b [...] c, b p [i] -> b p c", tensor, coords)

Global spatial mean
-------------------

Given an input tensor with shape ``(batch, spatial_axes..., channels)``, the global spatial mean can be expressed as follows:

..  code-block:: python

    y = einx.mean("b [...] c", x)

For example, given an input with two spatial dimensions, the above expression expands to:

..  code-block:: python

    y = einx.mean("b [h w] c -> b c", x)

Spatial mean-pooling
--------------------

Given an input tensor with shape ``(batch, spatial_axes..., channels)``, the spatial mean-pooling operation with a kernel size of ``k`` may be expressed as follows:

..  code-block:: python

    y = einx.mean("b (s [ds])... c", x, ds=k)

This divides each spatial dimension into groups of size ``k`` (if the dimensions size is evenly divisible by the group size) and computes the mean for each group.

For example, given an input with two spatial dimensions, the above expression expands to:

..  code-block:: python

    y = einx.mean("b (h [dh]) (w [dw]) c -> b h w c", x, dh=k, dw=k)

Space-to-depth and depth-to-space
---------------------------------

The space-to-depth operation rearranges an input tensor by creating patches of a given patch size (``k``) and flattening the patch and channel dimensions. Each
pixel/voxel/cell in the output tensor corresponds to a patch in the input tensor:

..  code-block:: python

    y = einx.id("b (s ds)... c -> b s... (ds... c)", x, ds=k)

The inverse operation, *i.e.* depth-to-space, creates patches by unflattening the pixels/voxels/cells of an input tensor into patches and arranging the patches along the spatial dimensions.
The operation is represented by simply swapping the input and output expressions of the space-to-depth operation:

..  code-block:: python

    z = einx.id("b s... (ds... c) -> b (s ds)... c", y, ds=k)

Broadcasted concatenation
-------------------------

Concatenation in einx is simply a type of vectorization and therefore compatible with other types of vectorization such as broadcasting,
permutation, and composition of axes. For example, this is the case when concatenating a vector to each pixel in a batch of images (*i.e.*
along the channel dimension):

..  code-block:: python

    img = np.random.rand(4, 3, 64, 64) # batch channel height width
    vec = np.random.rand(2)

    img_new = einx.id("b c1 h w, c2 -> b (c1 + c2) h w", img, vec)

Numpy requires separate shape alignment to express the same operation since ``np.concatenate`` does not support broadcasting:

..  code-block:: python

    vec_as_img = np.broadcast_to(vec[np.newaxis, :, np.newaxis, np.newaxis], (img.shape[0], vec.shape[0], img.shape[2], img.shape[3]))
    img_new = np.concatenate([img, vec_as_img], axis=1)

This is similar to other concatenation helpers such as ``einops.pack``:

..  code-block:: python

    vec_as_img = np.broadcast_to(vec[np.newaxis, :, np.newaxis, np.newaxis], (img.shape[0], vec.shape[0], img.shape[2], img.shape[3]))
    img_new, _ = einops.pack([img, vec_as_img], "b * h w")



Neural networks
===============

Fully-connected layer
---------------------

A fully-connected layer in neural net architectures consists of a matrix multiplication and addition of a bias term. This is expressed in einx as follows:

..  code-block:: python

    # Classical fully-connected layer
    x = einx.dot("... [c_in], [c_in] c_out -> ... c_out", x, weight)
    x = einx.add("... c_out, c_out -> ... c_out", x, bias)

einx notation allows expressing many variations of this layer. For example, a grouped linear layer can be expressed by representing the channel dimension
as a flattened axes of groups and channels per group:

..  code-block:: python

    # Grouped fully-connected layer: Different weights per group
    x = einx.dot("... (h [c_in]), h [c_in] c_out -> ... (h c_out)", x, weight, h=heads)
    x = einx.add("... (h c_out), h c_out -> ... (h c_out)", x, bias, h=heads)

    # Grouped fully-connected layer: Same weights per group
    x = einx.dot("... (h [c_in]), [c_in] c_out -> ... (h c_out)", x, weight, h=heads)
    x = einx.add("... (h c_out), c_out -> ... (h c_out)", x, bias, h=heads)

A fully-connected layer along spatial dimensions (such as in the `MLP-Mixer <https://arxiv.org/abs/2105.01601>`__ architecture) is expressed
by applying the dot operation along the spatial dimensions:

..  code-block:: python

    # Fully-connected layer along spatial dimensions: Forward
    x = einx.dot("b [s_in...] c, [s_in...] s_out -> b s_out c", x, weight)

    # Fully-connected layer along spatial dimensions: Backward
    x = einx.dot("b [s_out] c, [s_out] s_in... -> b s_in... c", x, weight)

Normalization layer
-------------------

Normalization layers such as `LayerNorm <https://arxiv.org/abs/1607.06450>`__ and `BatchNorm <https://arxiv.org/abs/1502.03167>`__ normalize the inputs along specific axes (and apply
a subsequent learnable scale and bias term). The normalization may be expressed by defining an elementary normalization operation and vectorizing it using einx notation:

..  code-block:: python

    def normalize(x, epsilon=1e-5):
        mean = jnp.mean(x)
        var = jnp.var(x)
        return (x - mean) / jax.lax.rsqrt(var + epsilon)

    einnormalize = einx.jax.adapt_with_vmap(normalize, signature="... -> ...")

    x = einnormalize("... [c]", x)                  # LayerNorm: https://arxiv.org/abs/1607.06450
    x = einnormalize("[...] c", x)                  # BatchNorm (but without computing running stats): https://arxiv.org/abs/1502.03167
    x = einnormalize("b [s...] c", x)               # InstanceNorm: https://arxiv.org/abs/1607.08022
    x = einnormalize("b [s...] (g [c])", x, g=8)    # GroupNorm: https://arxiv.org/abs/1803.08494

Multi-head attention
--------------------

The `multi-head attention <https://arxiv.org/abs/1706.03762>`__ operation over a set of queries ``q``, keys ``k`` and values ``v`` may be expressed in einx notation as follows:

..  code-block:: python

    a = einx.dot("b q (h [c_in]), b k (h [c_in]) -> b q k h", q, k, h=8)
    a = einx.softmax("b q [k] h", a)
    x = einx.dot("b q [k] h, b [k] (h c_out) -> b q (h c_out)", a, v)

We may alternatively define an elementary (*i.e.* single-query, single-head) attention operation, and vectorize it using einx notation:

..  code-block:: python

    def attention(q, k, v):
        a = einx.dot("[c], k [c] -> k", q, k)
        a = einx.softmax("[k]", a)
        return einx.dot("[k], [k] ->", a, v)

    einattention = einx.jax.adapt_with_vmap(attention)

    x = einattention("b q (h [c_in]), b [k] (h [c_in]), b [k] (h c_out) -> b q (h c_out)", q, k, v, h=heads)

Dropout
-------

The dropout layer randomly sets a fraction of the input units to zero during training. Different dropout layers differ among others in which groups of units are dropped together.
For example, in regular dropout, each unit is dropped independently, while in spatial dropout, all pixels of a channel are dropped together.

We may express these types of dropout layers using einx notation as follows:

..  code-block:: python

    key = jax.random.PRNGKey(42)
    drop_rate = 0.1
    dropout_factor = lambda shape: jax.random.bernoulli(key, 1.0 - drop_rate, shape) / (1.0 - drop_rate) # Divide by (1 - drop_rate) to maintain the expected value of the inputs

    x = einx.multiply("..., ...",     x, dropout_factor) # Regular dropout: Get one drop-decision for each value
    x = einx.multiply("b ... c, b c", x, dropout_factor) # Spatial dropout: Get one drop-decision for each channel (*i.e.* feature map)
    x = einx.multiply("b ..., b",     x, dropout_factor) # Drop-path/ stochastic depth: Get one drop-decision for each sample in the batch

Word embedding
--------------

A word embedding layer maps integer token indices to dense vector representations. This may be expressed in einx notation using a vectorized gather operation:

..  code-block:: python

    token_embeddings = einx.get_at("[v] c, b t -> b t c", vocabulary, token_indices)

Parameter initialization via tensor factory
-------------------------------------------

The support for :ref:`tensor factories <tutorial-tensor-factories>` in einx allows expressing parameter initialization using a simple initialization pattern.
Given a layer such as the following

..  code-block:: python

    x = einx.dot("... [c_in], [c_in] c_out -> ... c_out", x, weight, c_out=64)

we may pass a tensor factory for the ``weight`` parameter rather than initializing the weight tensor before the operation. This ensures that the weight dimensions
match the input dimensions as specified by the einx expression. Since tensor factories provide no constraints for the axis lengths, we must specify any remaining
dimensions (here ``c_out``) as additional constraints.

For example, in the `Flax Linen <https://flax-linen.readthedocs.io>`__ framework, a parameter is initialized by calling the ``self.param`` method inside a module.
We can reformulate ``self.param`` as a tensor factory with the appropriate arguments and forward it to the einx operation:

..  code-block:: python

    class MyLayer(nn.Module):
        @nn.compact
        def __call__(self, x):
            weight = lambda shape: self.param("weight", nn.initializers.normal(0.01), shape)
            x = einx.dot("... [c_in], [c_in] c_out -> ... c_out", x, weight, c_out=64)
            return x

A similar approach may be used in PyTorch with `UninitializedParameter <https://docs.pytorch.org/docs/stable/generated/torch.nn.parameter.UninitializedParameter.html>`__.