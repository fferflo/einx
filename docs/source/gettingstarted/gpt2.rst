Example: GPT-2
##############

    We succeeded in taking that picture, and, if you look at it, you see a dot. That's here. That's home. That's us. On it, *we wrote, "We are the people."*
    
    -- Carl Sagan & GPT-2

In this example, we will reimplement the GPT-2 architecture using einx and the deep learning framework `Haiku <https://github.com/google-deepmind/dm-haiku>`_, load
pretrained weights from Hugging Face and validate the model by generating some text.

..  code-block:: python

    import haiku as hk
    import jax, einx
    from functools import partial
    import einx.nn.haiku as einn
    import numpy as np

    # Define some layer types we will use.
    # 1. Use channels-last layout
    # 2. Use layer normalization, and an epsilon of 1e-5 as in the original implementation
    Linear = partial(einn.Linear, "... [_|channels]")
    Norm = partial(einn.Norm, "... [c]", epsilon=1e-5)

The main building block of GPT-2 consists of multi-head self-attention and a multi-layer perceptron (MLP). Each sub-block uses a residual connection and
layer normalization at the beginning of the residual block:

..  code-block:: python

    class Block(hk.Module):
        heads: int = 25
        mlp_ratio: int = 4

        def __call__(self, x):
            # ########### Attention block ###########
            x0 = x
            x = Norm()(x)

            # Predict queries, keys and values
            x = Linear(channels=3 * x.shape[-1])(x)
            q, k, v = jnp.split(x, 3, axis=-1)

            # Compute attention matrix over h heads
            q = q * ((q.shape[-1] // self.heads) ** -0.5)
            attn = einx.dot("b q (h c), b k (h c) -> b q k h", q, k, h=self.heads)

            # Apply causal mask
            mask = jnp.tril(jnp.ones((q.shape[1], q.shape[1]), dtype=bool))
            attn = einx.where("q k, b q k h, ", mask, attn, -jnp.inf)

            # Apply softmax and compute weighted average over the input tokens
            attn = einx.softmax("b q [k] h", attn)
            x = einx.dot("b q k h, b k (h c) -> b q (h c)", attn, v)

            # Output projection
            x = Linear(channels=x.shape[-1])(x)

            x = x + x0

            # ########### MLP block ###########
            x0 = x
            x = Norm()(x)

            x = Linear(channels=x.shape[-1] * self.mlp_ratio)(x)
            x = jax.nn.gelu(x)
            x = Linear(channels=x0.shape[-1])(x)

            x = x + x0

            return x

The multi-head attention requires no additional statements to split the channel axis into multiple heads or merge the heads back into a single axis.
We instead just specify the channels axis as an :ref:`axis composition <axiscomposition>` of ``h`` heads and ``c`` channels per head:

..  code-block:: python

    attn = einx.dot("b q (h c), b k (h c) -> b q k h", q, k, h=self.heads)
    ...
    x = einx.dot("b q k h, b k (h c) -> b q (h c)", attn, v)

We can verify the correctness of these operations by inspecting the jit-compiled function:

>>> graph = einx.dot("b q (h c), b k (h c) -> b q k h", q, k, h=self.heads, graph=True)
>>> print(graph)
def dot(i0, i1, backend):
    x1 = backend.to_tensor(i0)
    x2 = backend.reshape(x1, (1, 1024, 25, 64))
    x4 = backend.to_tensor(i1)
    x5 = backend.reshape(x4, (1, 1024, 25, 64))
    x6 = backend.einsum("abcd,aecd->abec", x2, x5)
    return x6

The final GPT-2 model first embeds the input tokens and adds positional embeddings. It then applies a number of main blocks and maps the output onto next token
logits using a linear layer:

..  code-block:: python

    class GPT2(hk.Module):
        channels: int = 1600
        depth: int = 48
        vocab_size: int = 50257
        block_size: int = 1024

        def __call__(self, x):
            # Word embedding: Retrieve embedding for each token from the word_embed table
            x = einx.get_at("[v] c, b t -> b t c", einn.param(name="word_embed"), x, v=self.vocab_size, c=self.channels)

            # Positional embedding
            x = einx.add("b [t c]", x, einn.param(name="pos_embed", init=hk.initializers.RandomNormal(stddev=0.02)))

            # Blocks
            for i in range(self.depth):
                x = Block(name=f"block{i}")(x)
            x = Norm()(x)

            # Classifier
            x = Linear(channels=self.vocab_size, bias=False)(x)

            return x

We use tensor factories with ``einn.param`` to construct the word and positional embeddings (see 
:doc:`Tutorial: Neural networks </gettingstarted/neuralnetworks>`).

With this, we're done with the model definition. Next, we'll define some input data that the model will be applied to and encode it to token representation:

..  code-block:: python

    text = ("We succeeded in taking that picture, and, if you look at it, you see a dot."
            "That's here. That's home. That's us. On it,")
    print(f"Input: {text}")

    # Encode text to tokens
    import tiktoken
    encoder = tiktoken.get_encoding("gpt2")
    tokens = np.asarray(encoder.encode_ordinary(text))
    n = len(tokens)

    # Pad tokens to input block size
    tokens = np.pad(tokens, (0, GPT2.block_size - n), constant_values=0)

The model is initialized using a dummy batch (see `Haiku Basics <https://dm-haiku.readthedocs.io/en/latest/notebooks/basics.html>`_):

..  code-block:: python

    import time
    rng = jax.random.PRNGKey(int(time.time() * 1000))
    model = hk.transform(lambda x: GPT2()(x))
    params = model.init(rng, tokens[np.newaxis]) # Add batch axis to tokens using np.newaxis

At this point, ``params`` contains only randomly initialized weights. We download the original model weights for the XL variant of GPT-2 from
`Hugging Face <https://huggingface.co/gpt2-xl>`_ and load them into our model using the
`weightbridge 🌉 <https://github.com/fferflo/weightbridge>`_ library:

..  code-block:: python

    # Download original weights
    import transformers # only used to download weights
    pretrained_params = {k: np.asarray(v) for k, v in transformers.GPT2LMHeadModel.from_pretrained(f"gpt2-xl").state_dict().items()}
    pretrained_params["lm_head.weight"] = np.transpose(pretrained_params["lm_head.weight"], (1, 0))
    pretrained_params = {k: v for k, v in pretrained_params.items() if not k.endswith(".attn.bias") and not k.endswith(".attn.masked_bias")}

    # Map weights to our model implementation
    import weightbridge
    params = weightbridge.adapt(pretrained_params, params, hints=[("norm_1", "ln_2")])

Finally, we can run several forward passes to predict next tokens:

..  code-block:: python

    apply = jax.jit(model.apply) # Just-in-time compile the forward pass
    temperature = 0.3
    for _ in range(10): # Predict 10 next tokens
        logits = apply(params, rng, tokens[np.newaxis])[0]
        logits = logits[n - 1] # Get logits for next token
        tokens[n] = jax.random.categorical(rng, logits / temperature) # Sample next token
        n += 1
    print(f"Prediction: {encoder.decode(tokens[:n])}")

Input:

    We succeeded in taking that picture, and, if you look at it, you see a dot. That's here. That's home. That's us. On it,
    
Prediction:

    We succeeded in taking that picture, and, if you look at it, you see a dot. That's here. That's home. That's us. On it, we wrote, "We are the people."

The `full example script can be found here <https://github.com/fferflo/weightbridge/blob/master/examples/gpt2haiku.py>`_, and a similar example script for the
`Mamba language model can be found here <https://github.com/fferflo/weightbridge/blob/master/examples/mamba2haiku.py>`_.