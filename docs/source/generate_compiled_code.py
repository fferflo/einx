import einx
import numpy as np
import os
import markdown
import jax.numpy as jnp
import torch
import array_api_compat

report = ""


def generate_op(func_str):
    func_str = func_str.strip().split("\n")

    rst = ""

    func_str_rst = "".join(["\n    " + line.strip() for line in func_str if line.strip()])
    rst += "The operation\n\n"
    rst += f"..  code-block:: Python\n{func_str_rst}\n\n"
    rst += "compiles to the following code:\n\n"
    return rst


def generate_code(func_str, backend=None):
    if backend is not None:
        backend = f', backend="{backend}"'
    else:
        backend = ""
    func_str = func_str.strip().split("\n")

    rst = ""

    func_str[-1] = "code = " + func_str[-1]
    func_str = "\n".join(func_str)
    func_str = func_str.strip()
    assert func_str.endswith(")")
    func_str = func_str[:-1] + f", graph=True{backend})"

    locals_globals = {"jnp": jnp, "einx": einx, "np": np}
    exec(func_str, locals_globals, locals_globals)
    code = eval("code", locals_globals, locals_globals)

    code = code.strip().split("\n")
    code = "".join(["\n    " + line for line in code if line.strip()])

    rst += "\n\n"
    rst += f"..  code-block:: Python\n{code}"
    rst += "\n\n"
    return rst


def generate(func_str):
    rst = ""

    rst += generate_op(func_str)
    rst += generate_code(func_str)

    return rst


def underline(text, char):
    return f"\n{text}\n{char * len(text)}\n\n"


report += underline("Examples of compiled code", "#")

report += """
The following are examples of various einx operations along with the Python code snippet that einx compiles for them, using either
the default backend or explicitly specifying a backend. The compiled code can be inspected by passing ``graph=True`` to the einx operation.
"""


report += underline("Axis permutation", "=")

code = """
x = np.zeros((10, 5, 2))
einx.id("a b c -> b c a", x)
""".strip()

report += generate_op(code)

report += underline('With ``backend="numpy"``', "-")
report += generate_code(code, backend="numpy")

report += underline('With ``backend="torch"``', "-")
report += generate_code(code, backend="torch")

report += underline('With ``backend="jax"``', "-")
report += generate_code(code, backend="jax")

report += underline('With ``backend="arrayapi"``', "-")
report += generate_code(code, backend="arrayapi")


report += underline("Axis flattening", "=")
report += generate("""
x = np.zeros((10, 5))
einx.id("(a b) c -> a (b c)", x, b=2)
""")


report += underline("No-op", "=")
report += generate("""
x = np.zeros((10, 5))
einx.id("a b -> a b", x)
""")


report += underline("Element-wise multiplication", "=")

code = """
x = jnp.zeros((2, (5 * 6)))
y = jnp.zeros((4, 3, 6))
einx.multiply("a (d e), c b e -> a b c d e", x, y)
"""

report += generate_op(code)

report += underline('With ``backend="jax.numpylike"``', "-")
report += generate_code(code, backend="jax.numpylike")

report += underline('With ``backend="jax.vmap"``', "-")
report += generate_code(code, backend="jax.vmap")

report += underline('With ``backend="jax.einsum"``', "-")
report += generate_code(code, backend="jax.einsum")


report += underline("Dot-product", "=")

code = """
x = jnp.zeros((2, 3))
y = jnp.zeros((4, 3))
einx.dot("a [b], c [b] -> c a", x, y)
"""

report += generate_op(code)

report += underline('With ``backend="jax.numpylike"``', "-")
report += generate_code(code, backend="jax.numpylike")

report += underline('With ``backend="jax.vmap"``', "-")
report += generate_code(code, backend="jax.vmap")

report += underline('With ``backend="jax.einsum"``', "-")
report += generate_code(code, backend="jax.einsum")


report += underline("Indexing", "=")

code = """
x = jnp.zeros((2, 128, 128, 3))
y = jnp.zeros((50, 2))
einx.get_at("b [h w] c, p [2] -> b p c", x, y)
"""

report += generate_op(code)

report += underline('With ``backend="jax.numpylike"``', "-")
report += generate_code(code, backend="jax.numpylike")

report += underline('With ``backend="jax.vmap"``', "-")
report += generate_code(code, backend="jax.vmap")


with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "more", "compiledcode.rst"), "w") as f:
    f.write(report)
