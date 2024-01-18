import os
import subprocess
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any

_seen_files = set[str]()


def assign_global(
    name: str,
    func: Any,
    signature: str,
    filename: str,
    format_file: bool = False,
):
    globals()[name] = func

    if not bool(int(os.environ.get("EINX_GENERATE_TYPES", "0"))):
        return

    global _seen_files

    # Add the type signatures to {filename}_generated_types.py
    assert filename.endswith(".py"), f'{filename=} must end with ".py"'
    in_file = Path(filename)
    types_dir = in_file.parent / "_generated_types"
    types_dir.mkdir(exist_ok=True)
    out_file = types_dir / in_file.name
    aggregate_file = types_dir / "__init__.py"

    # If this is the first file being processed, delete the aggregate typing file
    if not _seen_files:
        aggregate_file.unlink(missing_ok=True)

    # If the file hasn't been seen before, delete any existing generated types
    if (out_file_str := str(out_file.absolute())) not in _seen_files:
        out_file.unlink(missing_ok=True)

        # Import TArray
        with out_file.open("w") as f:
            _ = f.write("from typing import TYPE_CHECKING\n\n")
            _ = f.write("if TYPE_CHECKING:\n")
            _ = f.write("\timport typing as t\n\n")
            _ = f.write("\tfrom einx.type_util import Backend, TArray\n\n")

        # Add the file to the aggregate typing file
        with aggregate_file.open("a") as f:
            _ = f.write(f"from .{in_file.stem} import *\n")

        _seen_files.add(out_file_str)

    # Write the type signature to the file
    with out_file.open("a") as f:
        # Write the function signature
        _ = f.write(f"\tdef {name}{signature}:\n")

        # Write the function docstring
        if docstring := func.__doc__:
            docstring = docstring.replace("\n", "\n\t\t")
            _ = f.write(f'\t\t"""\n\t\t{docstring}\n\t\t"""\n')

        # Write the function body
        _ = f.write("\t\t...\n\n")

    # Run black on the file
    if format_file:
        try:
            _ = subprocess.run(["black", str(out_file.absolute())])
        except FileNotFoundError as e:
            warnings.warn(
                f"Failed to run black on {out_file.absolute()}: {e}. "
                "Please run `pip install black` to format the generated types."
            )


if TYPE_CHECKING:
    from typing import Literal, Never

    from typing_extensions import TypeVar

    try:
        from jax import Array as JaxArray
    except ImportError:
        JaxArray = Never
    try:
        from numpy.typing import ArrayLike as NumpyArray
    except ImportError:
        NumpyArray = Never
    try:
        from torch import Tensor as TorchTensor
    except ImportError:
        TorchTensor = Never
    try:
        from tensorflow import Tensor as TensorFlowTensor
    except ImportError:
        TensorFlowTensor = Never

    TArray = TypeVar(
        "TArray",
        TorchTensor,
        JaxArray,
        NumpyArray,
        TensorFlowTensor,
        infer_variance=True,
    )

    Backend = Literal["torch", "jax", "numpy", "tensorflow"]
