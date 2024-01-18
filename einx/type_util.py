import os
import subprocess
from pathlib import Path
from typing import Any

_seen_files = set[str]()


def assign_global(
    name: str,
    func: Any,
    filename: str,
    format_file: bool = False,
):
    globals()[name] = func

    if not bool(int(os.environ.get("EINX_GENERATE_TYPES", "0"))):
        return

    # Add the type signatures to {filename}_generated_types.py
    assert filename.endswith(".py"), f'{filename=} must end with ".py"'
    in_file = Path(filename)
    out_file = in_file.parent / f"{in_file.stem}_generated_types.py"
    aggregate_file = in_file.parent / "generated_types.py"

    # If this is the first file being processed, delete the aggregate typing file
    if not _seen_files and aggregate_file.exists():
        aggregate_file.unlink()

        with aggregate_file.open("w") as f:
            _ = f.write("from typing import TYPE_CHECKING\n\n")
            _ = f.write("if TYPE_CHECKING:\n")

    # If the file hasn't been seen before, delete any existing generated types
    if (out_file_str := str(out_file.absolute())) not in _seen_files:
        _seen_files.add(out_file_str)
        if out_file.exists():
            out_file.unlink()

        # Add the file to the aggregate typing file
        with aggregate_file.open("a") as f:
            _ = f.write(f"\tfrom .{in_file.stem}_generated_types import *\n")

    # Write the type signature to the file
    with out_file.open("a") as f:
        _ = f.write(f"def {name}(*args, **kwargs): ...\n\n")

    # Run black on the file
    if format_file:
        _ = subprocess.run(["black", str(out_file.absolute())])
