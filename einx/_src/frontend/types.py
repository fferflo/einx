from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # Annotation for type checkers
    Tensor = Any
else:
    # Annotation that can be used at runtime to determine which arguments of a function are tensors
    class Tensor:
        pass
