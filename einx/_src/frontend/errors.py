import traceback


def export(obj):
    obj.__module__ = "einx.errors"
    return obj


@export
class EinxError(Exception):
    pass


@export
class SyntaxError(EinxError):
    """
    This error occurs if einx operations encounter invalid syntax in an expression.

    Example:
        >>> x = np.random.rand(3, 4)
        >>> einx.id("a (b -> b a", x)
        einx.errors.SyntaxError: Found an opening parenthesis that is not closed:
        Expression: "a (b"
                    ^
    """

    def __init__(self, expression, message, pos=None):
        if not isinstance(expression, str):
            raise ValueError("expression must be a string")

        from einx._src.namedtensor import ExpressionIndicator

        indicator = ExpressionIndicator(expression)

        if isinstance(pos, int):
            self.pos = [pos]
        elif pos is None:
            self.pos = []
        else:
            self.pos = list(pos)
        assert all(p >= 0 and p < len(expression) for p in self.pos)

        if "%EXPR%" in message:
            message = message.replace("%EXPR%", indicator.create(self.pos))

        EinxError.__init__(self, message)


@export
class RankError(EinxError):
    """
    This error occurs if the ranks (i.e. number of dimensions) of all expressions cannot
    be determined under the given input tensor shapes and additional constraints.

    Example:
        >>> x = np.random.rand(3, 4)
        >>> einx.id("a b c -> a c b", x)
        einx.errors.RankError: The number of tensor dimensions and axes in the expression does not match.
        The operation was called with the following arguments:
        - Positional argument #1: Tensor with shape (3, 4)
    """

    def __init__(self, invocation, message, pos=None, constraints=None):
        if constraints is None:
            constraints = []

        if isinstance(pos, int):
            self.pos = [pos]
        elif pos is None:
            self.pos = []
        else:
            self.pos = list(pos)
        assert all(p >= 0 and p < len(invocation.expression) for p in self.pos)

        constraints2 = []
        for constraint in constraints:
            if isinstance(constraint, str):
                constraints2.append(constraint)
            else:
                assert len(constraint) == 2
                if constraint[0] is not None and constraint[1] is not None:
                    constraints2.append(f"{constraint[0]} = {constraint[1]}")

        if "%EXPR%" in message:
            message = message.replace("%EXPR%", invocation.indicator.create(self.pos))
        if len(constraints2) > 0:
            message += "\nThe following equation(s) were determined for the expression:"
            for constraint in constraints2:
                message += f"\n    {constraint}"
        message += "\n" + invocation.to_call_signature_string()

        EinxError.__init__(self, message)


@export
class AxisSizeError(EinxError):
    """
    This error occurs if the size of all axes in an einx expression cannot be determined
    under the given input tensor shapes and additional constraints.

    Example:
        >>> x = np.random.rand(3, 4)
        >>> y = np.random.rand(4, 4)
        >>> einx.add("a b, a c -> a b c", x, y)
        einx.errors.AxisSizeError: Failed to determine the size of all axes in the expression under the given constraints.
        Expression: "a b, a c -> a b c"

        The operation was called with the following arguments:
        - Positional argument #1: Tensor with shape (3, 4)
        - Positional argument #2: Tensor with shape (4, 4)
    """

    def __init__(self, invocation, message, pos=None, constraints=None):
        if constraints is None:
            constraints = []

        if isinstance(pos, int):
            self.pos = [pos]
        elif pos is None:
            self.pos = []
        else:
            self.pos = list(pos)
        assert all(p >= 0 and p < len(invocation.expression) for p in self.pos)

        constraints2 = []
        for constraint in constraints:
            if isinstance(constraint, str):
                constraints2.append(constraint)
            else:
                assert len(constraint) == 2
                if constraint[0] is not None and constraint[1] is not None:
                    constraints2.append(f"{constraint[0]} = {constraint[1]}")

        if "%EXPR%" in message:
            message = message.replace("%EXPR%", invocation.indicator.create(self.pos))
        if len(constraints2) > 0:
            message += "\nThe expression, tensor shapes and contraints resulted in the following equation(s):"
            for constraint in constraints2:
                message += f"\n    {constraint}"
        message += "\n" + invocation.to_call_signature_string()

        EinxError.__init__(self, message)


@export
class SemanticError(EinxError):
    """
    This error occurs if the expression requirements of a particular einx operation are not met.

    Example:
        >>> x = np.random.rand(3, 4)
        >>> y = np.random.rand(4, 5)
        >>> einx.dot("a [b], [c] d -> a d", x, y)
        einx.errors.SemanticError: All contracted axes must appear in exactly two input expressions.
        Expression: "a [b], [c] d -> a d"
                        ^    ^
        The operation was called with the following arguments:
        - Positional argument #1: Tensor with shape (3, 4)
        - Positional argument #2: Tensor with shape (4, 5)
    """

    def __init__(self, message, invocation=None, pos=None):
        assert isinstance(message, str)
        if invocation is not None:
            if isinstance(pos, int):
                self.pos = [pos]
            elif pos is None:
                self.pos = []
            else:
                self.pos = list(pos)
            assert all(p >= 0 and p < len(invocation.expression) for p in self.pos)

            if "%EXPR%" in message:
                message = message.replace("%EXPR%", invocation.indicator.create(self.pos))
            message += "\n" + invocation.to_call_signature_string()

        else:
            self.pos = None

        EinxError.__init__(self, message)


@export
class OperationNotSupportedError(EinxError):
    """
    This error occurs if an einx operation is not supported by the selected backend.

    Example:
        >>> x = np.random.rand(3, 4)
        >>> einx.min("a [b]", x, backend="numpy.einsum")
        einx.errors.OperationNotSupportedError: min operation is not supported by the numpy.einsum backend.
    """

    def __init__(self, message=None):
        if message is None:
            message = "This operation is not supported by this backend."
        EinxError.__init__(self, message)


@export
class ImportBackendError(EinxError):
    """
    This error occurs if a requested backend failed to be imported or initialized.

    Example:
        >>> sys.modules["jax"] = types.SimpleNamespace()  # Create some invalid jax module
        >>> x = np.random.rand(3, 4)
        >>> einx.min("a [b]", x, backend="jax")
        einx.errors.ImportBackendError: Failed to import backend "jax" for module "jax" due to the following error: ...
    """

    pass


@export
class BackendResolutionError(EinxError):
    """
    This error occurs if the backend with which to execute an einx operation could not be determined.

    Example:
        >>> x = jnp.zeros((10, 11))
        >>> y = torch.zeros(10, 11)
        >>> einx.add("a b, a b", x, y)
        einx.errors.BackendResolutionError: Failed to determine which backend to use for this operation: ...
    """

    pass


@export
class CallOperationError(EinxError):
    """
    This error occurs if the Python function that einx compiles for an einx operation fails to execute.

    Example:
        >>> op = lambda x: None  # Create some invalid operation
        >>> einop = einx.jax.adapt_with_vmap(op)
        >>> x = np.random.rand(3, 4)
        >>> einop("a [b] -> a", x)
        einx.errors.CallOperationError: The function that was compiled for this operation failed to execute.
        ...
        The error was: AssertionError: Expected 1st return value of the adapted function to be a tensor
    """

    @staticmethod
    def create(e, code):
        message = "The function that was compiled for this operation failed to execute."
        if code is not None:
            message += " The following code was generated:\n"
            for i, line in enumerate(code.splitlines(), 1):
                message += f"{i:4}: {line}\n"

            if e.__traceback__ is not None:
                tb = traceback.extract_tb(e.__traceback__)
                if tb:
                    linenos = [frame.lineno for frame in tb if frame.filename == "<string>"]
                    if len(linenos) == 1:
                        message += f"The error occurred at line {linenos[0]}.\n"

        message += "\nThe error was: "
        if hasattr(type(e), "__name__"):
            message += f"{type(e).__name__}"
        if len(str(e)) > 0:
            message += f": {e}"
        return CallOperationError(message)
