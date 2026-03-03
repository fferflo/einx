from ._util import _axis_to_axistuple


def logsumexp(classical):
    def logsumexp(x, axis=None, keepdims=False):
        if axis is None:
            axes = tuple(range(x.ndim))
        else:
            axes = _axis_to_axistuple(axis)

        x_max_keepdims = classical.max(x, axis=axis, keepdims=True)
        x_max_keepdims = classical.stop_gradient(x_max_keepdims)
        x_max_dropdims = classical.reshape(x_max_keepdims, tuple(s for i, s in enumerate(x_max_keepdims.shape) if i not in axes))

        x = classical.subtract(x, x_max_keepdims)
        x = classical.log(classical.sum(classical.exp(x), axis=axis, keepdims=keepdims))
        x = classical.add(x, x_max_keepdims if keepdims else x_max_dropdims)

        return x

    return logsumexp


def logaddexp(classical):
    def logaddexp(*xs):
        x_max = classical.maximum(*xs)
        x_max = classical.stop_gradient(x_max)

        xs = [classical.subtract(x, x_max) for x in xs]
        x = classical.log(classical.add(*[classical.exp(x) for x in xs]))
        x = classical.add(x, x_max)

        return x

    return logaddexp


def softmax(classical):
    def softmax(x, axis=None):
        x_max = classical.max(x, axis=axis, keepdims=True)
        x_max = classical.stop_gradient(x_max)
        x = classical.subtract(x, x_max)

        x_exp = classical.exp(x)

        return classical.divide(x_exp, classical.sum(x_exp, axis=axis, keepdims=True))

    return softmax


def log_softmax(classical):
    def log_softmax(x, axis=None):
        return classical.subtract(x, logsumexp(classical)(x, axis=axis, keepdims=True))

    return log_softmax


def count_nonzero(classical):
    def count_nonzero(x, axis=None):
        return classical.sum(classical.not_equal(x, 0), axis=axis)

    return count_nonzero


def flip(classical, getitem):
    def flip(x, axis=None):
        if isinstance(axis, int):
            axis = (axis,)
        if axis is None:
            axis = tuple(range(x.ndim))

        def _shift(axis):
            if axis < -x.ndim or axis >= x.ndim:
                raise ValueError(f"Invalid axis {axis} for array with {x.ndim} dimensions.")
            if axis < 0:
                axis += x.ndim
            return axis

        axis = tuple(_shift(a) for a in axis)

        x = getitem(x, tuple(slice(None) if i not in axis else slice(None, None, -1) for i in range(x.ndim)))
        return x

    return flip
