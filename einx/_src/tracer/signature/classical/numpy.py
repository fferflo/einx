import einx._src.tracer as tracer
import einx._src.tracer.signature as signature


class numpy:
    def __init__(self, np=None):
        if np is None:
            np = tracer.signature.python.import_("numpy", as_="np")

        self.ndarray = np.ndarray
        self.ndarray.__getitem__ = signature.classical.getitem()

        self.asarray = signature.classical.preserve_shape(np.asarray)
        self.reshape = signature.classical.set_shape(np.reshape)
        self.transpose = signature.classical.transpose(np.transpose)
        self.broadcast_to = signature.classical.set_shape(np.broadcast_to)
        self.arange = signature.classical.arange(np.arange)
        self.concatenate = signature.classical.concatenate(np.concatenate)
        self.split = signature.classical.split(np.split, cumulative=True)
        self.diagonal = signature.classical.diagonal(np.diagonal)

        self.add = signature.classical.elementwise(np.add, num_outputs=1)
        self.subtract = signature.classical.elementwise(np.subtract, num_outputs=1)
        self.multiply = signature.classical.elementwise(np.multiply, num_outputs=1)
        self.true_divide = signature.classical.elementwise(np.true_divide, num_outputs=1)
        self.floor_divide = signature.classical.elementwise(np.floor_divide, num_outputs=1)
        self.divide = signature.classical.elementwise(np.divide, num_outputs=1)
        self.logical_and = signature.classical.elementwise(np.logical_and, num_outputs=1)
        self.logical_or = signature.classical.elementwise(np.logical_or, num_outputs=1)
        self.where = signature.classical.elementwise(np.where, num_outputs=1)
        self.maximum = signature.classical.elementwise(np.maximum, num_outputs=1)
        self.minimum = signature.classical.elementwise(np.minimum, num_outputs=1)
        self.less = signature.classical.elementwise(np.less, num_outputs=1)
        self.less_equal = signature.classical.elementwise(np.less_equal, num_outputs=1)
        self.greater = signature.classical.elementwise(np.greater, num_outputs=1)
        self.greater_equal = signature.classical.elementwise(np.greater_equal, num_outputs=1)
        self.equal = signature.classical.elementwise(np.equal, num_outputs=1)
        self.not_equal = signature.classical.elementwise(np.not_equal, num_outputs=1)
        self.logaddexp = signature.classical.elementwise(np.logaddexp, num_outputs=1)
        self.exp = signature.classical.elementwise(np.exp, num_outputs=1)
        self.log = signature.classical.elementwise(np.log, num_outputs=1)
        self.negative = signature.classical.elementwise(np.negative, num_outputs=1)
        self.divmod = signature.classical.elementwise(np.divmod, num_outputs=2)

        self.sum = signature.classical.reduce(np.sum)
        self.mean = signature.classical.reduce(np.mean)
        self.var = signature.classical.reduce(np.var)
        self.std = signature.classical.reduce(np.std)
        self.prod = signature.classical.reduce(np.prod)
        self.count_nonzero = signature.classical.reduce(np.count_nonzero)
        self.all = signature.classical.reduce(np.all)
        self.any = signature.classical.reduce(np.any)
        self.min = signature.classical.reduce(np.min)
        self.max = signature.classical.reduce(np.max)
        self.argmax = signature.classical.reduce(np.argmax)
        self.argmin = signature.classical.reduce(np.argmin)

        self.take = signature.classical.take(np.take)
        self.put = signature.classical.inplace(np.put)
        self.add.at = signature.classical.inplace(np.add.at)
        self.subtract.at = signature.classical.inplace(np.subtract.at)

        self.dot = signature.classical.dot(np.dot)
        self.matmul = signature.classical.matmul(np.matmul)
        self.einsum = signature.classical.einsum(np.einsum)

        self.roll = signature.classical.preserve_shape(np.roll)
        self.flip = signature.classical.preserve_shape(np.flip)
        self.sort = signature.classical.preserve_shape(np.sort)
        self.argsort = signature.classical.preserve_shape(np.argsort)
