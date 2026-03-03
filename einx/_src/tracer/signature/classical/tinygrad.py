import einx._src.tracer as tracer
import einx._src.tracer.signature as signature


class tinygrad:
    def __init__(self, tinygrad=None):
        if tinygrad is None:
            tinygrad = tracer.signature.python.import_("tinygrad")

        if isinstance(tinygrad, tracer.Tracer):
            self.Tensor = tracer.cast(tinygrad.Tensor, lambda origin: Tensor(tinygrad.Tensor, origin=origin))
        else:
            self.Tensor = Tensor(tinygrad.Tensor)


class Tensor(tracer.Tracer):
    @property
    def _tracer_type(self):
        return lambda origin: Tensor(self._Tensor, origin=origin)

    def __init__(self, Tensor, origin=None):
        super().__init__(origin=origin)
        self._Tensor = Tensor

        self.__getitem__ = signature.classical.getitem()

        if isinstance(Tensor, tracer.Tracer):
            getattr = tracer.signature.python.getattr
        else:
            getattr = globals()["getattr"]

        self.reshape = signature.classical.set_shape(Tensor.reshape)
        self.permute = signature.classical.transpose(Tensor.permute)
        self.expand = signature.classical.set_shape(Tensor.expand)
        self.arange = signature.classical.arange(Tensor.arange)
        self.cat = signature.classical.concatenate(lambda tensors, **kwargs: Tensor.cat(*tensors, **kwargs), argname_axis="dim")
        self.split = signature.classical.split(Tensor.split, cumulative=False, argname_axis="dim")

        self.detach = signature.classical.preserve_shape(Tensor.detach)

        self.add = signature.classical.elementwise(Tensor.add, num_outputs=1)
        self.sub = signature.classical.elementwise(Tensor.sub, num_outputs=1)
        self.mul = signature.classical.elementwise(Tensor.mul, num_outputs=1)
        self.div = signature.classical.elementwise(Tensor.div, num_outputs=1)
        self.idiv = signature.classical.elementwise(Tensor.idiv, num_outputs=1)
        self.where = signature.classical.elementwise(Tensor.where, num_outputs=1)
        self.maximum = signature.classical.elementwise(Tensor.maximum, num_outputs=1)
        self.minimum = signature.classical.elementwise(Tensor.minimum, num_outputs=1)
        self.__lt__ = signature.classical.elementwise(getattr(Tensor, "__lt__"), num_outputs=1)
        self.__le__ = signature.classical.elementwise(getattr(Tensor, "__le__"), num_outputs=1)
        self.__gt__ = signature.classical.elementwise(getattr(Tensor, "__gt__"), num_outputs=1)
        self.__ge__ = signature.classical.elementwise(getattr(Tensor, "__ge__"), num_outputs=1)
        self.__eq__ = signature.classical.elementwise(getattr(Tensor, "__eq__"), num_outputs=1)
        self.__ne__ = signature.classical.elementwise(getattr(Tensor, "__ne__"), num_outputs=1)
        self.logaddexp = signature.classical.elementwise(Tensor.logaddexp, num_outputs=1)
        self.exp = signature.classical.elementwise(Tensor.exp, num_outputs=1)
        self.log = signature.classical.elementwise(Tensor.log, num_outputs=1)
        self.neg = signature.classical.elementwise(Tensor.neg, num_outputs=1)

        self.sum = signature.classical.reduce(Tensor.sum, argname_keepdims="keepdim")
        self.mean = signature.classical.reduce(Tensor.mean, argname_keepdims="keepdim")
        self.var = signature.classical.reduce(Tensor.var, argname_keepdims="keepdim")
        self.std = signature.classical.reduce(Tensor.std, argname_keepdims="keepdim")
        self.prod = signature.classical.reduce(Tensor.prod, argname_keepdims="keepdim")
        self.all = signature.classical.reduce(Tensor.all, argname_keepdims="keepdim")
        self.any = signature.classical.reduce(Tensor.any, argname_keepdims="keepdim")
        self.min = signature.classical.reduce(Tensor.min, argname_keepdims="keepdim")
        self.max = signature.classical.reduce(Tensor.max, argname_keepdims="keepdim")
        self.argmax = signature.classical.reduce(Tensor.argmax, argname_keepdims="keepdim")
        self.argmin = signature.classical.reduce(Tensor.argmin, argname_keepdims="keepdim")
        self.logsumexp = signature.classical.reduce(Tensor.logsumexp, argname_keepdims="keepdim")

        self.gather = signature.classical.take(lambda x, indices, axis=0: Tensor.gather(x, axis, indices))
        self.scatter = signature.classical.setitem(Tensor.scatter)
        self.scatter_reduce = signature.classical.setitem(Tensor.scatter_reduce)

        self.dot = signature.classical.dot(Tensor.dot)
        self.einsum = signature.classical.einsum(Tensor.einsum)

        self.roll = signature.classical.preserve_shape(Tensor.roll)
        self.flip = signature.classical.preserve_shape(Tensor.flip)
        self.sort = signature.classical.preserve_shape(Tensor.sort, num_outputs=2)

        self.softmax = signature.classical.preserve_shape(Tensor.softmax)
        self.log_softmax = signature.classical.preserve_shape(Tensor.log_softmax)

    def __call__(self, *args, **kwargs):
        return tracer.signature.python.call(self, args, kwargs)
