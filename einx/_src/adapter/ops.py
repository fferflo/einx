elementwise = [
    "add",
    "subtract",
    "multiply",
    "true_divide",
    "floor_divide",
    "divide",
    "logical_and",
    "logical_or",
    "where",
    "maximum",
    "minimum",
    "less",
    "less_equal",
    "greater",
    "greater_equal",
    "equal",
    "not_equal",
    "logaddexp",
    "exp",
    "log",
    "negative",
]

reduce = ["sum", "mean", "var", "std", "prod", "count_nonzero", "all", "any", "min", "max", "logsumexp"]

update_at = ["set_at", "add_at", "subtract_at"]

preserve_shape = ["flip", "roll", "sort", "argsort", "softmax", "log_softmax"]

argfind = ["argmax", "argmin"]

all = ["id", "dot", "get_at"] + elementwise + reduce + update_at + preserve_shape + argfind
