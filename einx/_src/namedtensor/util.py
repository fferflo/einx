class ExpressionIndicator:
    def __init__(self, text):
        self.text = text

    def create(self, pos):
        return f'Expression: "{self.text}"\n' + " " * 13 + "".join([("^" if i in pos else " ") for i in range(len(self.text))])

    def get_pos_for_literal(self, literal):
        pos = []
        for i in range(len(self.text)):
            if self.text[i:].startswith(literal):
                pos.extend(range(i, i + len(literal)))
        assert all(p >= 0 and p < len(self.text) for p in pos), f"{pos}"
        return pos

    def get_pos_for_exprs(self, exprs):
        if not isinstance(exprs, tuple | list):
            exprs = [exprs]
        pos = []
        for expr in exprs:
            pos.extend(range(expr.begin_pos, expr.end_pos))
        assert all(p >= 0 and p < len(self.text) for p in pos), f"{pos}"
        return pos

    def get_pos_for_axisnames(self, exprs, axisnames):
        if not isinstance(exprs, tuple | list):
            exprs = [exprs]
        from . import stage1, stage2, stage3

        pos = []
        for expr in exprs:
            if expr is not None:
                for expr in expr.nodes():
                    if isinstance(expr, stage1.Axis | stage2.Axis | stage3.Axis) and expr.name in axisnames:
                        pos.extend(range(expr.begin_pos, expr.end_pos))
        assert all(p >= 0 and p < len(self.text) for p in pos), f"{pos}"
        return pos

    def get_pos_for_ellipses(self, exprs):
        if not isinstance(exprs, tuple | list):
            exprs = [exprs]
        from . import stage1

        pos = []
        for expr in exprs:
            if expr is not None:
                for expr in expr.nodes():
                    if isinstance(expr, stage1.Ellipsis):
                        if expr.begin_pos >= 0:
                            pos.extend(range(expr.end_pos - 3, expr.end_pos))
        assert all(p >= 0 and p < len(self.text) for p in pos), f"{pos}"
        return pos

    def get_pos_for_concat(self, exprs):
        if not isinstance(exprs, tuple | list):
            exprs = [exprs]
        from . import stage1

        pos = []
        for expr in exprs:
            if expr is not None:
                for expr in expr.nodes():
                    if isinstance(expr, stage1.ConcatenatedAxis):
                        if expr.begin_pos >= 0:
                            pos.extend(range(expr.begin_pos, expr.end_pos))
        assert all(p >= 0 and p < len(self.text) for p in pos), f"{pos}"
        return pos

    def get_pos_for_brackets(self, exprs):
        if not isinstance(exprs, tuple | list):
            exprs = [exprs]
        from . import stage1

        pos = []
        for expr in exprs:
            if expr is not None:
                for expr in expr.nodes():
                    if isinstance(expr, stage1.Brackets):
                        if expr.begin_pos >= 0:
                            pos.extend([expr.begin_pos, expr.end_pos - 1])
        assert all(p >= 0 and p < len(self.text) for p in pos), f"{pos}"
        return pos
