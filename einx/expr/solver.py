import sympy
import math


class Expression:
    def __init__(self):
        pass

    def __add__(self, other):
        return Sum([self, other])

    def __radd__(self, other):
        return Sum([other, self])

    def __mul__(self, other):
        return Product([self, other])

    def __rmul__(self, other):
        return Product([other, self])


class Variable(Expression):
    def __init__(self, id, name, integer=True):
        Expression.__init__(self)
        self.id = id
        self.name = name
        self.integer = integer

    def __iter__(self):
        yield self

    def __eq__(self, other):
        return isinstance(other, Variable) and self.id == other.id

    def __hash__(self):
        return hash(self.id)

    def __str__(self):
        return f"{self.name}"

    def sympy(self):
        return sympy.Symbol(self.id, integer=self.integer)


class Constant(Expression):
    def __init__(self, value):
        Expression.__init__(self)
        self.value = value

    def __iter__(self):
        yield self

    def __eq__(self, other):
        return isinstance(other, Constant) and self.value == other.value

    def __hash__(self):
        return hash(self.value)

    def __str__(self):
        return str(self.value)

    def sympy(self):
        return self.value


class Sum(Expression):
    @staticmethod
    def maybe(children):
        if len(children) == 0:
            return Constant(0)
        elif len(children) == 1:
            return children[0]
        elif all(isinstance(c, Constant) for c in children):
            return Constant(sum(c.value for c in children))
        else:
            return Sum(children)

    def __init__(self, children):
        Expression.__init__(self)
        self.children = [to_term(c) for c in children]

    def __iter__(self):
        yield self
        for child in self.children:
            yield from child

    def __eq__(self, other):
        return isinstance(other, Sum) and self.children == other.children

    def __hash__(self):
        return hash(tuple(self.children))

    def __str__(self):
        return " + ".join(str(c) for c in self.children)

    def sympy(self):
        return sum([c.sympy() for c in self.children])


class Product(Expression):
    @staticmethod
    def maybe(children):
        if len(children) == 0:
            return Constant(1)
        elif len(children) == 1:
            return children[0]
        elif all(isinstance(c, Constant) for c in children):
            return Constant(math.prod(c.value for c in children))
        else:
            return Product(children)

    def __init__(self, children):
        Expression.__init__(self)
        self.children = [to_term(c) for c in children]

    def __iter__(self):
        yield self
        for child in self.children:
            yield from child

    def __eq__(self, other):
        return isinstance(other, Product) and self.children == other.children

    def __hash__(self):
        return hash(tuple(self.children))

    def __str__(self):
        return " * ".join(str(c) for c in self.children)

    def sympy(self):
        return math.prod([c.sympy() for c in self.children])


def to_term(x):
    if isinstance(x, int):
        return Constant(x)
    else:
        if not isinstance(x, Expression):
            raise TypeError(f"Expected Expression, got {type(x)}")
        return x


class SolveException(Exception):
    def __init__(self, message):
        super().__init__(message)


def solve(equations):
    equations = [(to_term(t1), to_term(t2)) for t1, t2 in equations]
    equations = [(t1, t2) for t1, t2 in equations if t1 != t2]
    equations = list(set(equations))
    variables = {
        v.id: v for terms in equations for term in terms for v in term if isinstance(v, Variable)
    }

    # Find equivalence classes of variables to speed up sympy solver #####
    # Find constant definitions
    constants = {}  # id: constant value
    for t1, t2 in equations:
        if isinstance(t1, Variable) and isinstance(t2, Constant):
            if constants.get(t1.id, t2.value) != t2.value:
                raise SolveException(
                    f"Found contradictory values { {constants[t1.id], t2.value} } for "
                    f"expression '{t1.name}'"
                )
            constants[t1.id] = t2.value
        elif isinstance(t1, Constant) and isinstance(t2, Variable):
            if constants.get(t2.id, t1.value) != t1.value:
                raise SolveException(
                    f"Found contradictory values { {constants[t2.id], t1.value} } for "
                    f"expression '{t2.name}'"
                )
            constants[t2.id] = t1.value
        elif isinstance(t1, Constant) and isinstance(t2, Constant):
            if t1.value != t2.value:
                raise SolveException(
                    f"Found contradictory values {t1.value} != {t2.value} in input equation"
                )

    # Find equivalence classes of variables
    classes = {v: {v} for v in variables}  # id: set of equivalent ids
    for t1, t2 in equations:
        if isinstance(t1, Variable) and isinstance(t2, Variable):
            assert t1.id in classes and t2.id in classes
            set1 = classes[t1.id]
            set2 = classes[t2.id]
            for t_id in set2:
                classes[t_id] = set1
                set1.add(t_id)

    # For every class: Use constant if it exists, or create single class variable
    origvar_to_solvevar = {}  # id: Variable or Constant
    for eclass in {id(s): s for s in classes.values()}.values():
        if any(n in constants for n in eclass):
            # Use constant
            class_constants = {constants[n] for n in eclass if n in constants}
            if len(class_constants) != 1:
                names = {variables[a].name for a in eclass}
                if len(names) == 1:
                    raise SolveException(
                        f"Found contradictory values {class_constants} for expression "
                        f"'{next(iter(names))}'"
                    )
                else:
                    raise SolveException(
                        f"Found contradictory values {class_constants} for equivalent "
                        f"expressions {names}"
                    )
            v = Constant(next(iter(class_constants)))
        else:
            # Create new variable for class
            v = Variable(
                f"Class-{id(eclass)}",
                f"Equivalent expressions { {variables[a].name for a in eclass} }",
            )
        for n in eclass:
            assert n not in origvar_to_solvevar
            origvar_to_solvevar[n] = v

    # Apply to equations
    def replace(t):
        if isinstance(t, Variable) and t.id in origvar_to_solvevar:
            return origvar_to_solvevar[t.id]
        elif isinstance(t, Constant):
            return t
        elif isinstance(t, Sum):
            return Sum.maybe([replace(c) for c in t.children])
        elif isinstance(t, Product):
            return Product.maybe([replace(c) for c in t.children])
        else:
            raise AssertionError()

    equations2 = []
    for t1o, t2o in equations:
        t1 = replace(t1o)
        t2 = replace(t2o)
        if isinstance(t1, Constant) and isinstance(t2, Constant):
            if t1.value != t2.value:
                raise SolveException(
                    f"Found contradictory values {t1.value} != {t2.value} "
                    "for same equivalence class"
                )
        elif t1 != t2:
            equations2.append((t1, t2))
    equations = equations2

    # Solve remaining equations using sympy #####
    solutions = {}
    if len(equations) > 0:
        sympy_equations = [sympy.Eq(t1.sympy(), t2.sympy()) for t1, t2 in equations]

        if all(eq.is_Boolean and bool(eq) for eq in sympy_equations):
            solutions = {}
        else:
            solutions = sympy.solve(sympy_equations, set=True, manual=True)
            if solutions == []:
                solutions = {}
            elif isinstance(solutions, tuple) and len(solutions) == 2:
                variables, solutions = solutions
                if len(solutions) == 0:
                    raise SolveException("Sympy returned no solutions")
                elif len(solutions) > 1:
                    raise SolveException("Sympy returned multiple possible solutions")
                else:
                    solutions = next(iter(solutions))
                    solutions = {
                        str(k): int(v) for k, v in zip(variables, solutions) if v.is_number
                    }
            else:
                raise SolveException("Sympy returned unexpected result")

    # Determine values for original variables in equivalence classes
    orig_solutions = {}
    for k, v in origvar_to_solvevar.items():
        if isinstance(v, Constant):
            orig_solutions[k] = v.value
        elif isinstance(v, Variable):
            if v.id in solutions:
                orig_solutions[k] = solutions[v.id]
        else:
            raise AssertionError()

    return orig_solutions
