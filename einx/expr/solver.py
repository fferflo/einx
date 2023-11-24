import sympy, math

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
    def __init__(self, name):
        Expression.__init__(self)
        self.name = name

    def __iter__(self):
        yield self

    def __eq__(self, other):
        return isinstance(other, Variable) and self.name == other.name

    def __hash__(self):
        return hash(self.name)

    def __str__(self):
        return f"V({self.name})"

    def sympy(self):
        return sympy.Symbol(self.name)

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
            raise TypeError('Expected Expression, got {}'.format(type(x)))
        return x

class SolveError(Exception):
    def __init__(self, message):
        self.message = message

def solve(equations):
    equations = [(to_term(t1), to_term(t2)) for t1, t2 in equations]
    equations = list(set(equations))
    variables = set(v.name for terms in equations for term in terms for v in term if isinstance(v, Variable))

    ##### Find equivalence classes of variables to speed up sympy solver #####
    # Find constant definitions
    constants = {}
    for t1, t2 in equations:
        if isinstance(t1, Variable) and isinstance(t2, Constant):
            if constants.get(t1.name, t2.value) != t2.value:
                raise SolveError("Failed to solve")
            constants[t1.name] = t2.value
        elif isinstance(t1, Constant) and isinstance(t2, Variable):
            if constants.get(t2.name, t1.value) != t1.value:
                raise SolveError("Failed to solve")
            constants[t2.name] = t1.value
        elif isinstance(t1, Constant) and isinstance(t2, Constant):
            if t1.value != t2.value:
                raise SolveError("Failed to solve")

    # Find equivalence classes of variables
    classes = {v: set([v]) for v in variables}
    for t1, t2 in equations:
        if isinstance(t1, Variable) and isinstance(t2, Variable):
            if t1.name in classes and t2.name in classes:
                set1 = classes[t1.name]
                set2 = classes[t2.name]
                for t_name in set2:
                    classes[t_name] = set1
                    set1.add(t_name)

    # For every class: Use constant if it exists, or create single class variable
    origvar_to_solvevar = {}
    for eclass in {id(s): s for s in classes.values()}.values():
        if any(n in constants for n in eclass):
            # Use constant
            class_constants = set(constants[n] for n in eclass if n in constants)
            if len(class_constants) != 1:
                raise SolveError("Failed to solve")
            v = Constant(next(iter(class_constants)))
        else:
            # Create new variable for class
            v = Variable(f"Class-{id(eclass)}")
        for n in eclass:
            assert not n in origvar_to_solvevar
            origvar_to_solvevar[n] = v

    # Apply to equations
    def replace(t):
        if isinstance(t, Variable) and t.name in origvar_to_solvevar:
            return origvar_to_solvevar[t.name]
        elif isinstance(t, Constant):
            return t
        elif isinstance(t, Sum):
            return Sum([replace(c) for c in t.children])
        elif isinstance(t, Product):
            return Product([replace(c) for c in t.children])
        else:
            assert False
    equations2 = []
    for t1, t2 in equations:
        t1 = replace(t1)
        t2 = replace(t2)
        if isinstance(t1, Constant) and isinstance(t2, Constant):
            if t1.value != t2.value:
                raise SolveError("Failed to solve")
        elif t1 != t2:
            equations2.append((t1, t2))
    equations = equations2

    ##### Solve remaining equations using sympy #####
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
                if len(solutions) != 1:
                    raise SolveError("Failed to solve")
                solutions = next(iter(solutions))
                solutions = {str(k): int(v) for k, v in zip(variables, solutions) if v.is_number}
            else:
                raise SolveError("Failed to solve")

    # Determine values for original variables in equivalence classes
    orig_solutions = {}
    for k, v in origvar_to_solvevar.items():
        if isinstance(v, Constant):
            orig_solutions[k] = v.value
        elif isinstance(v, Variable):
            if v.name in solutions:
                orig_solutions[k] = solutions[v.name]
        else:
            assert False

    return orig_solutions