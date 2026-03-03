from collections import defaultdict
import numpy as np
import einx._src.util.solver as solver
from .. import stage1
from einx._src.frontend.errors import RankError
from .tree import *
import uuid


def _input_expr(expr):
    if expr is None or isinstance(expr, stage1.Expression):
        return expr
    elif isinstance(expr, str):
        return stage1.parse_arg(expr)
    else:
        if isinstance(expr, np.ndarray):
            pass
        elif expr == [] or expr == ():
            expr = np.asarray(expr).astype("int32")
        else:
            try:
                expr = np.asarray(expr)
            except Exception as e:
                raise ValueError(f"Invalid expression '{expr}'") from e
        if not np.issubdtype(expr.dtype, np.integer):
            raise ValueError(f"Invalid expression '{expr}', must be integers")
        expr = " ".join([str(i) for i in expr.flatten()])
        expr = stage1.parse_arg(expr)
        return expr


def _get_expansion(expr):
    if isinstance(expr, str):
        return (stage1.parse_arg(expr).ndim,)
    elif isinstance(expr, stage1.Expression):
        return (expr.ndim,)
    elif isinstance(expr, np.ndarray):
        return tuple(expr.shape)
    else:
        return None


class Equation:
    def __init__(self, expr1, expr2=None, depth1=0, depth2=0, desc1=None, desc2=None):
        self.expansion1 = _get_expansion(expr1)
        self.expansion2 = _get_expansion(expr2)
        self.expr1 = _input_expr(expr1)
        self.expr2 = _input_expr(expr2)
        if desc1 is None:
            desc1 = f'expression "{self.expr1}"'
        if desc2 is None:
            desc2 = f'expression "{self.expr2}"'
        self.depth1 = depth1
        self.depth2 = None if expr2 is None else depth2
        self.desc1 = desc1
        self.desc2 = desc2


def solve(equations, invocation, verbose=False):
    exprs1 = [t.expr1 for t in equations]
    exprs2 = [t.expr2 for t in equations]
    expansions1 = [t.expansion1 for t in equations]
    expansions2 = [t.expansion2 for t in equations]
    depths1 = [t.depth1 for t in equations]
    depths2 = [t.depth2 for t in equations]
    descs1 = [t.desc1 for t in equations]
    descs2 = [t.desc2 for t in equations]
    if any(expr is not None and not isinstance(expr, stage1.Expression) for expr in exprs1 + exprs2):
        raise ValueError("Can only expand stage1.Expression")

    if verbose:
        print("(1) Equations:")
        for expansion1, expansion2, expr1, expr2, depth1, depth2, _, _ in zip(
            expansions1, expansions2, exprs1, exprs2, depths1, depths2, descs1, descs2, strict=False
        ):
            print(f'"{expr1}" (expansion={expansion1} depth={depth1}) = "{expr2}" (expansion={expansion2} depth={depth2})')

    expr_id_to_expr = {}
    ellipsis_id_to_exprs = defaultdict(list)
    axis_name_to_exprs = defaultdict(list)
    for root in exprs1 + exprs2:
        if root is not None:
            for expr in root.nodes():
                expr_id_to_expr[id(expr)] = expr
                if isinstance(expr, stage1.Ellipsis):
                    ellipsis_id_to_exprs[expr.ellipsis_id].append(expr)
                elif isinstance(expr, stage1.Axis):
                    axis_name_to_exprs[expr.name].append(expr)

    # ##### 1. Find expression depths #####
    equations = []

    symbolic_expr_depths = {}
    for root in exprs1 + exprs2:
        if root is not None:
            for expr in root.nodes():
                symbolic_expr_depths[id(expr)] = solver.Variable(f"symbolic_expr_depths[{id(expr)}]", str(expr))

    # Add equations: Depth relations between subexpressions
    for root in exprs1 + exprs2:
        if root is not None:
            for expr in root.nodes():
                if isinstance(expr, stage1.Ellipsis):
                    # Ellipsis increases depth by one
                    if verbose:
                        print(f"1. Depth({expr}) + 1 = Depth({expr.inner})")
                    equations.append((symbolic_expr_depths[id(expr)] + 1, symbolic_expr_depths[id(expr.inner)]))
                else:
                    # All other expressions have the same depth as their children
                    for child in expr.children:
                        if verbose:
                            print(f"2. Depth({expr}) = Depth({child})")
                        equations.append((symbolic_expr_depths[id(expr)], symbolic_expr_depths[id(child)]))

    # Add equations: Depth arguments
    for root, depth in zip(exprs1 + exprs2, depths1 + depths2, strict=False):
        if root is not None and depth is not None:
            if verbose:
                print(f"3. Depth({root}) = {depth}")
            equations.append((symbolic_expr_depths[id(root)], depth))

    # Add equations: Root depths
    for root1, root2, expansion1, expansion2 in zip(exprs1, exprs2, expansions1, expansions2, strict=False):
        if root1 is not None and root2 is not None and expansion1 is not None and expansion2 is not None:
            if verbose:
                print(f"4. Depth({root1}) + {len(expansion1)} = Depth({root2}) + {len(expansion2)}")
            equations.append((symbolic_expr_depths[id(root1)] + len(expansion1), symbolic_expr_depths[id(root2)] + len(expansion2)))

    # Add equations: Multiple occurrences of the same named axis must have the same depth
    symbolic_axis_depths = {}
    for root in exprs1 + exprs2:
        if root is not None:
            for axis in root.nodes():
                if isinstance(axis, stage1.Axis):
                    if axis.name not in symbolic_axis_depths:
                        symbolic_axis_depths[axis.name] = solver.Variable(f"symbolic_axis_depths[{axis.name}]", axis.name)
                    equations.append((symbolic_expr_depths[id(axis)], symbolic_axis_depths[axis.name]))

    # Add equations: Ellipses with the same id must have the same depth
    symbolic_ellipsis_depths = {}
    for root in exprs1 + exprs2:
        if root is not None:
            for ellipsis in root.nodes():
                if isinstance(ellipsis, stage1.Ellipsis):
                    if ellipsis.ellipsis_id not in symbolic_ellipsis_depths:
                        symbolic_ellipsis_depths[ellipsis.ellipsis_id] = solver.Variable(f"symbolic_ellipsis_depths[{ellipsis.ellipsis_id}]", str(ellipsis))
                    equations.append((symbolic_expr_depths[id(ellipsis)], symbolic_ellipsis_depths[ellipsis.ellipsis_id]))

    # Solve
    def solve(equations):
        solutions = solver.solve(equations)

        expr_depths = {}
        for k, v in solutions.items():
            if k.startswith("symbolic_expr_depths["):
                expr_depths[int(k[len("symbolic_expr_depths[") : -1])] = int(v)

        # Raise exception on negative depths
        failed_exprs = set()
        for root in exprs1 + exprs2:
            if root is not None:
                for expr in root.nodes():
                    if id(expr) in expr_depths and expr_depths[id(expr)] < 0:
                        failed_exprs.add(str(expr))
        if len(failed_exprs) > 0:
            raise solver.SolveExceptionNoSolution()

        # Raise exception on missing depths
        failed_exprs = set()
        for root in exprs1 + exprs2:
            if root is not None:
                for expr in root.nodes():
                    if id(expr) not in expr_depths:
                        failed_exprs.add(str(expr))
        if len(failed_exprs) > 0:
            raise solver.SolveExceptionTooManySolutions()

        return expr_depths

    illegal_depth_message = (
        "Please check the following:\n"
        " - Each axis name may be used either with or without an ellipsis, "
        "but not both.\n - The rank of a constraint "
        "must be equal to or less than the number of ellipses around the corresponding axis."
    )
    try:
        expr_depths = solve(equations)
    except solver.SolveExceptionTooManySolutions as e:
        raise RankError(
            invocation,
            message=f"Found an invalid usage of ellipses and/or constraints.\n%EXPR%\n{illegal_depth_message}",
            constraints=zip(exprs1, exprs2, strict=False),
        ) from e
    except solver.SolveExceptionNoSolution as e:
        # Check which axes are leading to contradiction
        def axisnames_in_equation(eq):
            axisnames = set()
            for term in eq:
                if isinstance(term, solver.Expression):
                    for var in term:
                        if isinstance(var, solver.Variable):
                            if var.id.startswith("symbolic_axis_depths["):
                                axisnames.add(var.id[len("symbolic_axis_depths[") : -1])
                            elif var.id.startswith("symbolic_expr_depths["):
                                expr_id = var.id[len("symbolic_expr_depths[") : -1]
                                expr = expr_id_to_expr.get(int(expr_id), None)
                                for expr in expr.nodes():
                                    if isinstance(expr, stage1.Axis):
                                        axisnames.add(expr.name)
            return list(axisnames)

        def equation_contains_axisconsistency(eq, axis_name):
            for term in eq:
                if isinstance(term, solver.Expression):
                    for var in term:
                        if isinstance(var, solver.Variable):
                            if var.id == f"symbolic_axis_depths[{axis_name}]":
                                return True
            return False

        contradicting_axis_names = []
        for axis_name in axis_name_to_exprs.keys():
            try_equations = [eq for eq in equations if [axis_name] != axisnames_in_equation(eq) and not equation_contains_axisconsistency(eq, axis_name)]
            try:
                solve(try_equations)
                still_contradicts = False
            except solver.SolveExceptionNoSolution:
                still_contradicts = True
            except solver.SolveExceptionTooManySolutions:
                still_contradicts = False
            if not still_contradicts:
                contradicting_axis_names.append(axis_name)

        if len(contradicting_axis_names) > 0:
            contradicting_axis_names = sorted(contradicting_axis_names)
            if len(contradicting_axis_names) == 1:
                axis = f"axis {contradicting_axis_names[0]}"
            else:
                axis = f"axes {', '.join(contradicting_axis_names)}"
            raise RankError(
                invocation,
                message=f"Found an invalid usage of ellipses and/or constraints for the {axis}:\n%EXPR%\n{illegal_depth_message}",
                pos=invocation.indicator.get_pos_for_axisnames(exprs1 + exprs2, contradicting_axis_names),
                constraints=zip(exprs1, exprs2, strict=False),
            ) from e
        else:
            raise RankError(
                invocation,
                message=f"Found an invalid usage of ellipses and/or constraints.\n%EXPR%\n{illegal_depth_message}",
                constraints=zip(exprs1, exprs2, strict=False),
            ) from e

    # Expand ellipses and add missing dimensions to expansions
    for exprs, expansions, _depths in zip([exprs1, exprs2], [expansions1, expansions2], [depths1, depths2], strict=False):
        for i in range(len(exprs)):
            if exprs[i] is not None:
                missing_depth = expr_depths[id(exprs[i])]
                assert missing_depth >= 0

                # Add missing dimensions to expansions
                if expansions[i] is not None:
                    assert len(expansions[i]) >= 1
                    if missing_depth > 0:
                        expansions[i] = [None] * missing_depth + list(expansions[i])

                # Add missing ellipses around root expressions
                if missing_depth > 0:
                    for _ in range(missing_depth):
                        exprs[i] = stage1.Ellipsis(exprs[i], -1, -1, ellipsis_id=uuid.uuid4().int)
                        expr_depths[id(exprs[i])] = expr_depths[id(exprs[i].inner)] - 1

    # ##### 2. Find ellipsis expansions #####
    if verbose:
        print("##################### Finding ellipsis expansions #####################")
    equations = []

    symbolic_expr_expansions = {}
    for root in exprs1 + exprs2:
        if root is not None:
            for expr in root.nodes():
                for depth in range(expr_depths[id(expr)] + 1):
                    key = (id(expr), depth)
                    symbolic_expr_expansions[key] = solver.Variable(f"symbolic_expr_expansions[{id(expr)},{depth}]", f"{expr} at depth {depth}")

    # Add equations: Expansion of an expression at depth d (less than own depth)
    # is equal to the expansion of each child at depth d
    for root in exprs1 + exprs2:
        if root is not None:
            for expr in root.nodes():
                for depth in range(expr_depths[id(expr)]):
                    for child in expr.children:
                        if verbose:
                            print(f"1. Expansion({expr}, depth={depth}) = Expansion({child}, depth={depth})")
                        equations.append((symbolic_expr_expansions[(id(expr), depth)], symbolic_expr_expansions[(id(child), depth)]))

    # Add equations: Relations between expressions and their children
    for root in exprs1 + exprs2:
        if root is not None:
            for expr in root.nodes():
                depth = expr_depths[id(expr)]
                if isinstance(expr, stage1.List):
                    v = sum(symbolic_expr_expansions[(id(child), depth)] for child in expr.children)
                    if verbose:
                        if len(expr.children) > 0:
                            print(
                                f"2. Expansion({expr}, depth={depth}) = " + " + ".join([f"Expansion({child}, depth={depth})" for child in expr.children]) + ""
                            )
                        else:
                            print(f"2. Expansion({expr}, depth={depth}) = 0")
                elif isinstance(expr, stage1.ConcatenatedAxis):
                    v = 1
                    if verbose:
                        print(f"2. Expansion({expr}, depth={depth}) = 1")
                elif isinstance(expr, stage1.Axis):
                    v = 1
                    if verbose:
                        print(f"2. Expansion({expr}, depth={depth}) = 1")
                elif isinstance(expr, stage1.FlattenedAxis):
                    v = 1
                    if verbose:
                        print(f"2. Expansion({expr}, depth={depth}) = 1")
                elif isinstance(expr, stage1.Brackets):
                    v = symbolic_expr_expansions[(id(expr.inner), depth)]
                    if verbose:
                        print(f"2. Expansion({expr}, depth={depth}) = Expansion({expr.inner}, depth={depth})")
                elif isinstance(expr, stage1.Ellipsis):
                    v = symbolic_expr_expansions[(id(expr.inner), depth)]
                    if verbose:
                        print(f"2. Expansion({expr}, depth={depth}) = Expansion({expr.inner}, depth={depth})")
                else:
                    raise AssertionError(f"{expr}")
                equations.append((symbolic_expr_expansions[(id(expr), depth)], v))

    # Add equations: Expansions stored in "expansions"
    for expansion1, expansion2, expr1, expr2, desc1, desc2 in zip(expansions1, expansions2, exprs1, exprs2, descs1, descs2, strict=False):
        if expansion1 is not None and expansion2 is not None:
            if len(expansion1) != len(expansion2) or any(
                e1 is not None and e2 is not None and e1 != e2 for e1, e2 in zip(expansion1, expansion2, strict=False)
            ):
                raise RankError(
                    invocation,
                    message=f"The number of dimensions of the {desc1} does not match the number of dimensions of the {desc2}.\n%EXPR%",
                    constraints=zip(exprs1, exprs2, strict=False),
                )

        if expansion1 is not None and expansion2 is not None:
            expansion = [e1 if e1 is not None else e2 for e1, e2 in zip(expansion1, expansion2, strict=False)]
        elif expansion1 is not None:
            expansion = expansion1
        elif expansion2 is not None:
            expansion = expansion2
        else:
            expansion = None

        if expansion is not None:
            for depth, e in enumerate(expansion):
                if e is not None:
                    if expr1 is not None and depth <= expr_depths[id(expr1)]:
                        if verbose:
                            print(f"3. Expansion({expr1}, depth={depth}) = {int(e)}")
                        equations.append((symbolic_expr_expansions[(id(expr1), depth)], int(e)))
                    if expr2 is not None and depth <= expr_depths[id(expr2)]:
                        if verbose:
                            print(f"3. Expansion({expr2}, depth={depth}) = {int(e)}")
                        equations.append((symbolic_expr_expansions[(id(expr2), depth)], int(e)))

    # Add equations: Multiple occurrences of the same named axis must have the same expansions
    symbolic_axis_expansions = {}
    for root in exprs1 + exprs2:
        if root is not None:
            for axis in root.nodes():
                if isinstance(axis, stage1.Axis):
                    for depth in range(expr_depths[id(axis)] + 1):
                        if axis.name not in symbolic_axis_expansions:
                            symbolic_axis_expansions[(axis.name, depth)] = solver.Variable(
                                f"symbolic_axis_expansions[{axis.name},{depth}]", f"{axis.name} at depth {depth}"
                            )
                        equations.append((symbolic_expr_expansions[(id(axis), depth)], symbolic_axis_expansions[(axis.name, depth)]))

    # Add equations: Ellipses with the same id must be repeated the same number of times
    symbolic_ellipsis_repetitions = {}
    for root in exprs1 + exprs2:
        if root is not None:
            for ellipsis in root.nodes():
                if isinstance(ellipsis, stage1.Ellipsis):
                    for depth in range(expr_depths[id(ellipsis)] + 1):
                        if ellipsis.ellipsis_id not in symbolic_ellipsis_repetitions:
                            symbolic_ellipsis_repetitions[(ellipsis.ellipsis_id, depth)] = solver.Variable(
                                f"symbolic_ellipsis_repetitions[{ellipsis.ellipsis_id},{depth}]", f"{ellipsis} at depth {depth}"
                            )
                        repetition = symbolic_ellipsis_repetitions[(ellipsis.ellipsis_id, depth)]

                        if verbose:
                            print(
                                f"4. Repetition({ellipsis}, depth={depth}) * Expansion({ellipsis.inner}, depth={depth + 1}) = "
                                f"Expansion({ellipsis}, depth={depth})"
                            )

                        equations.append((
                            repetition * symbolic_expr_expansions[(id(ellipsis.inner), depth + 1)],
                            symbolic_expr_expansions[(id(ellipsis), depth)],
                        ))

    # Add equations: Same root expansions
    for root1, root2 in zip(exprs1, exprs2, strict=False):
        if root1 is not None and root2 is not None:
            assert expr_depths[id(root1)] == expr_depths[id(root2)]
            for depth in range(expr_depths[id(root1)] + 1):
                equations.append((symbolic_expr_expansions[(id(root1), depth)], symbolic_expr_expansions[(id(root2), depth)]))
                if verbose:
                    print(f"5. Expansion({root1}, depth={depth}) = Expansion({root2}, depth={depth})")

    # TODO: add equations: empty ellipses have expansion 0

    # Solve
    def solve(equations):
        solutions = solver.solve(equations)

        expansion_values = {}
        for k, v in solutions.items():
            if k.startswith("symbolic_expr_expansions["):
                k = k[len("symbolic_expr_expansions[") : -1]
                id_expr, depth = str(k).split(",")
                try:
                    id_expr = int(id_expr)
                except ValueError:
                    continue
                depth = int(depth)
                expansion_values[(id_expr, depth)] = int(v)

        # Raise exception on negative expansions
        failed_exprs = set()
        for root in exprs1 + exprs2:
            if root is not None:
                for expr in root.nodes():
                    depth = expr_depths[id(expr)]
                    key = (id(expr), depth)
                    if key in expansion_values and expansion_values[key] < 0:
                        failed_exprs.add(expr)
        if len(failed_exprs) > 0:
            raise solver.SolveExceptionNoSolution()

        # Raise exception on missing expansions for expressions not wrapped in flattened-axis
        def is_at_root(expr):
            parent = expr.parent
            while parent is not None:
                if isinstance(parent, stage1.FlattenedAxis):
                    return False
                parent = parent.parent
            return True

        failed_exprs = []
        for root in exprs1 + exprs2:
            if root is not None:
                for expr in root.nodes():
                    if not is_at_root(expr):
                        continue
                    depth = expr_depths[id(expr)]
                    key = (id(expr), depth)
                    if key not in expansion_values:
                        failed_exprs.append(expr)
        if len(failed_exprs) > 0:
            raise solver.SolveExceptionTooManySolutions()

        return expansion_values

    try:
        expansion_values = solve(equations)
    except solver.SolveExceptionTooManySolutions as e:
        raise RankError(
            invocation,
            message="Failed to uniquely determine the expansion of ellipses in the expression. Please provide more constraints.\n%EXPR%",
            pos=invocation.indicator.get_pos_for_ellipses(exprs1 + exprs2),
            constraints=zip(exprs1, exprs2, strict=False),
        ) from e
    except solver.SolveExceptionNoSolution as e:
        if any(isinstance(expr, stage1.Ellipsis) for expr in exprs1 + exprs2):
            message = "Failed to find an expansion of ellipses in the expression such that the number of dimensions matches all given constraints.\n%EXPR%"
        else:
            message = "The number of tensor dimensions and axes in the expression does not match."
        raise RankError(
            invocation, message=message, pos=invocation.indicator.get_pos_for_ellipses(exprs1 + exprs2), constraints=zip(exprs1, exprs2, strict=False)
        ) from e

    # Expand ellipses and map stage1 expressions to stage2 expressions
    def map(expr, ellipsis_indices):
        if isinstance(expr, list):
            return [c for expr in expr for c in map(expr, ellipsis_indices=ellipsis_indices)]
        elif isinstance(expr, stage1.Axis):
            return [
                Axis(
                    expr.name + "".join(f".{idx}" for idx, _ in ellipsis_indices),
                    expr.value,
                    ellipsis_indices=ellipsis_indices,
                    begin_pos=expr.begin_pos,
                    end_pos=expr.end_pos,
                )
            ]
        elif isinstance(expr, stage1.List):
            return map(expr.children, ellipsis_indices=ellipsis_indices)
        elif isinstance(expr, stage1.ConcatenatedAxis):
            return [
                ConcatenatedAxis.create(
                    [List.create(map(c, ellipsis_indices=ellipsis_indices), ellipsis_indices=ellipsis_indices) for c in expr.children],
                    ellipsis_indices=ellipsis_indices,
                    begin_pos=expr.begin_pos,
                    end_pos=expr.end_pos,
                )
            ]
        elif isinstance(expr, stage1.FlattenedAxis):
            return [
                FlattenedAxis.create(
                    List.create(map(expr.inner, ellipsis_indices=ellipsis_indices), ellipsis_indices=ellipsis_indices),
                    ellipsis_indices=ellipsis_indices,
                    begin_pos=expr.begin_pos,
                    end_pos=expr.end_pos,
                )
            ]
        elif isinstance(expr, stage1.Brackets):
            return [
                Brackets.create(
                    List.create(map(expr.inner, ellipsis_indices=ellipsis_indices), ellipsis_indices=ellipsis_indices),
                    ellipsis_indices=ellipsis_indices,
                    begin_pos=expr.begin_pos,
                    end_pos=expr.end_pos,
                )
            ]
        elif isinstance(expr, stage1.Ellipsis):
            key = (id(expr), expr_depths[id(expr)])
            if key in expansion_values:
                # Ellipsis is expanded
                expansion = expansion_values[key]
                assert expansion >= 0
                return [c for i in range(expansion) for c in map(expr.inner, ellipsis_indices=ellipsis_indices + [(i, expansion)])]
            else:
                # Ellipsis is not expanded -> convert to named axis
                return [
                    Axis(
                        f"UnexpandedEllipsis({expr})" + "".join(f".{idx}" for idx, _ in ellipsis_indices),
                        expr.value,
                        ellipsis_indices=ellipsis_indices,
                        begin_pos=expr.begin_pos,
                        end_pos=expr.end_pos,
                    )
                ]
        else:
            raise AssertionError(f"{expr}")

    exprs1 = [List.create(map(root, ellipsis_indices=[]), ellipsis_indices=[]) if root is not None else None for root in exprs1]
    exprs2 = [List.create(map(root, ellipsis_indices=[]), ellipsis_indices=[]) if root is not None else None for root in exprs2]

    return exprs1, exprs2
