import einx._src.tracer as tracer
import einx._src.util.pytree as pytree
import numpy as np


class Scope:
    def __init__(self):
        self.required_scopes = None
        self.parent = None

    def is_predecessor_of(self, other):
        if other is None:
            raise ValueError("Other scope is None.")
        elif id(self) == id(other):
            return True
        elif other.parent is None:
            return False
        else:
            return self.is_predecessor_of(other.parent)


class _IntermediateMap:
    def __init__(self, global_scope):
        self.id_to_scopes = {}
        self.global_scope = global_scope

    def __contains__(self, x):
        if isinstance(x, str | int | float | bool | np.integer | np.floating | np.bool_) or x is None or x == () or x == []:
            return True

        return all(id(y) in self.id_to_scopes for y in pytree.flatten(x))

    def __getitem__(self, x):
        if isinstance(x, str | int | float | bool | np.integer | np.floating | np.bool_) or x is None or x == () or x == []:
            return [self.global_scope]

        result = [scope for y in pytree.flatten(x) for scope in self.id_to_scopes[id(y)]]
        assert isinstance(result, list) and all(isinstance(s, Scope) for s in result) and len(result) > 0
        return result

    def __setitem__(self, x, scopes):
        assert isinstance(scopes, list) and all(isinstance(s, Scope) for s in scopes) and len(scopes) > 0
        if isinstance(x, tracer.Tracer | tracer.Graph):
            self.id_to_scopes[id(x)] = scopes
        else:
            raise TypeError(f"Unsupported type for key: {type(x)}")

    def items(self):
        return self.id_to_scopes.items()

    def all_scopes(self):
        scopes = [scope for scopes in self.id_to_scopes.values() for scope in scopes]
        scopes = list({id(s): s for s in scopes}.values())
        return scopes


class Map:
    def __init__(self, global_scope, scopes):
        self.id_to_scope = {}
        self.global_scope = global_scope
        self.scopes = scopes

    def _find_common_scope(self, scopes):
        scopes = list({id(s): s for s in scopes}.values()) + [self.global_scope]

        scope = scopes[0]
        for scope2 in scopes[1:]:
            if id(scope) == id(scope2):
                pass
            elif scope.is_predecessor_of(scope2):
                scope = scope2
            elif scope2.is_predecessor_of(scope):
                pass
            else:
                raise ValueError(f"Scopes {scope} and {scope2} are not in a predecessor relationship, cannot determine common scope.")
        return scope

    def __getitem__(self, x):
        if isinstance(x, str | int | float | bool | np.integer | np.floating | np.bool_) or x is None or x == () or x == []:
            return self.global_scope

        if id(x) in self.id_to_scope:
            return self.id_to_scope[id(x)]
        elif isinstance(x, list | tuple):
            return self._find_common_scope([self[y] for y in x])
        elif isinstance(x, dict):
            return self._find_common_scope([self[y] for y in list(x.keys()) + list(x.values())])
        else:
            raise KeyError(f"Key {x} not found in scope map.")

    @property
    def root(self):
        return self.global_scope

    def values(self):
        return self.scopes


def get_scopes(x):
    # Determine required scopes for each tracer and dependencies between scopes
    # {tracer -> [required scopes]}
    global_scope = Scope()
    global_scope.required_scopes = [global_scope]
    required_scopes = _IntermediateMap(global_scope)

    def _get_required_scopes(x):
        if x not in required_scopes:
            if isinstance(x, tracer.Tracer) and x.origin is None:
                raise ValueError(f"Tracer {x} has no origin, cannot determine required scopes.")
            elif isinstance(x, tracer.Tracer) and x.origin is not None:
                input_scopes = [scope for input in x.origin.inputs for scope in _get_required_scopes(input)] + [global_scope]
                result = list({id(s): s for s in input_scopes}.values())
                required_scopes[x] = result

                for output in pytree.flatten(x.origin.output):
                    _get_required_scopes(output)

            elif isinstance(x, list | tuple):
                input_scopes = [scope for input in x for scope in _get_required_scopes(input)] + [global_scope]
                result = list({id(s): s for s in input_scopes}.values())
            elif isinstance(x, dict):
                input_scopes = [scope for input in list(x.keys()) + list(x.values()) for scope in _get_required_scopes(input)] + [global_scope]
                result = list({id(s): s for s in input_scopes}.values())
            elif isinstance(x, tracer.Graph):
                inner_scope = Scope()
                for input in x.inputs:
                    required_scopes[input] = [inner_scope]
                required_output_scopes = _get_required_scopes(x.output)
                required_output_scopes = [scope for scope in required_output_scopes if id(scope) != id(inner_scope)]
                if len(required_output_scopes) == 0:
                    required_output_scopes = [global_scope]
                assert len(required_output_scopes) > 0
                inner_scope.required_scopes = required_output_scopes
                required_scopes[x] = required_output_scopes
                result = required_output_scopes
            else:
                result = [global_scope]
        else:
            result = required_scopes[x]

        assert isinstance(result, list) and all(isinstance(s, Scope) for s in result) and len(result) > 0
        return result

    _get_required_scopes(x)

    scopes = required_scopes.all_scopes()

    # Convert dependencies to parent-child relationships
    def _is_predecessor_of(scope, scope2):
        assert scope2.required_scopes is not None
        if id(scope) == id(scope2):
            return True
        elif scope in scope2.required_scopes:
            return True
        else:
            return any(_is_predecessor_of(scope, s) for s in scope2.required_scopes)

    def _find_common_scope(scopes):
        scopes = list({id(s): s for s in scopes}.values())
        if len(scopes) == 0:
            raise ValueError("No scopes found")

        scope = scopes[0]
        for scope2 in scopes[1:]:
            if id(scope) == id(scope2):
                pass
            elif _is_predecessor_of(scope, scope2):
                scope = scope2
            elif _is_predecessor_of(scope2, scope):
                pass
            else:
                raise ValueError(f"Scopes {scope} and {scope2} are not in a predecessor relationship, cannot determine common scope.")
        return scope

    for scope in scopes:
        if id(scope) != id(global_scope):
            scope.parent = _find_common_scope(scope.required_scopes)

    # Create scopes map: {tracer -> innermost scope}
    scopes_map = Map(global_scope, scopes)
    for tracerid, scopes in required_scopes.items():
        scopes_map.id_to_scope[tracerid] = scopes_map._find_common_scope(scopes)

    return scopes_map
