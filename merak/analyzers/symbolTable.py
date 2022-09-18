from .scopeBuilder import ScopeBuilder


class SymbolAlreadyDefined(Exception):
    pass


class SymbolTable:
    def __init__(self) -> None:
        self.globals

    def build_scope(self, tree):
        self.globals = ScopeBuilder().build(tree)
