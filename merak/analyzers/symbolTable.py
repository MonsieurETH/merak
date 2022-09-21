from .scopeBuilder import ScopeBuilder


class SymbolTable:
    def __init__(self) -> None:
        self.scope

    def build_scope(self, tree):
        self.scope = ScopeBuilder().build(tree)


