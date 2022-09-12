from collections import OrderedDict
from ..parser import AST as ast


class SymbolAlreadyDefined(Exception):
    pass


class Symbol:
    def __init__(self, name, type, value, static=False) -> None:
        self.name = name
        self.type = type
        self.value = value
        self.static = static


class SymbolVariable(Symbol):
    def __init__(self, name, type, value, static) -> None:
        super().__init__(name, type, value, static)


class SymbolConstant(Symbol):
    def __init__(self, name, type, value, static) -> None:
        super().__init__(name, type, value, static)


class SymbolTable:
    def __init__(self):
        self._symbols = OrderedDict()

    def __str__(self):
        s = "Symbols: {symbols}".format(
            symbols=[value for value in self._symbols.values()]
        )
        return s

    __repr__ = __str__

    def define(self, symbol):
        if symbol.name in self._symbols:
            raise SymbolAlreadyDefined(f"{symbol.name}")
        self._symbols[symbol.name] = symbol

    def lookup(self, name):
        symbol = self._symbols.get(name)
        return symbol
