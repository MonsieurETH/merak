import pytest

from ...lexer import MerakLexer
from ...parser import MerakParser
from ..symbolTableBuilder import (
    Scope,
    SymbolTableBuilder,
    GlobalScope,
    StructScope,
    SymbolConstant,
    SymbolVariable,
    FunctionScope,
    SymbolArgument,
    LocalScope
)


def test_basic_scope():
    text = """
    impl contrateto {

        struct Data {
            A: u256;
            B: i256;
        }

        storage const supply: i256 = 10000;

        fn function1(param1: u256) -> (u256) view {
            return param1 + 1;
        }

        fn function2() -> (u256) {
            var1: u256 = 10;
        }
    }
    """

    merakLexer = MerakLexer()
    tokenized = merakLexer.tokenize(text)

    merakParser = MerakParser()
    tree = merakParser.parse(tokenized)

    symbolTable = SymbolTableBuilder().build(tree)
    scopeTree = symbolTable.getScope()

    assert isinstance(scopeTree, GlobalScope)
    assert scopeTree.parent is None

    assert len(scopeTree.symbols) == 1
    globalSymbol = scopeTree.symbols[0]
    assert isinstance(globalSymbol, SymbolConstant)
    assert globalSymbol.name == 'supply'
    assert globalSymbol.type == 'i256'
    # assert isinstance(globalSymbol.value, NumberExpression) ?
    assert globalSymbol.value.number == '10000'
    assert globalSymbol.value.sign == '+'

    assert len(scopeTree.childs) == 3

    structChild = scopeTree.childs[0]
    assert isinstance(structChild, StructScope)
    assert structChild.parent == scopeTree
    assert structChild.name == 'Data'
    assert len(structChild.symbols) == 2
    assert len(structChild.childs) == 0
    assert isinstance(structChild.symbols[0], SymbolVariable)
    assert isinstance(structChild.symbols[1], SymbolVariable)
    assert structChild.symbols[0].scope == structChild
    assert structChild.symbols[1].scope == structChild

    function1 = scopeTree.childs[1]
    assert isinstance(function1, FunctionScope)
    assert function1.parent == scopeTree
    assert function1.name == 'function1'
    assert len(function1.symbols) == 1
    function1arg = function1.symbols[0]
    assert isinstance(function1arg, SymbolArgument)
    assert function1arg.name == 'param1'
    assert function1arg.type == 'u256'
    assert function1arg.value == None
    assert function1arg.scope == function1
    assert len(function1.childs) == 1
    function1Local = function1.childs[0]
    assert isinstance(function1Local, LocalScope)
    assert function1Local.parent == function1
    assert len(function1Local.symbols) == 0
    assert len(function1Local.childs) == 0

    function2 = scopeTree.childs[2]
    assert isinstance(function2, FunctionScope)
    assert function2.parent == scopeTree
    assert function2.name == 'function2'
    assert len(function2.symbols) == 0
    assert len(function2.childs) == 1
    function2Local = function2.childs[0]
    assert isinstance(function2Local, LocalScope)
    assert function2Local.parent == function2
    assert len(function2Local.childs) == 0
    assert len(function2Local.symbols) == 1

    function2symbol = function2Local.symbols[0]
    assert isinstance(function2symbol, SymbolVariable)
    assert function2symbol.name == 'var1'
    assert function2symbol.type == 'u256'
    assert function2symbol.scope == function2Local
    # assert isinstance(function2symbol.value, NumberExpression) ?
    assert function2symbol.value.number == '10'
    assert function2symbol.value.sign == '+'





