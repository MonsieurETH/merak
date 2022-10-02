from py import code
import pytest

from merak.analyzers.symbolTableBuilder import SymbolTableBuilder

from ..codegen import CodeGenerator

from ..lexer import MerakLexer
from ..parser import MerakParser

# "Hello, world!".encode('utf-8').hex()

test = """
    impl contrateto {

        //abi UniswapV2Pair {
        //    getReserves();
        //    _safeTransfer(address token, address to, uint value);
        //}

        struct Data {
            A: u256;
            B: i256;
        }

        fn function1(param1: u256) -> (u256) {
            // This is a comment
            return param1 + 1;
            //pair1, pair2 = UniswapV2Pair(0x3333).getReserves();
        }
    }
    """

def test_basic():
    text = """
    impl contractName {

        struct Data {
            A: u256;
            B: i256;
        }

        fn function1(param1: u256) -> (u256) view {
            var1: u256 = 2;
            return param1 + 1 - var1;
        }
    }
    """

    merakLexer = MerakLexer()
    tokenized = merakLexer.tokenize(text)

    merakParser = MerakParser()
    tree = merakParser.parse(tokenized)

    symbolTable = SymbolTableBuilder().build(tree)
    codeIR = CodeGenerator()._code_gen(tree, scope=symbolTable.getScope())
    print(codeIR)
    a = 1

    # scopeEnv = ScopeBuilder().build(tree)
    # TypeChecker(scopeEnv).check(tree)
    # code = CodeGenerator(scopeEnv, typeEnv).generate(tree)

    #code = CodeGenerator().run(tree)
    a = 1
