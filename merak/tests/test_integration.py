import pytest

from ..codegen import CodeGenerator

from ..lexer import MerakLexer
from ..parser import MerakParser
from ..analyzers.scopeBuilder import ScopeBuilder
from ..analyzers.typeChecker import TypeChecker

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
    impl contrateto {

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

    # scopeEnv = ScopeBuilder().build(tree)
    # TypeChecker(scopeEnv).check(tree)
    # code = CodeGenerator(scopeEnv, typeEnv).generate(tree)

    code = CodeGenerator().run(tree)
    a = 1
