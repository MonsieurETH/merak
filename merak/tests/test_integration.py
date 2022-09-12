import pytest

from ..codegen import CodeGenerator

from ..lexer import MerakLexer
from ..parser import MerakParser
from ..codegen import BasicExecute


def test_basic():
    text = """
        impl contrateto {
        fn function1(param1: u256) -> (u256) {
            // This is a comment
            return param1 + 1;
        };
    }
    """

    llexer = MerakLexer()
    tokenized = llexer.tokenize(text)

    lparser = MerakParser()
    tree = lparser.parse(tokenized)

    code = CodeGenerator().run(tree)
    a = 1


def test_basic2():
    llexer = MerakLexer()
    lparser = MerakParser()

    text = """
        impl contrateto {
        fn algoif(roberto) -> (u256) {
            return roberto + 1;
        };
        fn algoels(alejandro) {
            algo = alejandro + 3;
        };
        fn algowhile(roxana) -> (u256) {
            roxana = 2;
        };
    }
    """

    # for a in llexer.tokenize(text):
    #    print(a)

    tree = lparser.parse(llexer.tokenize(text))

    aca = BasicExecute().run(tree)
    print(a)
    a = 1
