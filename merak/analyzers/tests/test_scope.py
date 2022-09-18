import pytest

from ...lexer import MerakLexer
from ...parser import MerakParser
from ..scopeBuilder import ScopeBuilder


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

    scopeTree = ScopeBuilder().build(tree)
    a = 1
