from sly import Lexer


class MerakLexer(Lexer):
    tokens = {
        CONTRACT,
        ID,
        LBRACE,
        RBRACE,
        LPAREN,
        RPAREN,
        IF,
        ELSE,
        WHILE,
        FUNC,
        SEMICOLON,
        COLON,
        COMMA,
        RARROW,
        RETURN,
        PLUS,
        MINUS,
        UMINUS,
        TIMES,
        DIVIDE,
        ASSIGN,
        NUMBER,
        INT,
        UINT,
        BOOL,
        PURE,
        VIEW,
        PAYABLE,
        NONPAYABLE,
        STORAGE,
        CONST,
        STRUCT,
    }

    # Tokens
    LBRACE = r"\{"
    LPAREN = r"\("
    RBRACE = r"\}"
    RPAREN = r"\)"
    SEMICOLON = r"\;"
    COLON = r"\:"
    COMMA = r"\,"
    RARROW = r"->"
    PLUS = r"\+"
    MINUS = r"\-"
    UMINUS = r"\-"
    TIMES = r"\*"
    DIVIDE = r"/"
    ASSIGN = r"="
    NUMBER = r"\d+"

    ID = r"[a-zA-Z_][a-zA-Z0-9_]*"

    # Special cases
    ID["impl"] = CONTRACT
    ID["if"] = IF
    ID["else"] = ELSE
    ID["while"] = WHILE
    ID["fn"] = FUNC
    ID["return"] = RETURN
    ID["pure"] = PURE
    ID["view"] = VIEW
    ID["payable"] = PAYABLE
    ID["nonpayable"] = NONPAYABLE
    ID["storage"] = STORAGE
    ID["const"] = CONST
    ID["i256"] = INT
    ID["u256"] = UINT
    ID["bool"] = BOOL
    ID["struct"] = STRUCT

    # Ignored pattern
    ignore = " \t"
    ignore_newline = r"\n+"
    ignore_comment = r"\/\/.*"

    # Extra action for newlines
    def ignore_newline(self, t):
        self.lineno += t.value.count("\n")

    def error(self, t):
        print("Illegal character '%s'" % t.value[0])
        self.index += 1
