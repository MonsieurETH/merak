from sha3 import keccak_256


class HuffContract:
    def __init__(self, functions=[]) -> None:
        # TODO Create Code from function selectors
        code = "    0x00 calldataload 0xE0 shr \n"
        jumps = []
        for fn in functions:
            func = bytes(fn.selector, encoding="utf8")
            sha3_hash = keccak_256(func).hexdigest()
            funcSelector = "0x" + sha3_hash[:8]
            code += f"    dup1 {funcSelector} eq {fn.id} jumpi\n"
            jumps.append(fn.id)

        code += "\n    0x00 0x00 revert\n"

        for jump in jumps:
            code += (
                "    " + jump + ":\n" + "        " + jump.upper() + "()" + "\n"
            )

        self.main = HuffFunction("MAIN", [], [], None, code)
        self.functions = functions

    def __repr__(self) -> str:

        main = repr(self.main)
        funcs = ""
        for func in self.functions:
            funcs = funcs + repr(func) + "\n"

        return main + "\n" + funcs

class HuffGlobalVar:

    def __init__(self, id, type) -> None:
        self.id = id
        self.type = type

class HuffGlobalConst:

    def __init__(self, id, type, value) -> None:
        self.id = id
        self.type = type
        self.value = value


class HuffFunction:
    def __init__(self, id, args, returns, types, code) -> None:
        self.id = id
        self.args = args
        self.returns = returns
        self.types = types
        self.code = code

    @property
    def selector(self):
        return self.id

    def __repr__(self) -> str:
        return (
            f"#define macro {self.id.upper()}() = takes ({len(self.args)}) returns ({len(self.returns)})"
            + " { \n"
            + str(self.code)
            + " \n}"
        )

class Interface:
    def __init__(self, selector, returns=None, modifiers=None) -> None:
        self.selector = selector
        self.returns = returns
        self.modifiers = modifiers

    def __repr__(self) -> str:
        define = f"#define function {self.selector} "
        define += " ".join(self.modifiers)
        if self.returns is not None:
            define += "returns ({self.returns})"
        return define
