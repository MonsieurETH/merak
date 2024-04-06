use crate::parser::parse;
use crate::type_checking::TypeChecker;

mod ast;
mod cfg;
mod environment;
mod parser;
mod ssa;
mod type_checking;

fn main() {
    let ast = parse(
        "
        contract Vault[Open, Full, Locked] {
            
        }

        Vault@Open(owner) {
            function stateful payable withdraw({amount: int | amount > 0}) -> {ret: bool | ret == false} {
                var ret: bool;
                if (amount > 0) {
                    ret = false;
                } else {
                    ret = true;
                }
                return ret;
            }
        }",
    )
    .unwrap();

    let mut tc = TypeChecker::new();
    let _env = tc.check_program(&ast);

    //println!("{:#?}", ast)
}
