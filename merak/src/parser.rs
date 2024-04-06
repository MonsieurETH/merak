use crate::ast::{
    Block, Contract, ContractConstructor, ContractFunction, ContractState, ContractStateData,
    Expression, FunctionParameter, Program, Statement,
};
use pest::Parser;
use pest_derive::Parser;

#[derive(Parser)]
#[grammar = "merak.pest"]
pub struct MerakParser;

pub fn parse(input: &str) -> Result<Program, Box<pest::error::Error<Rule>>> {
    let main = MerakParser::parse(Rule::main, input)?;
    let mut program = Program::new();
    let pairs = main.into_iter().next().unwrap().into_inner();
    for pair in pairs {
        match pair.as_rule() {
            Rule::contract => {
                let contract = parse_contract(pair)?;
                program.add_contract(contract);
            }
            Rule::state_def => {
                let state_def = parse_state_def(pair)?;
                program.add_state_def(state_def);
            }
            Rule::EOI => {}
            _ => unreachable!(),
        }
    }
    Ok(program)
}

fn parse_contract(
    pair: pest::iterators::Pair<Rule>,
) -> Result<Contract, Box<pest::error::Error<Rule>>> {
    let mut contract = Contract::new();
    for pair in pair.into_inner() {
        match pair.as_rule() {
            Rule::ident => {
                contract.set_name(pair.as_str().to_string());
            }
            Rule::bracket_content => {
                for pair in pair.into_inner() {
                    match pair.as_rule() {
                        Rule::ident => {
                            contract.add_state(pair.as_str().to_string());
                        }
                        Rule::EOI => {}
                        _ => unreachable!(),
                    }
                }
            }
            Rule::inner_contract => {
                for pair in pair.into_inner() {
                    match pair.as_rule() {
                        Rule::state_var | Rule::state_const => {
                            let state_var = parse_state_data(pair);
                            contract.add_state_data(state_var.unwrap());
                        }
                        Rule::constructor => {
                            let constructor = parse_constructor(pair);
                            contract.add_constructor(constructor.unwrap());
                        }
                        Rule::EOI => {}
                        _ => unreachable!(),
                    }
                }
            }
            Rule::EOI => {}
            _ => unreachable!(),
        }
    }
    Ok(contract)
}

fn parse_state_data(
    pair: pest::iterators::Pair<Rule>,
) -> Result<ContractStateData, Box<pest::error::Error<Rule>>> {
    let mut state_data = ContractStateData::new();

    let is_constant = match pair.as_rule() {
        Rule::state_var => false,
        Rule::state_const => true,
        _ => unreachable!(),
    };

    state_data.set_constant(is_constant);

    for inner_pair in pair.into_inner() {
        match inner_pair.as_rule() {
            Rule::ident => {
                state_data.set_name(inner_pair.as_str().to_string());
            }
            Rule::_type => {
                let ty = inner_pair.as_str().into();
                state_data.set_type(ty);
            }
            Rule::expression => {
                let expression = parse_expression(inner_pair)?;
                state_data.set_value(expression);
            }
            Rule::EOI => {}
            _ => unreachable!(),
        }
    }
    Ok(state_data)
}

fn parse_constructor(
    pair: pest::iterators::Pair<Rule>,
) -> Result<ContractConstructor, Box<pest::error::Error<Rule>>> {
    let mut constructor = ContractConstructor::new();
    for inner_pair in pair.into_inner() {
        match inner_pair.as_rule() {
            Rule::param_list => {
                for pair in inner_pair.into_inner() {
                    match pair.as_rule() {
                        Rule::param => {
                            let param = parse_param(pair)?;
                            constructor.add_parameter(param);
                        }
                        Rule::EOI => {}
                        _ => unreachable!(),
                    }
                }
            }
            Rule::block => {
                let mut block = parse_block(inner_pair)?;
                constructor.add_statements(&mut block);
            }
            Rule::EOI => {}
            _ => unreachable!(),
        }
    }
    Ok(constructor)
}

fn parse_state_def(
    pair: pest::iterators::Pair<Rule>,
) -> Result<ContractState, Box<pest::error::Error<Rule>>> {
    let mut state_def = ContractState::new();
    for inner_pair in pair.into_inner() {
        match inner_pair.as_rule() {
            Rule::contract_at_state => {
                let mut inner = inner_pair.into_inner();
                state_def.set_contract(inner.next().unwrap().as_str().to_string());
                state_def.set_name(inner.next().unwrap().as_str().to_string());
            }
            Rule::convenor => {
                state_def.set_convenor(inner_pair.as_str().to_string());
            }
            Rule::function => {
                let function = parse_function(inner_pair)?;
                state_def.add_function(function);
            }
            Rule::EOI => {}
            _ => unreachable!(),
        }
    }
    Ok(state_def)
}

fn parse_function(
    pair: pest::iterators::Pair<Rule>,
) -> Result<ContractFunction, Box<pest::error::Error<Rule>>> {
    let mut function = ContractFunction::new();
    for inner_pair in pair.into_inner() {
        match inner_pair.as_rule() {
            Rule::visibility => {
                function.set_visibility(inner_pair.as_str().into());
            }
            Rule::modifiers => {
                function.add_modifier(inner_pair.as_str().into());
            }
            Rule::ident => {
                function.set_name(inner_pair.as_str().to_string());
            }
            Rule::param_list => {
                for pair in inner_pair.into_inner() {
                    match pair.as_rule() {
                        Rule::param => {
                            let param = parse_param(pair)?;
                            function.add_parameter(param);
                        }
                        Rule::EOI => {}
                        _ => unreachable!(),
                    }
                }
            }
            Rule::function_return => {
                for pair in inner_pair.into_inner() {
                    match pair.as_rule() {
                        Rule::param => {
                            let param = parse_param(pair)?;
                            function.set_return_type(param);
                        }
                        Rule::EOI => {}
                        _ => unreachable!(),
                    }
                }
            }
            Rule::block => {
                let mut block = parse_block(inner_pair)?;
                function.add_statements(&mut block);
            }
            Rule::EOI => {}
            _ => unreachable!(),
        }
    }
    Ok(function)
}

fn parse_param(
    pair: pest::iterators::Pair<Rule>,
) -> Result<FunctionParameter, Box<pest::error::Error<Rule>>> {
    let mut param = FunctionParameter::new();
    for inner_pair in pair.into_inner() {
        match inner_pair.as_rule() {
            Rule::ident => {
                param.set_name(inner_pair.as_str().to_string());
            }
            Rule::_type => {
                let ty = inner_pair.as_str().into();
                param.set_type(ty);
            }
            Rule::constraint => {
                let mut inner = inner_pair.into_inner(); // drop |
                let ident = inner.next().unwrap().as_str().to_string();
                let logical = inner.next().unwrap().as_str().into();
                let expr = parse_expression(inner.next().unwrap())?;
                param.set_constraint(Expression::binary_op(
                    Expression::identifier(ident),
                    logical,
                    expr,
                ));
            }
            Rule::EOI => {}
            _ => unreachable!(),
        }
    }
    Ok(param)
}

fn parse_block(
    pair: pest::iterators::Pair<Rule>,
) -> Result<Vec<Statement>, Box<pest::error::Error<Rule>>> {
    let mut block: Vec<Statement> = vec![];
    for inner_pair in pair.into_inner() {
        match inner_pair.as_rule() {
            Rule::statement => {
                let statement = parse_statement(inner_pair)?;
                block.push(statement);
            }
            Rule::EOI => {}
            _ => unreachable!(),
        }
    }
    Ok(block)
}

fn parse_statement(
    pair: pest::iterators::Pair<Rule>,
) -> Result<Statement, Box<pest::error::Error<Rule>>> {
    let pair = pair.into_inner().next().unwrap();
    let statement = match pair.as_rule() {
        Rule::expression => {
            let expr = parse_expression(pair)?;
            Statement::Expression(expr)
        }
        Rule::assignment_statement => {
            let mut inner = pair.into_inner();
            let ident = inner.next().unwrap().as_str().to_string();
            let expr = parse_expression(inner.next().unwrap())?;
            Statement::VarAssignment(ident, expr)
        }
        Rule::return_statement => {
            let expr = parse_expression(pair.into_inner().next().unwrap())?;
            Statement::Return(Some(expr))
        }
        Rule::if_statement => {
            let mut inner = pair.into_inner();
            let condition = parse_expression(inner.next().unwrap())?;
            let if_block = parse_block(inner.next().unwrap())?;
            let else_block = if let Some(else_block) = inner.next() {
                Some(parse_block(else_block)?)
            } else {
                None
            };
            Statement::If(condition, if_block, else_block)
        }
        Rule::while_statement => {
            let mut inner = pair.into_inner();
            let condition = parse_expression(inner.next().unwrap())?;
            let block = parse_block(inner.next().unwrap())?;
            Statement::While(condition, block)
        }
        Rule::var_declaration => {
            let mut inner = pair.into_inner();
            let ident = inner.next().unwrap().as_str().to_string();
            let ty = inner.next().unwrap().as_str().into();
            let expr = if let Some(expr) = inner.next() {
                Some(parse_expression(expr)?)
            } else {
                None
            };
            Statement::VarDeclaration(ident, ty, expr)
        }
        Rule::const_declaration => {
            let mut inner = pair.into_inner();
            let ident = inner.next().unwrap().as_str().to_string();
            let ty = inner.next().unwrap().as_str().into();
            let expr = parse_expression(inner.next().unwrap())?;
            Statement::ConstDeclaration(ident, ty, expr)
        }
        _ => unreachable!(),
    };
    Ok(statement)
}

fn parse_expression(
    pair: pest::iterators::Pair<Rule>,
) -> Result<Expression, Box<pest::error::Error<Rule>>> {
    println!("{:?}", pair.as_rule());
    match pair.as_rule() {
        Rule::expression => {
            let mut inner_pairs = pair.into_inner();
            let mut expr = parse_expression(inner_pairs.next().unwrap())?;

            while let Some(next_pair) = inner_pairs.next() {
                if let Rule::binop = next_pair.as_rule() {
                    let operator = next_pair.as_str().into();
                    let next_term = parse_expression(inner_pairs.next().unwrap())?;
                    expr = Expression::binary_op(expr, operator, next_term);
                }
            }
            Ok(expr)
        }
        Rule::term => {
            let inner_pairs = pair.into_inner();

            let (unops, factor_pair) =
                inner_pairs.partition::<Vec<_>, _>(|p| p.as_rule() == Rule::unop);

            let factor_expr = parse_expression(factor_pair.into_iter().next().unwrap())?;

            let expr = unops.into_iter().rev().fold(factor_expr, |acc, unop_pair| {
                let operator = unop_pair.as_str().into();
                Expression::unary_op(operator, acc)
            });

            Ok(expr)
        }
        Rule::factor => {
            let inner_pair = pair.into_inner().next().unwrap();
            match inner_pair.as_rule() {
                Rule::literal => parse_literal(inner_pair),
                Rule::expression => parse_expression(inner_pair),
                Rule::ident => {
                    let value = String::from(inner_pair.as_str());
                    Ok(Expression::identifier(value))
                }
                _ => unreachable!(),
            }
        }
        _ => unreachable!(),
    }
}

fn parse_literal(
    pair: pest::iterators::Pair<Rule>,
) -> Result<Expression, Box<pest::error::Error<Rule>>> {
    let inner = pair.into_inner().next().unwrap();
    Ok(Expression::Literal(inner.as_str().into()))
}

#[cfg(test)]
mod tests {

    use crate::ast::{Function, Literal, Modifier, Type, Visibility};

    use super::*;

    #[test]
    fn empty_contract() {
        let ast = parse("contract Empty[] {}").unwrap();
        assert_eq!(ast.contracts().len(), 1);
        assert_eq!(ast.state_defs().len(), 0);
        assert_eq!(ast.contracts()[0].name(), "Empty");
        assert_eq!(ast.contracts()[0].state_data().len(), 0);
        assert_eq!(ast.contracts()[0].states().len(), 0);
        assert!(ast.contracts()[0].constructor().is_none());
    }

    #[test]
    fn empty_contract_with_states() {
        let ast = parse("contract Vault[Open, Full, Locked] {}").unwrap();
        assert_eq!(ast.contracts().len(), 1);
        assert_eq!(ast.state_defs().len(), 0);
        assert_eq!(ast.contracts()[0].name(), "Vault");
        assert_eq!(ast.contracts()[0].state_data().len(), 0);
        assert_eq!(ast.contracts()[0].states().len(), 3);
        assert_eq!(ast.contracts()[0].states()[0], "Open");
        assert_eq!(ast.contracts()[0].states()[1], "Full");
        assert_eq!(ast.contracts()[0].states()[2], "Locked");
        assert!(ast.contracts()[0].constructor().is_none());
    }

    #[test]
    fn contract_with_state_data() {
        let ast = parse(
            "contract Vault[Open, Full, Locked] {
                state var owner: address;
                state const balance: int = 0;
            }",
        )
        .unwrap();
        assert_eq!(ast.contracts().len(), 1);
        assert_eq!(ast.state_defs().len(), 0);
        assert_eq!(ast.contracts()[0].name(), "Vault");
        assert_eq!(ast.contracts()[0].state_data().len(), 2);
        assert_eq!(ast.contracts()[0].state_data()[0].name(), "owner");
        assert_eq!(ast.contracts()[0].state_data()[0].is_constant(), false);
        assert_eq!(
            ast.contracts()[0].state_data()[0].get_type(),
            &Type::Address
        );
        assert!(ast.contracts()[0].state_data()[0].value().is_none());
        assert_eq!(ast.contracts()[0].state_data()[1].name(), "balance");
        assert_eq!(ast.contracts()[0].state_data()[1].is_constant(), true);
        assert_eq!(ast.contracts()[0].state_data()[1].get_type(), &Type::Int);
        assert!(ast.contracts()[0].constructor().is_none());
    }

    #[test]
    fn contract_with_constructor() {
        let ast = parse(
            "contract Vault[Open, Full, Locked] {
                constructor({owner: address}) {
                    2;
                }
            }",
        )
        .unwrap();
        assert_eq!(ast.contracts().len(), 1);
        assert_eq!(ast.state_defs().len(), 0);
        assert!(ast.contracts()[0].constructor().is_some());
        let constructor = ast.contracts()[0].constructor().as_ref().unwrap();
        assert_eq!(constructor.parameters().len(), 1);
        assert_eq!(constructor.parameters()[0].name(), "owner");
        assert_eq!(constructor.parameters()[0].get_type(), Type::Address);
        assert_eq!(constructor.statements().len(), 1);
    }

    #[test]
    fn contract_with_constructor_two_params() {
        let ast = parse(
            "contract Vault[Open, Full, Locked] {
                constructor({owner: address}, {amount: int}) {
                    2;
                }
            }",
        )
        .unwrap();
        assert_eq!(ast.contracts().len(), 1);
        assert_eq!(ast.state_defs().len(), 0);
        assert!(ast.contracts()[0].constructor().is_some());
        let constructor = ast.contracts()[0].constructor().as_ref().unwrap();
        assert_eq!(constructor.parameters().len(), 2);
        assert_eq!(constructor.parameters()[0].name(), "owner");
        assert_eq!(constructor.parameters()[0].get_type(), Type::Address);
        assert_eq!(constructor.parameters()[1].name(), "amount");
        assert_eq!(constructor.parameters()[1].get_type(), Type::Int);
        assert_eq!(constructor.statements().len(), 1);
    }

    #[test]
    fn contract_with_constrained_constructor() {
        let ast = parse(
            "contract Vault[Open, Full, Locked] {
                constructor({owner: address | owner != 0x1234}) {
                    false;
                }
            }",
        )
        .unwrap();
        assert_eq!(ast.contracts().len(), 1);
        assert_eq!(ast.state_defs().len(), 0);
        assert!(ast.contracts()[0].constructor().is_some());
        let constructor = ast.contracts()[0].constructor().as_ref().unwrap();
        assert_eq!(constructor.parameters().len(), 1);
        assert_eq!(constructor.parameters()[0].name(), "owner");
        assert_eq!(constructor.parameters()[0].get_type(), Type::Address);
        assert_eq!(
            constructor.parameters()[0].constraint().as_ref().unwrap(),
            &Expression::binary_op(
                Expression::identifier("owner".to_string()),
                "!=".into(),
                Expression::Literal(Literal::Address("0x1234".to_string()))
            )
        );
        assert_eq!(constructor.statements().len(), 1);
    }

    #[test]
    fn contract_with_state_def() {
        let ast = parse(
            "contract Vault[Open, Full, Locked] {
                state var owner: address;
                state const balance: int = 0;
            }

            Vault@Open(owner) {
                function stateful payable withdraw({amount: int | amount > 0}) {
                    var balance: int = balance - amount;
                    const n: int = balance;
                    return false;
                }
            }",
        )
        .unwrap();
        assert_eq!(ast.contracts().len(), 1);
        assert_eq!(ast.state_defs().len(), 1);
        assert_eq!(ast.state_defs()[0].contract(), "Vault");
        assert_eq!(ast.state_defs()[0].name(), "Open");
        assert_eq!(ast.state_defs()[0].convenor(), "owner");
        assert_eq!(ast.state_defs()[0].functions().len(), 1);
        let function = &ast.state_defs()[0].functions()[0];
        assert_eq!(function.visibility(), &Visibility::Function);
        assert_eq!(function.modifiers().len(), 2);
        assert_eq!(function.modifiers()[0], Modifier::Stateful);
        assert_eq!(function.modifiers()[1], Modifier::Payable);
        assert_eq!(function.name(), "withdraw");
        assert_eq!(function.parameters().len(), 1);
        assert_eq!(function.parameters()[0].name(), "amount");
        assert_eq!(function.parameters()[0].get_type(), Type::Int);
        assert_eq!(
            function.parameters()[0].constraint().as_ref().unwrap(),
            &Expression::binary_op(
                Expression::identifier("amount".to_string()),
                ">".into(),
                Expression::Literal(Literal::Int(0))
            )
        );
        assert_eq!(function.statements().len(), 3);
    }

    #[test]
    fn contract_with_state_def_and_function_return() {
        let ast = parse(
            "contract Vault[Open, Full, Locked] {
                state var owner: address;
                state const balance: int = 0;
            }

            Vault@Open(0x123456789123456789) {
                function payable withdraw({amount: int | amount > 0}) -> {n: int | n == balance} {
                    var balance: int = balance - amount;
                    const n: int = balance;
                    return n;
                }
            }",
        )
        .unwrap();
        assert_eq!(ast.contracts().len(), 1);
        assert_eq!(ast.state_defs().len(), 1);
        assert_eq!(ast.state_defs()[0].contract(), "Vault");
        assert_eq!(ast.state_defs()[0].name(), "Open");
        assert_eq!(ast.state_defs()[0].convenor(), "0x123456789123456789");
        assert_eq!(ast.state_defs()[0].functions().len(), 1);
        let function = &ast.state_defs()[0].functions()[0];
        assert_eq!(function.visibility(), &Visibility::Function);
        assert_eq!(function.modifiers().len(), 1);
        assert_eq!(function.modifiers()[0], Modifier::Payable);
        assert_eq!(function.name(), "withdraw");
        assert_eq!(function.parameters().len(), 1);
        assert_eq!(function.parameters()[0].name(), "amount");
        assert_eq!(function.parameters()[0].get_type(), Type::Int);
        assert_eq!(
            function.parameters()[0].constraint().as_ref().unwrap(),
            &Expression::binary_op(
                Expression::identifier("amount".to_string()),
                ">".into(),
                Expression::Literal(Literal::Int(0))
            )
        );
        assert_eq!(
            function.return_type().as_ref().unwrap().get_type(),
            Type::Int
        );
        assert_eq!(function.statements().len(), 3);
    }

    #[test]
    fn contract_state_def_with_more_functions() {
        let ast = parse(
            "contract Vault[Open, Full, Locked] {
                state var owner: address;
                state const balance: int = 0;
            }

            Vault@Open(0x123456789123456789) {
                function payable withdraw({amount: int | amount > 0}) -> {n: int | n == balance} {
                    var balance: int = balance - amount;
                    const n: int = balance;
                    return n;
                }

                function stateful deposit({amount: int}) {
                    false;
                }
            }",
        )
        .unwrap();
        assert_eq!(ast.contracts().len(), 1);
        assert_eq!(ast.state_defs().len(), 1);
        assert_eq!(ast.state_defs()[0].name(), "Open");
        assert_eq!(ast.state_defs()[0].contract(), "Vault");
        assert_eq!(ast.state_defs()[0].convenor(), "0x123456789123456789");
        assert_eq!(ast.state_defs()[0].functions().len(), 2);

        let function = &ast.state_defs()[0].functions()[0];
        assert_eq!(function.visibility(), &Visibility::Function);
        assert_eq!(function.modifiers().len(), 1);
        assert_eq!(function.modifiers()[0], Modifier::Payable);
        assert_eq!(function.name(), "withdraw");
        assert_eq!(function.parameters().len(), 1);
        assert_eq!(function.parameters()[0].name(), "amount");
        assert_eq!(function.parameters()[0].get_type(), Type::Int);
        assert_eq!(
            function.parameters()[0].constraint().as_ref().unwrap(),
            &Expression::binary_op(
                Expression::identifier("amount".to_string()),
                ">".into(),
                Expression::Literal(Literal::Int(0))
            )
        );
        assert_eq!(
            function.return_type().as_ref().unwrap().get_type(),
            Type::Int
        );
        assert_eq!(function.statements().len(), 3);

        let function = &ast.state_defs()[0].functions()[1];
        assert_eq!(function.visibility(), &Visibility::Function);
        assert_eq!(function.modifiers().len(), 1);
        assert_eq!(function.modifiers()[0], Modifier::Stateful);
        assert_eq!(function.name(), "deposit");
        assert_eq!(function.parameters().len(), 1);
        assert_eq!(function.parameters()[0].name(), "amount");
        assert_eq!(function.parameters()[0].get_type(), Type::Int);
        assert!(function.parameters()[0].constraint().is_none());
        assert_eq!(function.statements().len(), 1);
    }

    #[test]
    fn contract_with_more_state_defs() {
        let ast = parse(
            "contract Vault[Open, Full, Locked] {
                state var owner: address;
                state const balance: int = 0;
            }

            Vault@Open(owner) {
                function payable withdraw({amount: int | amount > 0}) -> {n: int | n == balance} {
                    var balance: int = balance - amount;
                    const n: int = balance;
                    return n;
                }
            }

            Vault@Full(any) {
                entrypoint reduceOne() {
                    true;
                }
            }",
        )
        .unwrap();
        assert_eq!(ast.contracts().len(), 1);
        assert_eq!(ast.state_defs().len(), 2);
        assert_eq!(ast.state_defs()[0].name(), "Open");
        assert_eq!(ast.state_defs()[0].contract(), "Vault");
        assert_eq!(ast.state_defs()[0].convenor(), "owner");
        assert_eq!(ast.state_defs()[0].functions().len(), 1);

        let function = &ast.state_defs()[0].functions()[0];
        assert_eq!(function.visibility(), &Visibility::Function);
        assert_eq!(function.modifiers().len(), 1);
        assert_eq!(function.modifiers()[0], Modifier::Payable);
        assert_eq!(function.name(), "withdraw");
        assert_eq!(function.parameters().len(), 1);
        assert_eq!(function.parameters()[0].name(), "amount");
        assert_eq!(function.parameters()[0].get_type(), Type::Int);
        assert_eq!(
            function.parameters()[0].constraint().as_ref().unwrap(),
            &Expression::binary_op(
                Expression::identifier("amount".to_string()),
                ">".into(),
                Expression::Literal(Literal::Int(0))
            )
        );
        assert_eq!(
            function.return_type().as_ref().unwrap().get_type(),
            Type::Int
        );
        assert_eq!(function.statements().len(), 3);

        assert_eq!(ast.state_defs()[1].name(), "Full");
        assert_eq!(ast.state_defs()[1].convenor(), "any");
        assert_eq!(ast.state_defs()[1].functions().len(), 1);

        let function = &ast.state_defs()[1].functions()[0];
        assert_eq!(function.visibility(), &Visibility::Entrypoint);
        assert_eq!(function.modifiers().len(), 0);
        assert_eq!(function.name(), "reduceOne");
        assert_eq!(function.parameters().len(), 0);
        assert!(function.return_type().is_none());
        assert_eq!(function.statements().len(), 1);
    }
}
