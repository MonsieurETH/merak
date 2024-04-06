use std::{
    collections::{BTreeSet, HashMap, HashSet},
    error::Error,
    hash::Hash,
};

use crate::ast::{
    BinaryOperator, Block, Expression, Function, Literal, Program, Statement, UnaryOperator,
};

pub struct FunctionsCFG {
    functions: HashMap<String, ControlFlowGraph>,
}

impl FunctionsCFG {
    pub fn new() -> FunctionsCFG {
        FunctionsCFG {
            functions: HashMap::new(),
        }
    }

    pub fn parse_program(&mut self, ast: &Program) {
        if let Some(constructor) = ast.contracts().last().unwrap().constructor() {
            let cfg = ControlFlowGraph::build_cfg(constructor.statements().to_vec());
            self.functions.insert("constructor".to_string(), cfg);
        }

        for state in ast.state_defs() {
            for function in state.functions() {
                let cfg = ControlFlowGraph::build_cfg(function.statements().to_vec());
                self.functions.insert(function.name().to_string(), cfg);
            }
        }
    }

    pub fn get(&self, name: &str) -> Option<&ControlFlowGraph> {
        self.functions.get(name)
    }

    pub fn get_mut(&mut self, name: &str) -> Option<&mut ControlFlowGraph> {
        self.functions.get_mut(name)
    }
}

#[derive(Debug, PartialEq)]
pub struct ControlFlowGraph {
    nodes: Vec<ControlFlowNode>,
    dominators: Option<HashMap<usize, BTreeSet<usize>>>,
    last_node_id: usize,
}

impl ControlFlowGraph {
    pub fn new() -> ControlFlowGraph {
        ControlFlowGraph {
            nodes: Vec::new(),
            last_node_id: 0,
            dominators: None,
        }
    }

    pub fn add_node(&mut self, node: ControlFlowNode) {
        self.dominators = None;
        self.last_node_id = node.id;
        self.nodes.push(node);
    }

    pub fn with_nodes(mut self, nodes: Vec<ControlFlowNode>) -> ControlFlowGraph {
        for node in nodes {
            self.add_node(node);
        }
        self
    }

    pub fn nodes(&self) -> &Vec<ControlFlowNode> {
        &self.nodes
    }

    pub fn dominators(&self) -> &Option<HashMap<usize, BTreeSet<usize>>> {
        &self.dominators
    }

    pub fn defined_variables(&self) -> HashSet<String> {
        let mut defined = HashSet::new();
        for node in &self.nodes {
            for statement in node.statements() {
                match statement {
                    Statement::VarDeclaration(name, _, _) => {
                        defined.insert(name.clone());
                    }
                    Statement::VarAssignment(name, _) => {
                        defined.insert(name.clone());
                    }
                    Statement::ConstDeclaration(name, _, _) => {
                        defined.insert(name.clone());
                    }
                    _ => {}
                }
            }

            if let Some(expr) = node.condition() {
                let vars = expr.get_vars();
                defined.extend(vars);
            }
        }
        defined
    }

    pub fn build_cfg(statements: Vec<Statement>) -> ControlFlowGraph {
        let mut cfg = ControlFlowGraph::new();
        for statement in statements {
            match statement {
                Statement::If(condition, true_branch, else_branch) => {
                    let if_cfg = cfg.parse_if(condition, true_branch, else_branch);
                    cfg.extend_and_connect_graph(if_cfg)
                }
                Statement::While(condition, inner_statements) => {
                    let while_cfg = cfg.parse_while(condition, inner_statements);
                    cfg.extend_and_connect_graph(while_cfg)
                }
                _ => {
                    if cfg.nodes.is_empty() {
                        let mut node =
                            ControlFlowNode::new(cfg.last_node_id, NodeType::Normal, None);
                        node.add_statement(statement);
                        cfg.add_node(node);
                    } else {
                        let index = cfg.nodes.len() - 1;
                        cfg.nodes[index].add_statement(statement);
                    }
                }
            }
        }

        cfg
    }

    fn parse_if(
        &mut self,
        condition: Expression,
        true_branch: Vec<Statement>,
        false_branch: Option<Vec<Statement>>,
    ) -> ControlFlowGraph {
        let mut cfg = ControlFlowGraph::new();

        let condition_node = ControlFlowNode::new(
            cfg.last_node_id,
            NodeType::Condition,
            Some(condition.clone()),
        );

        let condition_id = condition_node.id;
        cfg.add_node(condition_node);
        let condition_index = 0; //cfg.nodes.len() - 1;

        let true_graph = ControlFlowGraph::build_cfg(true_branch);
        let true_first_id = 1; //true_graph.nodes.first().unwrap().id;
        let true_first_index = 1;
        cfg.extend_graph(true_graph);

        let true_last_index = cfg.nodes.len() - 1;
        let true_last_id = cfg.nodes[true_last_index].id;

        let (false_first_index, false_last_index) = if let Some(fbranch) = false_branch {
            let false_graph = ControlFlowGraph::build_cfg(fbranch);
            let false_graph_len = false_graph.nodes.len();
            cfg.extend_graph(false_graph);
            let false_first_index = cfg.nodes.len() - false_graph_len;
            let false_last_index = cfg.nodes.len() - 1;
            (false_first_index, false_last_index)
        } else {
            (0, 0)
        };

        let next_node = ControlFlowNode::new(cfg.last_node_id + 1, NodeType::Normal, None);
        let next_node_id = next_node.id;
        cfg.add_node(next_node);
        let next_node_index = cfg.nodes.len() - 1;

        cfg.nodes[condition_index].add_to_edge(true_first_id);
        cfg.nodes[true_first_index].add_from_edge(condition_index);

        cfg.nodes[true_last_index].add_to_edge(next_node_id);
        cfg.nodes[next_node_index].add_from_edge(true_last_id);

        if false_first_index != 0 || false_last_index != 0 {
            let false_first_id = cfg.nodes[false_first_index].id;
            let false_last_id = cfg.nodes[false_last_index].id;

            cfg.nodes[condition_index].add_to_edge(false_first_id);
            cfg.nodes[false_first_index].add_from_edge(condition_id);

            cfg.nodes[false_last_index].add_to_edge(next_node_id);
            cfg.nodes[next_node_index].add_from_edge(false_last_id);
        } else {
            cfg.nodes[condition_index].add_to_edge(next_node_id);
            cfg.nodes[next_node_index].add_from_edge(condition_id);
        }

        cfg
    }

    fn parse_while(
        &mut self,
        condition: Expression,
        statements: Vec<Statement>,
    ) -> ControlFlowGraph {
        let mut cfg = ControlFlowGraph::new();

        let condition_node_id = 0; // always 0
        let condition_node = ControlFlowNode::new(0, NodeType::Condition, Some(condition.clone()));

        cfg.add_node(condition_node);
        let condition_index = 0; //cfg.nodes.len() - 1;

        let while_graph = ControlFlowGraph::build_cfg(statements);
        let while_first_id = 1;
        let while_first_index = 1; //aca
        cfg.extend_graph(while_graph);

        let while_last_index = cfg.nodes.len() - 1;
        let while_last_id = cfg.nodes[while_last_index].id;

        let next_node = ControlFlowNode::new(cfg.last_node_id + 1, NodeType::Normal, None);
        let next_node_id = next_node.id;
        cfg.add_node(next_node);
        let next_node_index = cfg.nodes.len() - 1;

        cfg.nodes[condition_index].add_to_edge(while_first_id);
        cfg.nodes[while_first_index].add_from_edge(condition_node_id);

        cfg.nodes[condition_index].add_to_edge(next_node_id);
        cfg.nodes[next_node_index].add_from_edge(condition_node_id);

        cfg.nodes[while_last_index].add_to_edge(condition_node_id);
        cfg.nodes[condition_index].add_from_edge(while_last_id);

        cfg
    }

    fn extend_and_connect_graph(&mut self, graph: ControlFlowGraph) {
        let original_size = self.nodes.len();
        self.extend_graph(graph);

        if original_size == 0 {
            return;
        }
        let graph_first_id = self.nodes[original_size].id();
        self.nodes[original_size - 1]
            .to_edges
            .insert(graph_first_id);

        let original_id = self.nodes[original_size - 1].id();
        self.nodes[original_size].from_edges.insert(original_id);
    }

    fn extend_graph(&mut self, graph: ControlFlowGraph) {
        let mut extra_size_id = 0;
        if !self.nodes.is_empty() {
            extra_size_id = self.last_node_id + 1;
        }
        for node in graph.nodes {
            let mut new_node = node;
            new_node.id += extra_size_id;
            let to_edges = new_node.to_edges.iter().map(|x| x + extra_size_id);
            let from_edges = new_node.from_edges.iter().map(|x| x + extra_size_id);
            new_node.to_edges = to_edges.collect();
            new_node.from_edges = from_edges.collect();
            self.add_node(new_node);
        }
    }

    pub fn calculate_dominators(&mut self) {
        match self.dominators {
            Some(_) => return,
            None => {}
        }
        let mut dominators = HashMap::new();

        let all_nodes: BTreeSet<usize> = self.nodes.iter().map(|n| n.id).collect();
        for node in &self.nodes {
            dominators.insert(
                node.id,
                if node.id == 0 {
                    BTreeSet::from([0])
                } else {
                    all_nodes.clone()
                },
            );
        }

        let mut changed = true;
        while changed {
            changed = false;
            for node in &self.nodes {
                if node.id == 0 {
                    continue;
                }

                let pred_dominators: Vec<BTreeSet<usize>> = node
                    .from_edges
                    .iter()
                    .filter_map(|&id| dominators.get(&id).cloned())
                    .collect();

                let intersection: BTreeSet<usize> = if !pred_dominators.is_empty() {
                    pred_dominators
                        .iter()
                        .skip(1)
                        .fold(pred_dominators[0].clone(), |acc, d| &acc & d)
                } else {
                    BTreeSet::new()
                };

                let mut new_dom = intersection;
                new_dom.insert(node.id);

                if new_dom != *dominators.get(&node.id).unwrap() {
                    dominators.insert(node.id, new_dom);
                    changed = true;
                }
            }
        }

        self.dominators = Some(dominators.clone());
    }

    /*fn immediate_dominator(&self, node_id: usize) -> Option<usize> {
        self.dominators
            .as_ref()?
            .get(&node_id)
            .and_then(|dominators_set| {
                let mut dominators_iter = dominators_set.iter();
                let last = dominators_iter.next_back()?;
                if node_id == *last {
                    dominators_iter.next_back().copied()
                } else {
                    None
                }
            })
    }*/
    pub fn dominance_frontiers(&self) -> HashMap<usize, BTreeSet<usize>> {
        let mut frontiers = HashMap::new();

        for node in &self.nodes {
            let node_id = node.id;
            let mut node_frontier = BTreeSet::new();

            for &succ in &node.to_edges {
                let succ_predecessors = &self.nodes[succ].from_edges;

                let dominates_predecessor = succ_predecessors
                    .iter()
                    .any(|&pred| self.dominates(node_id, pred));
                let dominates_successor = self.dominates(node_id, succ);

                if dominates_predecessor && !dominates_successor {
                    node_frontier.insert(succ);
                }
            }

            frontiers.insert(node_id, node_frontier);
        }

        frontiers
    }

    fn dominates(&self, dominator: usize, dominated: usize) -> bool {
        self.dominators
            .as_ref()
            .unwrap()
            .get(&dominated)
            .map_or(false, |doms| doms.contains(&dominator))
    }
}

#[derive(Debug, PartialEq)]
pub struct ControlFlowNode {
    id: usize,
    statements: Vec<Statement>,
    condition: Option<Expression>,
    node_type: NodeType,
    to_edges: BTreeSet<usize>,
    from_edges: BTreeSet<usize>,
}

#[derive(Debug, PartialEq, Clone)]
pub enum NodeType {
    Condition,
    Normal,
}

impl ControlFlowNode {
    pub fn new(id: usize, node_type: NodeType, condition: Option<Expression>) -> ControlFlowNode {
        ControlFlowNode {
            id,
            statements: Vec::new(),
            node_type,
            condition,
            to_edges: BTreeSet::new(),
            from_edges: BTreeSet::new(),
        }
    }

    pub fn add_statement(&mut self, statement: Statement) {
        self.statements.push(statement);
    }

    pub fn remove_statement(&mut self, index: usize) {
        self.statements.remove(index);
    }

    pub fn add_to_edge(&mut self, edge: usize) {
        self.to_edges.insert(edge);
    }

    pub fn add_from_edge(&mut self, edge: usize) {
        self.from_edges.insert(edge);
    }

    pub fn id(&self) -> usize {
        self.id
    }

    pub fn statements(&self) -> &Vec<Statement> {
        &self.statements
    }

    pub fn condition(&self) -> &Option<Expression> {
        &self.condition
    }

    pub fn to_edges(&self) -> &BTreeSet<usize> {
        &self.to_edges
    }

    pub fn from_edges(&self) -> &BTreeSet<usize> {
        &self.from_edges
    }

    pub fn node_type(&self) -> &NodeType {
        &self.node_type
    }

    pub fn defines_variable(&self, var: &str) -> bool {
        for statement in &self.statements {
            match statement {
                Statement::VarDeclaration(name, _, _) => {
                    if name == var {
                        return true;
                    }
                }
                Statement::VarAssignment(name, _) => {
                    if name == var {
                        return true;
                    }
                }
                Statement::ConstDeclaration(name, _, _) => {
                    if name == var {
                        return true;
                    }
                }
                _ => {}
            }
        }
        false
    }

    pub fn local_value_numbering(&mut self) {
        let mut table: HashMap<NumberingEntry, String> = HashMap::new();
        let mut num2expr: Vec<Expression> = Vec::new();
        let mut var2num: HashMap<String, usize> = HashMap::new();
        let mut num2var: HashMap<usize, String> = HashMap::new();
        let mut new_statements = Vec::new();

        for statement in self.statements() {
            match statement {
                Statement::VarDeclaration(name, ty, Some(expr)) => {
                    // Id operation (in var declaration)
                    if let Expression::Identifier(var) = expr {
                        //TODO is_identifier (inside subs, (((()))), etc)
                        if var2num.contains_key(var) {
                            let old_value = var2num.get(var).unwrap();
                            let new_expr = num2expr.get(*old_value).unwrap();
                            if !num2var.contains_key(old_value) {
                                num2var.insert(old_value.clone(), name.clone());
                            }
                            var2num.insert(name.to_string(), old_value.clone());

                            new_statements.push(Statement::VarDeclaration(
                                name.clone(),
                                ty.clone(),
                                Some(new_expr.clone()),
                            ));
                        } else {
                            //var is paramenter and never used before
                            let value = num2expr.len();
                            num2expr.push(expr.clone());
                            var2num.insert(name.clone(), value);
                            if !num2var.contains_key(&value) {
                                num2var.insert(value.clone(), var.clone());
                            }
                            new_statements.push(Statement::VarDeclaration(
                                name.clone(),
                                ty.clone(),
                                Some(Expression::Identifier(var.clone())),
                            ));
                        }
                        continue;
                    }

                    let num_expr = exp2numbering(&expr, &mut var2num);

                    if table.contains_key(&num_expr) {
                        let inserted_name = table.get(&num_expr).unwrap();
                        let old_value = var2num.get(inserted_name).unwrap();
                        if !num2var.contains_key(old_value) {
                            num2var.insert(old_value.clone(), name.clone());
                        }
                        var2num.insert(name.clone(), old_value.clone());
                        new_statements.push(Statement::VarDeclaration(
                            name.clone(),
                            ty.clone(),
                            Some(Expression::Identifier(inserted_name.clone())),
                        ));
                    } else {
                        num2expr.push(expr.clone());
                        table.insert(num_expr.clone(), name.clone());
                        let value = table.len() - 1;
                        var2num.insert(name.clone(), value);
                        if !num2var.contains_key(&value) {
                            num2var.insert(value, name.clone());
                        }
                        let expr_nums = numbering2exp(&num_expr, &num2var);
                        match expr_nums {
                            Ok(exprn) => new_statements.push(Statement::VarDeclaration(
                                name.clone(),
                                ty.clone(),
                                Some(exprn),
                            )),
                            Err(_) => new_statements.push(statement.clone()),
                        }
                    }
                }
                Statement::ConstDeclaration(name, ty, expr) => {
                    num2expr.push(expr.clone());
                    let num_expr = exp2numbering(&expr, &mut var2num);
                    if table.contains_key(&num_expr) {
                        let inserted_name = table.get(&num_expr).unwrap();
                        let old_value = var2num.get(inserted_name).unwrap();
                        if !num2var.contains_key(old_value) {
                            num2var.insert(old_value.clone(), name.clone());
                        }
                        var2num.insert(name.clone(), old_value.clone());

                        new_statements.push(Statement::ConstDeclaration(
                            name.clone(),
                            ty.clone(),
                            Expression::Identifier(inserted_name.clone()),
                        ));
                    } else {
                        num2expr.push(expr.clone());
                        table.insert(num_expr.clone(), name.clone());
                        let value = table.len() - 1;
                        var2num.insert(name.clone(), value);
                        if !num2var.contains_key(&value) {
                            num2var.insert(value, name.clone());
                        }
                        let expr_nums = numbering2exp(&num_expr, &num2var);
                        match expr_nums {
                            Ok(exprn) => new_statements.push(Statement::ConstDeclaration(
                                name.clone(),
                                ty.clone(),
                                exprn,
                            )),
                            Err(_) => new_statements.push(statement.clone()),
                        }
                    }
                }
                Statement::Return(Some(expr)) => {
                    let num_expr = exp2numbering(&expr, &mut var2num);
                    let expr_nums = numbering2exp(&num_expr, &num2var);
                    match expr_nums {
                        Ok(exprn) => new_statements.push(Statement::Return(Some(exprn))),
                        Err(_) => new_statements.push(statement.clone()),
                    }
                }
                _ => new_statements.push(statement.clone()),
            }
        }
        self.statements = new_statements;
        self.dead_code_elimination();
    }

    fn dead_code_elimination(&mut self) {
        let mut used = HashSet::new();
        let mut new_statements = Vec::new();

        for statement in self.statements() {
            used.extend(statement.used_vars());
        }

        for statement in self.statements() {
            match statement {
                Statement::VarDeclaration(name, ty, Some(expr)) => {
                    if used.contains(name) {
                        new_statements.push(Statement::VarDeclaration(
                            name.clone(),
                            ty.clone(),
                            Some(expr.clone()),
                        ));
                    }
                }
                Statement::VarAssignment(name, expr) => {
                    if used.contains(name) {
                        new_statements.push(Statement::VarAssignment(name.clone(), expr.clone()));
                    }
                }
                Statement::ConstDeclaration(name, ty, expr) => {
                    if used.contains(name) {
                        new_statements.push(Statement::ConstDeclaration(
                            name.clone(),
                            ty.clone(),
                            expr.clone(),
                        ));
                    }
                }
                Statement::Return(Some(expr)) => {
                    new_statements.push(Statement::Return(Some(expr.clone())));
                }
                _ => new_statements.push(statement.clone()),
            }
        }

        self.statements = new_statements;
    }
}

#[derive(Debug, PartialEq, Eq, Hash, Clone)]
pub enum NumberingEntry {
    Value(Literal),
    Var(usize),
    BinaryOp(BinaryOperator, Box<NumberingEntry>, Box<NumberingEntry>),
    UnaryOp(UnaryOperator, Box<NumberingEntry>),
}

fn exp2numbering(expr: &Expression, var2num: &mut HashMap<String, usize>) -> NumberingEntry {
    fn ordered_exp2numbering(
        expr: &Expression,
        var2num: &mut HashMap<String, usize>,
    ) -> (NumberingEntry, usize) {
        match expr {
            Expression::BinaryOp(left, op, right) => {
                let (left_expr, left_num) = ordered_exp2numbering(left, var2num);
                let (right_expr, right_num) = ordered_exp2numbering(right, var2num);

                if left_num <= right_num {
                    (
                        NumberingEntry::BinaryOp(
                            op.clone(),
                            Box::new(left_expr),
                            Box::new(right_expr),
                        ),
                        left_num,
                    )
                } else {
                    (
                        NumberingEntry::BinaryOp(
                            op.clone(),
                            Box::new(right_expr),
                            Box::new(left_expr),
                        ),
                        right_num,
                    )
                }
            }
            Expression::Identifier(name) => {
                if var2num.contains_key(name) {
                    (
                        NumberingEntry::Var(*var2num.get(name).unwrap()),
                        *var2num.get(name).unwrap(),
                    )
                } else {
                    let num = var2num.values().max().unwrap_or(&0) + 1;
                    var2num.insert(name.clone(), num);
                    (NumberingEntry::Var(num), num)
                }
            }
            Expression::Literal(literal) => match literal {
                Literal::Bool(_) => (NumberingEntry::Value(literal.clone()), 0),
                Literal::Int(_) => (NumberingEntry::Value(literal.clone()), 0),
                Literal::Address(_) => (NumberingEntry::Value(literal.clone()), 0),
                Literal::String(_) => (NumberingEntry::Value(literal.clone()), 0),
            },
            Expression::UnaryOp(op, expr) => {
                let (expr, num) = ordered_exp2numbering(expr, var2num);
                (NumberingEntry::UnaryOp(op.clone(), Box::new(expr)), num)
            }
        }
    }

    let (numbering_entry, _) = ordered_exp2numbering(expr, var2num);
    numbering_entry
}

fn numbering2exp(
    entry: &NumberingEntry,
    num2var: &HashMap<usize, String>,
) -> Result<Expression, String> {
    match entry {
        NumberingEntry::Value(literal) => Ok(Expression::Literal(literal.clone())),
        NumberingEntry::Var(num) => {
            if num2var.contains_key(num) {
                Ok(Expression::Identifier(num2var.get(num).unwrap().clone()))
            } else {
                Err("Variable not found".to_string())
            }
        }
        NumberingEntry::BinaryOp(op, left, right) => Ok(Expression::BinaryOp(
            Box::new(numbering2exp(left, num2var)?),
            op.clone(),
            Box::new(numbering2exp(right, num2var)?),
        )),
        NumberingEntry::UnaryOp(op, expr) => Ok(Expression::UnaryOp(
            op.clone(),
            Box::new(numbering2exp(expr, num2var)?),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{BinaryOperator, Literal, Type};
    use crate::parser::parse;
    use crate::type_checking::TypeChecker;

    #[test]
    fn test_if() {
        let ast = parse(
            "
            contract Vault[Open, Full, Locked] {

            }

            Vault@Open(owner) {
                function stateful payable withdraw() {
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

        let mut cfg = FunctionsCFG::new();
        cfg.parse_program(&ast);

        let function_cfg: &ControlFlowGraph = cfg.get("withdraw").unwrap();

        //println!("{:?}", function_cfg);

        let expected = ControlFlowGraph {
            nodes: vec![
                ControlFlowNode {
                    id: 0,
                    statements: vec![Statement::VarDeclaration(
                        "ret".to_string(),
                        Type::Bool,
                        None,
                    )],
                    condition: None,
                    node_type: NodeType::Normal,
                    to_edges: BTreeSet::from([1]),
                    from_edges: BTreeSet::new(),
                },
                ControlFlowNode {
                    id: 1,
                    statements: vec![],
                    condition: Some(Expression::BinaryOp(
                        Box::new(Expression::Identifier("amount".to_string())),
                        BinaryOperator::Gt,
                        Box::new(Expression::Literal(Literal::Int(0))),
                    )),
                    node_type: NodeType::Condition,
                    to_edges: BTreeSet::from([2, 3]),
                    from_edges: BTreeSet::from([0]),
                },
                ControlFlowNode {
                    id: 2,
                    statements: vec![Statement::VarAssignment(
                        "ret".to_string(),
                        Expression::Literal(Literal::Bool(false)),
                    )],
                    condition: None,
                    node_type: NodeType::Normal,
                    to_edges: BTreeSet::from([4]),
                    from_edges: BTreeSet::from([1]),
                },
                ControlFlowNode {
                    id: 3,
                    statements: vec![Statement::VarAssignment(
                        "ret".to_string(),
                        Expression::Literal(Literal::Bool(true)),
                    )],
                    condition: None,
                    node_type: NodeType::Normal,
                    to_edges: BTreeSet::from([4]),
                    from_edges: BTreeSet::from([1]),
                },
                ControlFlowNode {
                    id: 4,
                    statements: vec![Statement::Return(Some(Expression::Identifier(
                        "ret".to_string(),
                    )))],
                    condition: None,
                    node_type: NodeType::Normal,
                    to_edges: BTreeSet::new(),
                    from_edges: BTreeSet::from([2, 3]),
                },
            ],
            last_node_id: 4,
            dominators: None,
        };

        assert_eq!(function_cfg, &expected);
    }

    #[test]
    fn test_while() {
        let ast = parse(
            "
            contract Vault[Open, Full, Locked] {

            }

            Vault@Open(owner) {
                function stateful payable withdraw() {
                    var ret: bool;
                    while (amount > 0) {
                        ret = false;
                    }
                    return ret;
                }
            }",
        )
        .unwrap();

        let mut tc = TypeChecker::new();
        let _env = tc.check_program(&ast);

        let mut cfg = FunctionsCFG::new();
        cfg.parse_program(&ast);

        let function_cfg: &ControlFlowGraph = cfg.get("withdraw").unwrap();

        let expected = ControlFlowGraph {
            nodes: vec![
                ControlFlowNode {
                    id: 0,
                    statements: vec![Statement::VarDeclaration(
                        "ret".to_string(),
                        Type::Bool,
                        None,
                    )],
                    condition: None,
                    node_type: NodeType::Normal,
                    to_edges: BTreeSet::from([1]),
                    from_edges: BTreeSet::new(),
                },
                ControlFlowNode {
                    id: 1,
                    statements: vec![],
                    condition: Some(Expression::BinaryOp(
                        Box::new(Expression::Identifier("amount".to_string())),
                        BinaryOperator::Gt,
                        Box::new(Expression::Literal(Literal::Int(0))),
                    )),
                    node_type: NodeType::Condition,
                    to_edges: BTreeSet::from([2, 3]),
                    from_edges: BTreeSet::from([0, 2]),
                },
                ControlFlowNode {
                    id: 2,
                    statements: vec![Statement::VarAssignment(
                        "ret".to_string(),
                        Expression::Literal(Literal::Bool(false)),
                    )],
                    condition: None,
                    node_type: NodeType::Normal,
                    to_edges: BTreeSet::from([1]),
                    from_edges: BTreeSet::from([1]),
                },
                ControlFlowNode {
                    id: 3,
                    statements: vec![Statement::Return(Some(Expression::Identifier(
                        "ret".to_string(),
                    )))],
                    condition: None,
                    node_type: NodeType::Normal,
                    to_edges: BTreeSet::new(),
                    from_edges: BTreeSet::from([1]),
                },
            ],
            last_node_id: 3,
            dominators: None,
        };

        assert_eq!(function_cfg, &expected);
    }

    #[test]
    fn test_constructor() {
        let ast = parse(
            "
            contract Vault[Open, Full, Locked] {
                constructor() {
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

        let mut cfg = FunctionsCFG::new();
        cfg.parse_program(&ast);

        let function_cfg: &ControlFlowGraph = cfg.get("constructor").unwrap();

        let expected = ControlFlowGraph {
            nodes: vec![
                ControlFlowNode {
                    id: 0,
                    statements: vec![Statement::VarDeclaration(
                        "ret".to_string(),
                        Type::Bool,
                        None,
                    )],
                    condition: None,
                    node_type: NodeType::Normal,
                    to_edges: BTreeSet::from([1]),
                    from_edges: BTreeSet::new(),
                },
                ControlFlowNode {
                    id: 1,
                    statements: vec![],
                    condition: Some(Expression::BinaryOp(
                        Box::new(Expression::Identifier("amount".to_string())),
                        BinaryOperator::Gt,
                        Box::new(Expression::Literal(Literal::Int(0))),
                    )),
                    node_type: NodeType::Condition,
                    to_edges: BTreeSet::from([2, 3]),
                    from_edges: BTreeSet::from([0]),
                },
                ControlFlowNode {
                    id: 2,
                    statements: vec![Statement::VarAssignment(
                        "ret".to_string(),
                        Expression::Literal(Literal::Bool(false)),
                    )],
                    condition: None,
                    node_type: NodeType::Normal,
                    to_edges: BTreeSet::from([4]),
                    from_edges: BTreeSet::from([1]),
                },
                ControlFlowNode {
                    id: 3,
                    statements: vec![Statement::VarAssignment(
                        "ret".to_string(),
                        Expression::Literal(Literal::Bool(true)),
                    )],
                    condition: None,
                    node_type: NodeType::Normal,
                    to_edges: BTreeSet::from([4]),
                    from_edges: BTreeSet::from([1]),
                },
                ControlFlowNode {
                    id: 4,
                    statements: vec![Statement::Return(Some(Expression::Identifier(
                        "ret".to_string(),
                    )))],
                    condition: None,
                    node_type: NodeType::Normal,
                    to_edges: BTreeSet::new(),
                    from_edges: BTreeSet::from([2, 3]),
                },
            ],
            last_node_id: 4,
            dominators: None,
        };

        assert_eq!(function_cfg, &expected);
    }

    #[test]
    fn test_if_inside_while() {
        let ast = parse(
            "
            contract Vault[Open, Full, Locked] {
                constructor() {
                    var ret: bool;
                    while (amount > 2) {
                        if (amount > 0) {
                            ret = false;
                        } else {
                            ret = true;
                        }
                    }
                    return ret;
                }
            }",
        )
        .unwrap();

        let mut tc = TypeChecker::new();
        let _env = tc.check_program(&ast);

        let mut cfg = FunctionsCFG::new();
        cfg.parse_program(&ast);

        let function_cfg: &ControlFlowGraph = cfg.get("constructor").unwrap();
        let expected = ControlFlowGraph {
            nodes: vec![
                ControlFlowNode {
                    id: 0,
                    statements: vec![Statement::VarDeclaration(
                        "ret".to_string(),
                        Type::Bool,
                        None,
                    )],
                    condition: None,
                    node_type: NodeType::Normal,
                    to_edges: BTreeSet::from([1]),
                    from_edges: BTreeSet::new(),
                },
                ControlFlowNode {
                    id: 1,
                    statements: vec![],
                    condition: Some(Expression::BinaryOp(
                        Box::new(Expression::Identifier("amount".to_string())),
                        BinaryOperator::Gt,
                        Box::new(Expression::Literal(Literal::Int(2))),
                    )),
                    node_type: NodeType::Condition,
                    to_edges: BTreeSet::from([2, 6]),
                    from_edges: BTreeSet::from([0, 5]),
                },
                ControlFlowNode {
                    id: 2,
                    statements: vec![],
                    condition: Some(Expression::BinaryOp(
                        Box::new(Expression::Identifier("amount".to_string())),
                        BinaryOperator::Gt,
                        Box::new(Expression::Literal(Literal::Int(0))),
                    )),
                    node_type: NodeType::Condition,
                    to_edges: BTreeSet::from([3, 4]),
                    from_edges: BTreeSet::from([1]), // falta este!
                },
                ControlFlowNode {
                    id: 3,
                    statements: vec![Statement::VarAssignment(
                        "ret".to_string(),
                        Expression::Literal(Literal::Bool(false)),
                    )],
                    condition: None,
                    node_type: NodeType::Normal,
                    to_edges: BTreeSet::from([5]),
                    from_edges: BTreeSet::from([2]),
                },
                ControlFlowNode {
                    id: 4,
                    statements: vec![Statement::VarAssignment(
                        "ret".to_string(),
                        Expression::Literal(Literal::Bool(true)),
                    )],
                    condition: None,
                    node_type: NodeType::Normal,
                    to_edges: BTreeSet::from([5]),
                    from_edges: BTreeSet::from([2]),
                },
                ControlFlowNode {
                    id: 5,
                    statements: vec![],
                    condition: None,
                    node_type: NodeType::Normal,
                    to_edges: BTreeSet::from([1]),
                    from_edges: BTreeSet::from([3, 4]),
                },
                ControlFlowNode {
                    id: 6,
                    statements: vec![Statement::Return(Some(Expression::Identifier(
                        "ret".to_string(),
                    )))],
                    condition: None,
                    node_type: NodeType::Normal,
                    to_edges: BTreeSet::new(),
                    from_edges: BTreeSet::from([1]),
                },
            ],
            last_node_id: 6,
            dominators: None,
        };

        assert_eq!(function_cfg, &expected);
    }

    #[test]
    fn test_while_inside_if() {
        let ast = parse(
            "
            contract Vault[Open, Full, Locked] {
                constructor() {
                    var ret: bool;
                    if (amount > 0) {
                        while (amount > 2) {
                            ret = false;
                        }
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

        let mut cfg = FunctionsCFG::new();
        cfg.parse_program(&ast);

        let function_cfg: &ControlFlowGraph = cfg.get("constructor").unwrap();

        let expected = ControlFlowGraph {
            nodes: vec![
                ControlFlowNode {
                    id: 0,
                    statements: vec![Statement::VarDeclaration(
                        "ret".to_string(),
                        Type::Bool,
                        None,
                    )],
                    condition: None,
                    node_type: NodeType::Normal,
                    to_edges: BTreeSet::from([1]),
                    from_edges: BTreeSet::new(),
                },
                ControlFlowNode {
                    id: 1,
                    statements: vec![],
                    condition: Some(Expression::BinaryOp(
                        Box::new(Expression::Identifier("amount".to_string())),
                        BinaryOperator::Gt,
                        Box::new(Expression::Literal(Literal::Int(0))),
                    )),
                    node_type: NodeType::Condition,
                    to_edges: BTreeSet::from([2, 5]),
                    from_edges: BTreeSet::from([0]),
                },
                ControlFlowNode {
                    id: 2,
                    statements: vec![],
                    condition: Some(Expression::BinaryOp(
                        Box::new(Expression::Identifier("amount".to_string())),
                        BinaryOperator::Gt,
                        Box::new(Expression::Literal(Literal::Int(2))),
                    )),
                    node_type: NodeType::Condition,
                    to_edges: BTreeSet::from([3, 4]),
                    from_edges: BTreeSet::from([1, 3]),
                },
                ControlFlowNode {
                    id: 3,
                    statements: vec![Statement::VarAssignment(
                        "ret".to_string(),
                        Expression::Literal(Literal::Bool(false)),
                    )],
                    condition: None,
                    node_type: NodeType::Normal,
                    to_edges: BTreeSet::from([2]),
                    from_edges: BTreeSet::from([2]),
                },
                ControlFlowNode {
                    id: 4,
                    statements: vec![],
                    condition: None,
                    node_type: NodeType::Normal,
                    to_edges: BTreeSet::from([6]),
                    from_edges: BTreeSet::from([2]),
                },
                ControlFlowNode {
                    id: 5,
                    statements: vec![Statement::VarAssignment(
                        "ret".to_string(),
                        Expression::Literal(Literal::Bool(true)),
                    )],
                    condition: None,
                    node_type: NodeType::Normal,
                    to_edges: BTreeSet::from([6]),
                    from_edges: BTreeSet::from([1]),
                },
                ControlFlowNode {
                    id: 6,
                    statements: vec![Statement::Return(Some(Expression::Identifier(
                        "ret".to_string(),
                    )))],
                    condition: None,
                    node_type: NodeType::Normal,
                    to_edges: BTreeSet::new(),
                    from_edges: BTreeSet::from([4, 5]),
                },
            ],
            last_node_id: 6,
            dominators: None,
        };

        assert_eq!(function_cfg, &expected);
    }

    #[test]
    fn test_while_inside_while() {
        let ast = parse(
            "
            contract Vault[Open, Full, Locked] {
                constructor() {
                    var ret: bool;
                    while (amount > 0) {
                        while (amount > 2) {
                            ret = false;
                        }
                    }
                    return ret;
                }
            }",
        )
        .unwrap();

        let mut tc = TypeChecker::new();
        let _env = tc.check_program(&ast);

        let mut cfg = FunctionsCFG::new();
        cfg.parse_program(&ast);

        let function_cfg: &ControlFlowGraph = cfg.get("constructor").unwrap();

        let expected = ControlFlowGraph {
            nodes: vec![
                ControlFlowNode {
                    id: 0,
                    statements: vec![Statement::VarDeclaration(
                        "ret".to_string(),
                        Type::Bool,
                        None,
                    )],
                    condition: None,
                    node_type: NodeType::Normal,
                    to_edges: BTreeSet::from([1]),
                    from_edges: BTreeSet::new(),
                },
                ControlFlowNode {
                    id: 1,
                    statements: vec![],
                    condition: Some(Expression::BinaryOp(
                        Box::new(Expression::Identifier("amount".to_string())),
                        BinaryOperator::Gt,
                        Box::new(Expression::Literal(Literal::Int(0))),
                    )),
                    node_type: NodeType::Condition,
                    to_edges: BTreeSet::from([2, 5]),
                    from_edges: BTreeSet::from([0, 4]),
                },
                ControlFlowNode {
                    id: 2,
                    statements: vec![],
                    condition: Some(Expression::BinaryOp(
                        Box::new(Expression::Identifier("amount".to_string())),
                        BinaryOperator::Gt,
                        Box::new(Expression::Literal(Literal::Int(2))),
                    )),
                    node_type: NodeType::Condition,
                    to_edges: BTreeSet::from([3, 4]),
                    from_edges: BTreeSet::from([1, 3]),
                },
                ControlFlowNode {
                    id: 3,
                    statements: vec![Statement::VarAssignment(
                        "ret".to_string(),
                        Expression::Literal(Literal::Bool(false)),
                    )],
                    condition: None,
                    node_type: NodeType::Normal,
                    to_edges: BTreeSet::from([2]),
                    from_edges: BTreeSet::from([2]),
                },
                ControlFlowNode {
                    id: 4,
                    statements: vec![],
                    condition: None,
                    node_type: NodeType::Normal,
                    to_edges: BTreeSet::from([1]),
                    from_edges: BTreeSet::from([2]),
                },
                ControlFlowNode {
                    id: 5,
                    statements: vec![Statement::Return(Some(Expression::Identifier(
                        "ret".to_string(),
                    )))],
                    condition: None,
                    node_type: NodeType::Normal,
                    to_edges: BTreeSet::new(),
                    from_edges: BTreeSet::from([1]),
                },
            ],
            last_node_id: 5,
            dominators: None,
        };

        assert_eq!(function_cfg, &expected);
    }

    #[test]
    fn test_if_inside_if() {
        let ast = parse(
            "
            contract Vault[Open, Full, Locked] {
                constructor() {
                    var ret: bool;
                    if (amount > 0) {
                        if (amount > 0) {
                            ret = false;
                        } else {
                            ret = true;
                        }
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

        let mut cfg = FunctionsCFG::new();
        cfg.parse_program(&ast);

        let function_cfg: &ControlFlowGraph = cfg.get("constructor").unwrap();

        let expected = ControlFlowGraph {
            nodes: vec![
                ControlFlowNode {
                    id: 0,
                    statements: vec![Statement::VarDeclaration(
                        "ret".to_string(),
                        Type::Bool,
                        None,
                    )],
                    condition: None,
                    node_type: NodeType::Normal,
                    to_edges: BTreeSet::from([1]),
                    from_edges: BTreeSet::new(),
                },
                ControlFlowNode {
                    id: 1,
                    statements: vec![],
                    condition: Some(Expression::BinaryOp(
                        Box::new(Expression::Identifier("amount".to_string())),
                        BinaryOperator::Gt,
                        Box::new(Expression::Literal(Literal::Int(0))),
                    )),
                    node_type: NodeType::Condition,
                    to_edges: BTreeSet::from([2, 6]),
                    from_edges: BTreeSet::from([0]),
                },
                ControlFlowNode {
                    id: 2,
                    statements: vec![],
                    condition: Some(Expression::BinaryOp(
                        Box::new(Expression::Identifier("amount".to_string())),
                        BinaryOperator::Gt,
                        Box::new(Expression::Literal(Literal::Int(0))),
                    )),
                    node_type: NodeType::Condition,
                    to_edges: BTreeSet::from([3, 4]),
                    from_edges: BTreeSet::from([1]),
                },
                ControlFlowNode {
                    id: 3,
                    statements: vec![Statement::VarAssignment(
                        "ret".to_string(),
                        Expression::Literal(Literal::Bool(false)),
                    )],
                    condition: None,
                    node_type: NodeType::Normal,
                    to_edges: BTreeSet::from([5]),
                    from_edges: BTreeSet::from([2]),
                },
                ControlFlowNode {
                    id: 4,
                    statements: vec![Statement::VarAssignment(
                        "ret".to_string(),
                        Expression::Literal(Literal::Bool(true)),
                    )],
                    condition: None,
                    node_type: NodeType::Normal,
                    to_edges: BTreeSet::from([5]),
                    from_edges: BTreeSet::from([2]),
                },
                ControlFlowNode {
                    id: 5,
                    statements: vec![],
                    condition: None,
                    node_type: NodeType::Normal,
                    to_edges: BTreeSet::from([7]),
                    from_edges: BTreeSet::from([3, 4]),
                },
                ControlFlowNode {
                    id: 6,
                    statements: vec![Statement::VarAssignment(
                        "ret".to_string(),
                        Expression::Literal(Literal::Bool(true)),
                    )],
                    condition: None,
                    node_type: NodeType::Normal,
                    to_edges: BTreeSet::from([7]),
                    from_edges: BTreeSet::from([1]),
                },
                ControlFlowNode {
                    id: 7,
                    statements: vec![Statement::Return(Some(Expression::Identifier(
                        "ret".to_string(),
                    )))],
                    condition: None,
                    node_type: NodeType::Normal,
                    to_edges: BTreeSet::new(),
                    from_edges: BTreeSet::from([5, 6]),
                },
            ],
            last_node_id: 7,
            dominators: None,
        };

        assert_eq!(function_cfg, &expected);
    }

    #[test]
    fn test_multiple_statements() {
        let ast = parse(
            "
            contract Vault[Open, Full, Locked] {
                constructor() {
                    var ret: bool = false;
                    var x: int = 1;
                    var y: int = 2;
                    var z: int = x + y;
                    if (amount > 0) {
                        ret = false;
                        z = x + y - 2;
                    }
                    const w: int = 7;
                    z = x + y + w;
                    return z;
                }
            }",
        )
        .unwrap();

        let mut tc = TypeChecker::new();
        let _env = tc.check_program(&ast);

        let mut cfg = FunctionsCFG::new();
        cfg.parse_program(&ast);

        let function_cfg: &ControlFlowGraph = cfg.get("constructor").unwrap();

        let expected = ControlFlowGraph {
            nodes: vec![
                ControlFlowNode {
                    id: 0,
                    statements: vec![
                        Statement::VarDeclaration(
                            "ret".to_string(),
                            Type::Bool,
                            Some(Expression::Literal(Literal::Bool(false))),
                        ),
                        Statement::VarDeclaration(
                            "x".to_string(),
                            Type::Int,
                            Some(Expression::Literal(Literal::Int(1))),
                        ),
                        Statement::VarDeclaration(
                            "y".to_string(),
                            Type::Int,
                            Some(Expression::Literal(Literal::Int(2))),
                        ),
                        Statement::VarDeclaration(
                            "z".to_string(),
                            Type::Int,
                            Some(Expression::BinaryOp(
                                Box::new(Expression::Identifier("x".to_string())),
                                BinaryOperator::Add,
                                Box::new(Expression::Identifier("y".to_string())),
                            )),
                        ),
                    ],
                    condition: None,
                    node_type: NodeType::Normal,
                    to_edges: BTreeSet::from([1]),
                    from_edges: BTreeSet::new(),
                },
                ControlFlowNode {
                    id: 1,
                    statements: vec![],
                    condition: Some(Expression::BinaryOp(
                        Box::new(Expression::Identifier("amount".to_string())),
                        BinaryOperator::Gt,
                        Box::new(Expression::Literal(Literal::Int(0))),
                    )),
                    node_type: NodeType::Condition,
                    to_edges: BTreeSet::from([2, 3]),
                    from_edges: BTreeSet::from([0]),
                },
                ControlFlowNode {
                    id: 2,
                    statements: vec![
                        Statement::VarAssignment(
                            "ret".to_string(),
                            Expression::Literal(Literal::Bool(false)),
                        ),
                        Statement::VarAssignment(
                            "z".to_string(),
                            Expression::BinaryOp(
                                Box::new(Expression::BinaryOp(
                                    Box::new(Expression::Identifier("x".to_string())),
                                    BinaryOperator::Add,
                                    Box::new(Expression::Identifier("y".to_string())),
                                )),
                                BinaryOperator::Sub,
                                Box::new(Expression::Literal(Literal::Int(2))),
                            ),
                        ),
                    ],
                    condition: None,
                    node_type: NodeType::Normal,
                    to_edges: BTreeSet::from([3]),
                    from_edges: BTreeSet::from([1]),
                },
                ControlFlowNode {
                    id: 3,
                    statements: vec![
                        Statement::ConstDeclaration(
                            "w".to_string(),
                            Type::Int,
                            Expression::Literal(Literal::Int(7)),
                        ),
                        Statement::VarAssignment(
                            "z".to_string(),
                            Expression::BinaryOp(
                                Box::new(Expression::BinaryOp(
                                    Box::new(Expression::Identifier("x".to_string())),
                                    BinaryOperator::Add,
                                    Box::new(Expression::Identifier("y".to_string())),
                                )),
                                BinaryOperator::Add,
                                Box::new(Expression::Identifier("w".to_string())),
                            ),
                        ),
                        Statement::Return(Some(Expression::Identifier("z".to_string()))),
                    ],
                    condition: None,
                    node_type: NodeType::Normal,
                    to_edges: BTreeSet::new(),
                    from_edges: BTreeSet::from([1, 2]),
                },
            ],
            last_node_id: 3,
            dominators: None,
        };

        assert_eq!(function_cfg, &expected);
    }

    #[test]
    fn test_if_while_dominators() {
        let ast = parse(
            "
            contract Vault[Open, Full, Locked] {
                constructor() {
                    var ret: bool;
                    if (amount > 0) {
                        while (amount > 2) {
                            ret = false;
                        }
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

        let mut cfg = FunctionsCFG::new();
        cfg.parse_program(&ast);

        let function_cfg: &mut ControlFlowGraph = cfg.get_mut("constructor").unwrap();

        function_cfg.calculate_dominators();
        let expected = vec![
            (0, BTreeSet::from([0])),
            (1, BTreeSet::from([0, 1])),
            (2, BTreeSet::from([0, 1, 2])),
            (3, BTreeSet::from([0, 1, 2, 3])),
            (4, BTreeSet::from([0, 1, 2, 4])),
            (5, BTreeSet::from([0, 1, 5])),
            (6, BTreeSet::from([0, 1, 6])),
        ]
        .into_iter()
        .collect();

        assert_eq!(function_cfg.dominators, Some(expected));
    }

    #[test]
    fn test_while_if_dominators() {
        let ast = parse(
            "
            contract Vault[Open, Full, Locked] {
                constructor() {
                    var ret: bool;
                    while (amount > 0) {
                        if (amount > 2) {
                            ret = false;
                        } else {
                            ret = true;
                        }
                    }
                    return ret;
                }
            }",
        )
        .unwrap();

        let mut tc = TypeChecker::new();
        let _env = tc.check_program(&ast);

        let mut cfg = FunctionsCFG::new();
        cfg.parse_program(&ast);

        let function_cfg: &mut ControlFlowGraph = cfg.get_mut("constructor").unwrap();

        function_cfg.calculate_dominators();
        let expected = vec![
            (0, BTreeSet::from([0])),
            (1, BTreeSet::from([0, 1])),
            (2, BTreeSet::from([0, 1, 2])),
            (3, BTreeSet::from([0, 1, 2, 3])),
            (4, BTreeSet::from([0, 1, 2, 4])),
            (5, BTreeSet::from([0, 1, 2, 5])),
            (6, BTreeSet::from([0, 1, 6])),
        ]
        .into_iter()
        .collect();

        assert_eq!(function_cfg.dominators, Some(expected));
    }

    /*#[test]
    fn test_immediate_dominator() {
        let ast = parse(
            "
            contract Vault[Open, Full, Locked] {
                constructor() {
                    var ret: bool;
                    while (amount > 0) {
                        if (amount > 2) {
                            ret = false;
                        } else {
                            ret = true;
                        }
                    }
                    return ret;
                }
            }",
        )
        .unwrap();

        let mut tc = TypeChecker::new();
        let _env = tc.check_program(&ast);

        let mut cfg = FunctionsCFG::new();
        cfg.parse_program(&ast);

        let function_cfg: &mut ControlFlowGraph = cfg.get_mut("constructor").unwrap();

        function_cfg.calculate_dominators();

        assert_eq!(function_cfg.immediate_dominator(0), None);
        assert_eq!(function_cfg.immediate_dominator(1), Some(0));
        assert_eq!(function_cfg.immediate_dominator(2), Some(1));
        assert_eq!(function_cfg.immediate_dominator(3), Some(2));
        assert_eq!(function_cfg.immediate_dominator(4), Some(2));
        assert_eq!(function_cfg.immediate_dominator(5), Some(2));
        assert_eq!(function_cfg.immediate_dominator(6), Some(1));
    }*/

    #[test]
    fn test_if_dominance_frontier() {
        let ast = parse(
            "
            contract Vault[Open, Full, Locked] {
                constructor() {
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

        let mut cfg = FunctionsCFG::new();
        cfg.parse_program(&ast);

        let function_cfg: &mut ControlFlowGraph = cfg.get_mut("constructor").unwrap();

        function_cfg.calculate_dominators();
        let dominance_frontier = function_cfg.dominance_frontiers();

        let expected = vec![
            (0, BTreeSet::new()),
            (1, BTreeSet::new()),
            (2, BTreeSet::from([4])),
            (3, BTreeSet::from([4])),
            (4, BTreeSet::new()),
        ]
        .into_iter()
        .collect();

        assert_eq!(dominance_frontier, expected);
    }

    #[test]
    fn test_while_if_dominance_frontier() {
        let ast = parse(
            "
            contract Vault[Open, Full, Locked] {
                constructor() {
                    var ret: bool;
                    while (amount > 0) {
                        if (amount > 2) {
                            ret = false;
                        } else {
                            ret = true;
                        }
                    }
                    return ret;
                }
            }",
        )
        .unwrap();

        let mut tc = TypeChecker::new();
        let _env = tc.check_program(&ast);

        let mut cfg = FunctionsCFG::new();
        cfg.parse_program(&ast);

        let function_cfg: &mut ControlFlowGraph = cfg.get_mut("constructor").unwrap();

        function_cfg.calculate_dominators();
        let dominance_frontier = function_cfg.dominance_frontiers();

        let expected = vec![
            (0, BTreeSet::new()),
            (1, BTreeSet::new()),
            (2, BTreeSet::new()),
            (3, BTreeSet::from([5])),
            (4, BTreeSet::from([5])),
            (5, BTreeSet::from([1])),
            (6, BTreeSet::new()),
        ]
        .into_iter()
        .collect();

        assert_eq!(dominance_frontier, expected);
    }

    #[test]
    fn test_local_value_numbering() {
        let ast = parse(
            "
            contract Vault[Open, Full, Locked] {
                constructor() {
                    var a: int = 4;
                    var b: int = 2;
                    var sum1: int = a + b;
                    var sum2: int = a + b;
                    var prod: int = sum1 * sum2;
                    return prod;
                }
            }",
        )
        .unwrap();

        let mut tc = TypeChecker::new();
        let _env = tc.check_program(&ast);

        let mut cfg = FunctionsCFG::new();
        cfg.parse_program(&ast);

        let function_cfg: &mut ControlFlowGraph = cfg.get_mut("constructor").unwrap();

        assert!(function_cfg.nodes.len() == 1);
        let mut node = function_cfg.nodes.get_mut(0).unwrap();
        node.local_value_numbering();

        let expected = ControlFlowNode {
            id: 0,
            statements: vec![
                Statement::VarDeclaration(
                    "a".to_string(),
                    Type::Int,
                    Some(Expression::Literal(Literal::Int(4))),
                ),
                Statement::VarDeclaration(
                    "b".to_string(),
                    Type::Int,
                    Some(Expression::Literal(Literal::Int(2))),
                ),
                Statement::VarDeclaration(
                    "sum1".to_string(),
                    Type::Int,
                    Some(Expression::BinaryOp(
                        Box::new(Expression::Identifier("a".to_string())),
                        BinaryOperator::Add,
                        Box::new(Expression::Identifier("b".to_string())),
                    )),
                ),
                Statement::VarDeclaration(
                    "prod".to_string(),
                    Type::Int,
                    Some(Expression::BinaryOp(
                        Box::new(Expression::Identifier("sum1".to_string())),
                        BinaryOperator::Mul,
                        Box::new(Expression::Identifier("sum1".to_string())),
                    )),
                ),
                Statement::Return(Some(Expression::Identifier("prod".to_string()))),
            ],
            condition: None,
            node_type: NodeType::Normal,
            to_edges: BTreeSet::new(),
            from_edges: BTreeSet::new(),
        };

        assert_eq!(node, &expected);
    }

    #[test]
    fn test_local_value_numbering_const_propagation() {
        let ast = parse(
            "
            contract Vault[Open, Full, Locked] {
                constructor() {
                    var x: int = 4;
                    var copy1: int = x;
                    var copy2: int = copy1;
                    var copy3: int = copy2;
                    return copy3;
                }
            }",
        )
        .unwrap();

        let mut tc = TypeChecker::new();
        let _env = tc.check_program(&ast);

        let mut cfg = FunctionsCFG::new();
        cfg.parse_program(&ast);

        let function_cfg: &mut ControlFlowGraph = cfg.get_mut("constructor").unwrap();

        assert!(function_cfg.nodes.len() == 1);
        let mut node = function_cfg.nodes.get_mut(0).unwrap();
        node.local_value_numbering();

        let expected = ControlFlowNode {
            id: 0,
            statements: vec![
                Statement::VarDeclaration(
                    "x".to_string(),
                    Type::Int,
                    Some(Expression::Literal(Literal::Int(4))),
                ),
                Statement::Return(Some(Expression::Identifier("x".to_string()))),
            ],
            condition: None,
            node_type: NodeType::Normal,
            to_edges: BTreeSet::new(),
            from_edges: BTreeSet::new(),
        };

        assert_eq!(node, &expected);
    }

    #[test]
    fn test_local_value_numbering_copy_propagation() {
        let ast = parse(
            "
            contract Vault[Open, Full, Locked] {
                constructor({y:int}) {
                    var x: int = y;
                    var copy1: int = x;
                    var copy2: int = copy1;
                    var copy3: int = copy2;
                    return copy3;
                }
            }",
        )
        .unwrap();

        let mut tc = TypeChecker::new();
        let _env = tc.check_program(&ast);

        let mut cfg = FunctionsCFG::new();
        cfg.parse_program(&ast);

        let function_cfg: &mut ControlFlowGraph = cfg.get_mut("constructor").unwrap();

        assert!(function_cfg.nodes.len() == 1);
        let mut node = function_cfg.nodes.get_mut(0).unwrap();
        node.local_value_numbering();

        let expected = ControlFlowNode {
            id: 0,
            statements: vec![Statement::Return(Some(Expression::Identifier(
                "y".to_string(),
            )))],
            condition: None,
            node_type: NodeType::Normal,
            to_edges: BTreeSet::new(),
            from_edges: BTreeSet::new(),
        };

        assert_eq!(node, &expected);
    }

    #[test]
    fn test_local_value_numbering_commutativity() {
        let ast = parse(
            "
            contract Vault[Open, Full, Locked] {
                constructor({x:int}, {y:int}) {
                    var sum1: int = x + y;
                    var sum2: int = y + x;
                    return sum2;
                }
            }",
        )
        .unwrap();

        let mut tc = TypeChecker::new();
        let _env = tc.check_program(&ast);

        let mut cfg = FunctionsCFG::new();
        cfg.parse_program(&ast);

        let function_cfg: &mut ControlFlowGraph = cfg.get_mut("constructor").unwrap();

        assert!(function_cfg.nodes.len() == 1);
        let mut node = function_cfg.nodes.get_mut(0).unwrap();
        node.local_value_numbering();

        let expected = ControlFlowNode {
            id: 0,
            statements: vec![
                Statement::VarDeclaration(
                    "sum1".to_string(),
                    Type::Int,
                    Some(Expression::BinaryOp(
                        Box::new(Expression::Identifier("x".to_string())),
                        BinaryOperator::Add,
                        Box::new(Expression::Identifier("y".to_string())),
                    )),
                ),
                Statement::Return(Some(Expression::Identifier("sum1".to_string()))),
            ],
            condition: None,
            node_type: NodeType::Normal,
            to_edges: BTreeSet::new(),
            from_edges: BTreeSet::new(),
        };

        assert_eq!(node, &expected);
    }
}
