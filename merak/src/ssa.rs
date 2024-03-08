use std::collections::{BTreeSet, HashMap, HashSet};

use crate::{
    ast::{Expression, Statement},
    cfg::{ControlFlowGraph, ControlFlowNode, NodeType},
};

#[derive(Debug)]
pub struct SSAGraph {
    nodes: Vec<SSANode>,
}

#[derive(Debug)]
pub struct SSANode {
    id: usize,
    statements: Vec<Statement>,
    condition: Option<Expression>,
    node_type: NodeType,
    to_edges: BTreeSet<usize>,
    from_edges: BTreeSet<usize>,
    phi_params: HashMap<String, Vec<String>>,
}

impl SSANode {
    pub fn init_phi_params(&mut self, key: &str) {
        self.phi_params.insert(key.to_string(), Vec::new());
    }

    pub fn add_phi_param(&mut self, key: &str, param: &str) {
        self.phi_params
            .entry(key.to_string())
            .or_insert(Vec::new())
            .push(param.to_string());
    }

    pub fn remove_phi_param(&mut self, key: &str, param: &str) {
        if let Some(params) = self.phi_params.get_mut(key) {
            params.retain(|p| p != param);
        }
    }

    pub fn phi_params_by_key(&self, key: &str) -> &Vec<String> {
        &self.phi_params[key]
    }

    pub fn phi_params_keys(&self) -> Vec<String> {
        self.phi_params.keys().cloned().collect()
    }

    pub fn update_statement(&mut self, index: usize, statement: Statement) {
        self.statements[index] = statement;
    }
}

impl From<&ControlFlowNode> for SSANode {
    fn from(node: &ControlFlowNode) -> Self {
        SSANode {
            id: node.id(),
            statements: node.statements().clone(),
            condition: node.condition().clone(),
            node_type: node.node_type().clone(),
            to_edges: node.to_edges().clone(),
            from_edges: node.from_edges().clone(),
            phi_params: HashMap::new(),
        }
    }
}

impl From<ControlFlowGraph> for SSAGraph {
    fn from(mut cfg: ControlFlowGraph) -> Self {
        if cfg.nodes().is_empty() {
            return SSAGraph { nodes: vec![] };
        }

        if cfg.dominators().is_none() {
            cfg.calculate_dominators();
        }

        let mut ssa_nodes: Vec<SSANode> = cfg
            .nodes()
            .iter()
            .map(|node| SSANode {
                id: node.id(),
                statements: node.statements().clone(),
                condition: node.condition().clone(),
                node_type: node.node_type().clone(),
                to_edges: node.to_edges().clone(),
                from_edges: node.from_edges().clone(),
                phi_params: HashMap::new(),
            })
            .collect();

        let all_vars = cfg.defined_variables();
        let dominance_frontiers = cfg.dominance_frontiers();

        // Add phi parameters to the nodes
        for variable in all_vars.iter() {
            let mut worklist: HashSet<usize> = HashSet::new();
            let mut visited: HashSet<usize> = HashSet::new();

            for node in &ssa_nodes {
                if node
                    .statements
                    .iter()
                    .any(|s| s.defines_variable(&variable))
                {
                    worklist.insert(node.id);
                }
            }

            while !worklist.is_empty() {
                let node_id = *worklist.iter().next().unwrap();
                worklist.remove(&node_id);
                visited.insert(node_id);

                for df in dominance_frontiers[&node_id].iter() {
                    if !visited.contains(&df) {
                        let df_index = ssa_nodes.iter().position(|n| n.id == *df).unwrap();
                        ssa_nodes[df_index].init_phi_params(variable);
                        if !worklist.contains(&df) {
                            worklist.insert(*df);
                        }
                    }
                }
            }
        }

        // Rename variables in SSA form
        let mut variable_map: HashMap<String, usize> =
            all_vars.into_iter().map(|k| (k, 0)).collect();

        let first_id = 0;
        let mut next_nodes = vec![first_id];
        let mut visited: HashSet<usize> = HashSet::new();
        while !next_nodes.is_empty() {
            let node_id = next_nodes.remove(0);

            if visited.contains(&node_id) {
                continue;
            }

            let node_index = ssa_nodes.iter().position(|n| n.id == node_id).unwrap();

            let node_to_edges = ssa_nodes[node_index].to_edges.clone();
            for to_edge in node_to_edges.iter() {
                let to_node_index = ssa_nodes.iter().position(|n| n.id == *to_edge).unwrap();
                let to_node = &mut ssa_nodes[to_node_index];
                for param in to_node.phi_params_keys() {
                    let new_param = format!("{}_{}", param, variable_map[&param]);
                    to_node.add_phi_param(&param, &new_param);
                    //to_node.remove_phi_param(param);
                }
            }

            next_nodes.extend(node_to_edges);

            let node = &mut ssa_nodes[node_index];

            if node.from_edges.len() > 1 {
                for param in node.phi_params_keys() {
                    let new_param = format!("{}_{}", param, variable_map[&param]);
                    if let Some(values) = node.phi_params.remove(&param) {
                        node.phi_params.insert(new_param, values);
                    }
                }
            }

            let mut new_statements = Vec::new();
            for (i, statement) in node.statements.iter().enumerate() {
                let new_statement = rename_statement(statement, &mut variable_map);
                new_statements.push((i, new_statement));
            }

            // There is probably a better way to do this
            for (i, new_statement) in new_statements {
                node.update_statement(i, new_statement);
            }

            if let Some(cond) = &node.condition {
                let new_cond = cond.index_vars(&variable_map);
                node.condition = Some(new_cond);
            }

            visited.insert(node_id);
        }

        SSAGraph { nodes: ssa_nodes }
    }
}

fn rename_statement(statement: &Statement, variable_map: &mut HashMap<String, usize>) -> Statement {
    let new_statement = match statement {
        Statement::VarDeclaration(name, ty, expr) => {
            let new_expr = match expr {
                Some(e) => Some(e.index_vars(&variable_map)),
                None => None,
            };
            let mut last_index = 0;
            variable_map
                .entry(name.to_string())
                .and_modify(|e| {
                    last_index = *e;
                    *e += 1
                })
                .or_insert(0);
            Statement::VarDeclaration(format!("{}_{}", name, last_index), ty.clone(), new_expr)
        }
        Statement::VarAssignment(name, expr) => {
            let new_expr = expr.index_vars(&variable_map);
            let mut last_index = 0;
            variable_map
                .entry(name.to_string())
                .and_modify(|e| {
                    last_index = *e;
                    *e += 1
                })
                .or_insert(0);
            Statement::VarAssignment(format!("{}_{}", name, last_index), new_expr)
        }
        Statement::ConstDeclaration(name, ty, expr) => {
            let new_expr = expr.index_vars(&variable_map);
            //let last_index = variable_map.get(name).unwrap();
            let mut last_index = 0;
            variable_map
                .entry(name.to_string())
                .and_modify(|e| {
                    last_index = *e;
                    *e += 1
                })
                .or_insert(0);
            Statement::ConstDeclaration(format!("{}_{}", name, last_index), ty.clone(), new_expr)
        }
        Statement::If(cond, true_branch, false_branch) => {
            let new_cond = cond.index_vars(&variable_map);
            let new_true_branch: Vec<Statement> = true_branch
                .iter()
                .map(|s| rename_statement(s, variable_map))
                .collect();
            let new_false_branch = false_branch.as_ref().map(|branch| {
                branch
                    .iter()
                    .map(|s| rename_statement(s, variable_map))
                    .collect()
            });

            Statement::If(new_cond, new_true_branch, new_false_branch)
        }
        Statement::While(cond, while_body) => {
            let new_cond = cond.index_vars(&variable_map);
            let new_while_body: Vec<Statement> = while_body
                .iter()
                .map(|s| rename_statement(s, variable_map))
                .collect();
            Statement::While(new_cond, new_while_body)
        }
        Statement::Return(Some(expr)) => {
            let new_expr = expr.index_vars(&variable_map);
            Statement::Return(Some(new_expr))
        }
        Statement::Return(None) => Statement::Return(None),
        Statement::Expression(expr) => Statement::Expression(expr.index_vars(&variable_map)),
    };

    new_statement
}

#[cfg(test)]
mod tests {

    use crate::ast::{BinaryOperator, Expression, Literal, Statement, Type};
    use crate::cfg::{ControlFlowGraph, ControlFlowNode, NodeType};
    use crate::ssa::SSAGraph;

    #[test]
    fn test_if() {
        let mut node0 = ControlFlowNode::new(0, NodeType::Normal, None);
        node0.add_statement(Statement::VarDeclaration(
            "ret".to_string(),
            Type::Bool,
            None,
        ));
        node0.add_to_edge(1);

        let mut node1 = ControlFlowNode::new(
            1,
            NodeType::Condition,
            Some(Expression::BinaryOp(
                Box::new(Expression::Identifier("amount".to_string())),
                BinaryOperator::Gt,
                Box::new(Expression::Literal(Literal::Int(0))),
            )),
        );
        node1.add_to_edge(2);
        node1.add_to_edge(3);
        node1.add_from_edge(0);

        let mut node2 = ControlFlowNode::new(2, NodeType::Normal, None);
        node2.add_statement(Statement::VarAssignment(
            "ret".to_string(),
            Expression::Literal(Literal::Bool(false)),
        ));
        node2.add_to_edge(4);
        node2.add_from_edge(1);

        let mut node3 = ControlFlowNode::new(3, NodeType::Normal, None);
        node3.add_statement(Statement::VarAssignment(
            "ret".to_string(),
            Expression::Literal(Literal::Bool(true)),
        ));
        node3.add_to_edge(4);
        node3.add_from_edge(1);

        let mut node4 = ControlFlowNode::new(4, NodeType::Normal, None);
        node4.add_statement(Statement::Return(Some(Expression::Identifier(
            "ret".to_string(),
        ))));
        node4.add_from_edge(2);
        node4.add_from_edge(3);

        let nodes = vec![node0, node1, node2, node3, node4];

        let graph = ControlFlowGraph::new().with_nodes(nodes);

        let ssa_graph: SSAGraph = graph.into();
        //println!("{:?}", ssa_graph);

        assert_eq!(ssa_graph.nodes.len(), 5);
        assert_eq!(ssa_graph.nodes[0].phi_params_keys().len(), 0);
        assert_eq!(ssa_graph.nodes[1].phi_params_keys().len(), 0);
        assert_eq!(ssa_graph.nodes[2].phi_params_keys().len(), 0);
        assert_eq!(ssa_graph.nodes[3].phi_params_keys().len(), 0);
        assert_eq!(ssa_graph.nodes[4].phi_params_keys().len(), 1);
        assert_eq!(ssa_graph.nodes[4].phi_params_by_key("ret_3").len(), 2);
        assert_eq!(ssa_graph.nodes[4].phi_params_by_key("ret_3")[0], "ret_1");
        assert_eq!(ssa_graph.nodes[4].phi_params_by_key("ret_3")[1], "ret_2");
    }
}
