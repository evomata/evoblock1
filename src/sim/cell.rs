pub mod brain;

#[derive(Debug, Clone)]
pub enum Cell {
    Brain {
        network: brain::Network,
        hiddens: brain::Hiddens,
    },
    None,
}

impl Default for Cell {
    fn default() -> Self {
        Cell::None
    }
}
