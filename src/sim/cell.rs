pub mod brain;

pub use brain::{Hiddens, InputVector, Network};

#[derive(Debug, Clone, Default)]
pub struct Brain {
    pub network: Network,
    pub hiddens: Hiddens,
}

#[derive(Debug, Clone)]
pub enum Cell {
    Brain(Brain),
    LifeBlock,
    None,
}

impl Default for Cell {
    fn default() -> Self {
        Cell::None
    }
}
