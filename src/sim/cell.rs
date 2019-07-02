pub mod brain;

pub use brain::{Hiddens, InputVector, Network};

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Brain {
    pub network: Network,
    pub hiddens: Hiddens,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Cell {
    Brain(Brain),
    LifeBlock,
    DeathBlock,
    None,
}

impl Default for Cell {
    fn default() -> Self {
        Cell::None
    }
}
