pub mod brain;

pub use brain::{Hiddens, InputVector, Network};

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Brain {
    pub network: Network,
    pub hiddens: Hiddens,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Block {
    Birth,
    Death,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Life {
    pub brain: Brain,
    pub holding: Option<Block>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Cell {
    Life(Life),
    Block(Block),
    None,
}

impl Default for Cell {
    fn default() -> Self {
        Cell::None
    }
}
