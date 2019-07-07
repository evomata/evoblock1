pub mod brain;

pub use brain::{Hiddens, InputVector, Network};
use Block::*;

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Brain {
    pub network: Network,
    pub hiddens: Hiddens,
}

#[derive(Debug, Clone, Copy, PartialEq)]
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

impl Cell {
    #[inline]
    pub fn signal(&self) -> f32 {
        match self {
            Cell::Life(Life {
                brain: Brain { hiddens, .. },
                ..
            }) => hiddens.output()[0],
            Cell::Block(block) => match block {
                Birth => -0.1,
                Death => -0.2,
            },
            Cell::None => -1.0,
        }
    }

    #[inline]
    pub fn give(&mut self, block: Block) {
        match self {
            Cell::Life(Life { holding, .. }) => *holding = Some(block),
            _ => *self = Cell::Block(block),
        }
    }
}

impl Default for Cell {
    fn default() -> Self {
        Cell::None
    }
}
