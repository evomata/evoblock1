use gridsim::{Sim, Neighborhood};
use gridsim::neumann::*;

enum EvoBlock {}

struct Cell;
struct Diff;
struct Move;

impl<'a> Sim<'a> for EvoBlock {
    type Cell = Cell;
    type Diff = Diff;
    type Move = Move;

    type Neighbors = Neighbors<&'a Cell>;
    type MoveNeighbors = Neighbors<Move>;

    fn step(cell: &Cell, neighbors: Self::Neighbors) -> (Diff, Self::MoveNeighbors) {
        (Diff, Neighbors::new(|_| Move))
    }

    fn update(cell: &mut Cell, diff: Diff, moves: Self::MoveNeighbors) {
    }
}