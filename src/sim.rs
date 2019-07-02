mod cell;

pub use cell::{Brain, Cell};
use cell::{Hiddens, InputVector};

use boolinator::Boolinator;
use gridsim::neumann::*;
use gridsim::{Neighborhood, Sim};
use rand::Rng;

const MUTATE_LAMBDA: f64 = 0.001;
const CELL_SPAWN_CHANCE: f64 = 0.001;
const LIFE_SPAWN_CHANCE: f64 = 0.0001;

pub enum EvoBlock {}

pub enum Diff {
    Hiddens(Hiddens),
    Destroy,
    None,
}

pub enum Move {
    Brain(Brain),
    Destroy,
    Nothing,
}

impl<'a> Sim<'a> for EvoBlock {
    type Cell = Cell;
    type Diff = Diff;
    type Move = Move;

    type Neighbors = Neighbors<&'a Cell>;
    type MoveNeighbors = Neighbors<Move>;

    fn step(cell: &Cell, neighbors: Self::Neighbors) -> (Diff, Self::MoveNeighbors) {
        match cell {
            Cell::Brain(Brain { network, hiddens }) => {
                let input: InputVector =
                    InputVector::from_iterator(neighbors.iter().map(|cell| match cell {
                        Cell::Brain { .. } => 1.0,
                        Cell::LifeBlock => 0.5,
                        Cell::None => 0.0,
                    }));
                let hiddens = network.apply(&input, hiddens.clone());
                let move_index = hiddens
                    .output()
                    .iter()
                    .take(8)
                    .cloned()
                    .enumerate()
                    .max_by_key(|&(_, n)| float_ord::FloatOrd(n))
                    .and_then(|(ix, value)| (value > 0.5).as_some(ix));
                let moves = Neighbors::new(|dir| {
                    if move_index.is_some() && move_index.unwrap() == dir_to_index(dir) {
                        Move::Brain(Brain {
                            network: network.clone(),
                            hiddens: hiddens.clone(),
                        })
                    } else if hiddens.output()[dir_to_index(dir) + 8] > 0.5 {
                        Move::Destroy
                    } else {
                        Move::Nothing
                    }
                });
                (
                    if move_index.is_some() {
                        Diff::Destroy
                    } else {
                        Diff::Hiddens(hiddens.clone())
                    },
                    moves,
                )
            }
            Cell::None | Cell::LifeBlock => (Diff::None, Neighbors::new(|_| Move::Nothing)),
        }
    }

    fn update(cell: &mut Cell, diff: Diff, moves: Self::MoveNeighbors) {
        // Handle diffs
        match cell {
            Cell::Brain(Brain { network, hiddens }) => {
                // Mutate network
                network.mutate(MUTATE_LAMBDA);
                // Update hiddens
                match diff {
                    Diff::Hiddens(new_hiddens) => *hiddens = new_hiddens,
                    Diff::Destroy => *cell = Cell::None,
                    Diff::None => {}
                }
            }
            Cell::None | Cell::LifeBlock => {}
        }

        // Handle moves
        for mv in moves.iter() {
            match mv {
                Move::Brain(brain) => *cell = Cell::Brain(brain),
                Move::Destroy => *cell = Cell::None,
                Move::Nothing => {}
            }
        }

        // Handle spawn-in
        if rand::thread_rng().gen_bool(CELL_SPAWN_CHANCE) {
            *cell = Cell::Brain(Brain::default());
        }
        if rand::thread_rng().gen_bool(LIFE_SPAWN_CHANCE) {
            *cell = Cell::LifeBlock;
        }
    }
}

#[inline]
fn dir_to_index(dir: Direction) -> usize {
    match dir {
        Direction::Right => 0,
        Direction::UpRight => 1,
        Direction::Up => 2,
        Direction::UpLeft => 3,
        Direction::Left => 4,
        Direction::DownLeft => 5,
        Direction::Down => 6,
        Direction::DownRight => 7,
    }
}
