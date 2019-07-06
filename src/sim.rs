mod cell;

pub use cell::{
    Block::{self, *},
    Brain, Cell, Life,
};
use cell::{Hiddens, InputVector};

use boolinator::Boolinator;
use gridsim::neumann::*;
use gridsim::{Neighborhood, Sim};
use rand::Rng;

const MUTATE_LAMBDA: f64 = 0.001;
const SPAWN_RATE: f64 = 0.000001;
const CELL_SPAWN: f64 = 1.0 * SPAWN_RATE;
const BIRTH_SPAWN: f64 = 2.0 * SPAWN_RATE;
const DEATH_SPAWN: f64 = 3.0 * SPAWN_RATE;

pub enum EvoBlock {}

pub enum Diff {
    Hiddens(Hiddens),
    Destroy,
    None,
}

pub enum Move {
    Brain(Brain),
    Incubate(Brain),
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
            Cell::Life(Life {
                brain: Brain { network, hiddens },
                ..
            }) => {
                let input: InputVector =
                    InputVector::from_iterator(neighbors.iter().map(Cell::signal));
                let hiddens = network.apply(&input, hiddens.clone());
                let move_index = hiddens
                    .output()
                    .iter()
                    .skip(8)
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
                    } else if hiddens.output()[dir_to_index(dir) + 16] > 0.5 {
                        Move::Incubate(Brain {
                            network: network.clone(),
                            hiddens: hiddens.clone(),
                        })
                    } else if hiddens.output()[dir_to_index(dir) + 24] > 0.5 {
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
            Cell::None | Cell::Block(..) => (Diff::None, Neighbors::new(|_| Move::Nothing)),
        }
    }

    fn update(cell: &mut Cell, diff: Diff, moves: Self::MoveNeighbors) {
        // Handle diffs
        match cell {
            Cell::Life(Life {
                brain: Brain { network, hiddens },
                ..
            }) => {
                // Mutate network
                network.mutate(MUTATE_LAMBDA);
                // Update hiddens
                match diff {
                    Diff::Hiddens(new_hiddens) => *hiddens = new_hiddens,
                    Diff::Destroy => *cell = Cell::None,
                    Diff::None => {}
                }
            }
            Cell::None | Cell::Block(..) => {}
        }

        // Handle moves
        let incubate_count = moves
            .as_ref()
            .iter()
            .filter(|m| {
                if let Move::Incubate(..) = m {
                    true
                } else {
                    false
                }
            })
            .count();
        let was_death = Cell::Block(Death) == *cell;
        let was_birth = Cell::Block(Birth) == *cell;
        for mv in moves.iter() {
            match mv {
                Move::Brain(brain) => {
                    if was_death {
                        *cell = Cell::None;
                    } else {
                        *cell = Cell::Life(Life {
                            brain,
                            holding: None,
                        });
                    }
                }
                Move::Incubate(brain) => {
                    if was_birth {
                        if incubate_count == 1 {
                            *cell = Cell::Life(Life {
                                brain,
                                holding: None,
                            });
                        }
                    }
                }
                Move::Destroy => {
                    if !was_birth && !was_death {
                        *cell = Cell::None;
                    }
                }
                Move::Nothing => {}
            }
        }

        // Handle spawn-in
        if rand::thread_rng().gen_bool(CELL_SPAWN) {
            *cell = Cell::Life(Life::default());
        }
        if rand::thread_rng().gen_bool(BIRTH_SPAWN) {
            *cell = Cell::Block(Birth);
        }
        if rand::thread_rng().gen_bool(DEATH_SPAWN) {
            *cell = Cell::Block(Death);
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
