mod cell;

pub use cell::{
    Block::{self, *},
    Brain, Cell, Life,
};
use cell::{Hiddens, InputVector};

use gridsim::neumann::*;
use gridsim::{Direction, Neighborhood, Sim};
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

    type Neighbors = NeumannNeighbors<&'a Cell>;
    type MoveNeighbors = NeumannNeighbors<Move>;

    fn step(cell: &Cell, neighbors: Self::Neighbors) -> (Diff, Self::MoveNeighbors) {
        match cell {
            Cell::Life(Life {
                brain: Brain { network, hiddens },
                ..
            }) => {
                let input: InputVector =
                    InputVector::from_iterator(neighbors.iter().map(Cell::signal));
                let hiddens = network.apply(&input, hiddens.clone());
                let output = hiddens.output().as_slice();
                let move_choice =
                    NeumannDirection::chooser_slice(&output[8..16]).map(|(dir, _)| dir);
                let incubate_chooser = NeumannNeighbors::chooser_slice(&output[16..24]);
                let destroy_chooser = NeumannNeighbors::chooser_slice(&output[24..32]);
                let moves = NeumannNeighbors::new(|dir| {
                    if Some(dir) == move_choice {
                        Move::Brain(Brain {
                            network: network.clone(),
                            hiddens: hiddens.clone(),
                        })
                    } else if incubate_chooser[dir] {
                        Move::Incubate(Brain {
                            network: network.clone(),
                            hiddens: hiddens.clone(),
                        })
                    } else if destroy_chooser[dir] {
                        Move::Destroy
                    } else {
                        Move::Nothing
                    }
                });
                (
                    if move_choice.is_some() {
                        Diff::Destroy
                    } else {
                        Diff::Hiddens(hiddens.clone())
                    },
                    moves,
                )
            }
            Cell::None | Cell::Block(..) => (Diff::None, NeumannNeighbors::new(|_| Move::Nothing)),
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
