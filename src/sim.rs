mod cell;

pub use cell::{
    Block::{self, *},
    Brain, Cell, Life,
};
use cell::{Hiddens, InputVector};

use boolinator::Boolinator;
use gridsim::neumann::*;
use gridsim::{Direction, Neighborhood, Sim};
use rand::Rng;

const MUTATE_LAMBDA: f64 = 0.001;
const SPAWN_RATE: f64 = 0.00001;
const CELL_SPAWN: f64 = 1.0 * SPAWN_RATE;
const BIRTH_SPAWN: f64 = 1000.0 * SPAWN_RATE;
const DEATH_SPAWN: f64 = 1.0 * SPAWN_RATE;

pub enum EvoBlock {}

pub enum Diff {
    Update(Hiddens, Option<Block>),
    Destroy,
    None,
}

pub enum Move {
    Life(Life),
    Incubate(Brain),
    Drop(Block),
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
                holding,
            }) => {
                let input: InputVector = InputVector::from_iterator(
                    neighbors
                        .iter()
                        .map(Cell::signal)
                        .chain(std::iter::once(holding_signal(holding))),
                );
                let hiddens = network.apply(&input, hiddens.clone());
                let output = hiddens.output().as_slice();
                let move_choice =
                    NeumannDirection::chooser_slice(&output[8..16]).map(|(dir, _)| dir);
                let incubate_chooser = NeumannNeighbors::chooser_slice(&output[16..24]);
                let drop_choice =
                    NeumannDirection::chooser_slice(&output[24..32]).map(|(dir, _)| dir);
                let moves = NeumannNeighbors::new(|dir| {
                    if Some(dir) == move_choice {
                        Move::Life(Life {
                            brain: Brain {
                                network: network.clone(),
                                hiddens: hiddens.clone(),
                            },
                            holding: if drop_choice.is_some() {
                                None
                            } else {
                                *holding
                            },
                        })
                    } else if Some(dir) == drop_choice {
                        if let Some(block) = holding {
                            Move::Drop(*block)
                        } else {
                            Move::Nothing
                        }
                    } else if incubate_chooser[dir] {
                        Move::Incubate(Brain {
                            network: network.clone(),
                            hiddens: hiddens.clone(),
                        })
                    } else {
                        Move::Nothing
                    }
                });
                (
                    if move_choice.is_some() {
                        Diff::Destroy
                    } else {
                        Diff::Update(
                            hiddens.clone(),
                            drop_choice.is_none().as_option().and(*holding),
                        )
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
                brain: Brain { hiddens, .. },
                holding,
            }) => {
                // Update hiddens
                match diff {
                    Diff::Update(new_hiddens, new_holding) => {
                        *hiddens = new_hiddens;
                        *holding = new_holding;
                    }
                    Diff::Destroy => *cell = Cell::None,
                    Diff::None => {}
                }
            }
            _ => {}
        }

        let was_death = Cell::Block(Death) == *cell;
        let was_birth = Cell::Block(Birth) == *cell;
        let was_occupied = if let Cell::Life(..) = cell {
            true
        } else {
            false
        };

        // Count number of successful incubate attempts.
        let incubate_count = if was_birth {
            moves
                .as_ref()
                .iter()
                .filter(|m| {
                    if let Move::Incubate(..) = m {
                        true
                    } else {
                        false
                    }
                })
                .count()
        } else {
            0
        };

        // Count number of block drops.
        let drop_count = moves
            .as_ref()
            .iter()
            .filter(|m| if let Move::Drop(..) = m { true } else { false })
            .count();
        // Count number of life movements.
        let life_move_count = moves
            .as_ref()
            .iter()
            .filter(|m| if let Move::Life(..) = m { true } else { false })
            .count();

        // We will obliterate the cell if more than one thing has entered it
        // or life enters a death cell. This avoids bias towards any particular direction.
        let obliteration = incubate_count + drop_count + life_move_count > 1
            || (life_move_count == 1 && was_death);

        if obliteration {
            *cell = Cell::None;
        } else {
            for mv in moves.iter() {
                match mv {
                    Move::Life(life) => {
                        if let Cell::Block(block) = cell {
                            *cell = Cell::Life(Life {
                                brain: life.brain,
                                holding: Some(*block),
                            });
                        } else if !was_occupied {
                            *cell = Cell::Life(life);
                        }
                    }
                    Move::Incubate(brain) => {
                        if was_birth {
                            *cell = Cell::Life(Life {
                                brain,
                                holding: None,
                            });
                        }
                    }
                    Move::Drop(block) => {
                        cell.give(block);
                    }
                    Move::Nothing => {}
                }
            }
        }

        // Handle spawn-in
        if rand::thread_rng().gen_bool(CELL_SPAWN) {
            *cell = Cell::Life(Life::default());
        }
        if rand::thread_rng().gen_bool(BIRTH_SPAWN) {
            cell.give(Birth);
        }
        if rand::thread_rng().gen_bool(DEATH_SPAWN) {
            // This will potentially allow a cell to hold a death block,
            // which is intended and would allow smart cells to build safe areas.
            cell.give(Death);
        }

        // Handle mutation
        match cell {
            Cell::Life(life) => {
                // Mutate network
                life.brain.network.mutate(MUTATE_LAMBDA);
            }
            _ => {}
        }
    }
}

#[inline]
fn holding_signal(holding: &Option<Block>) -> f32 {
    holding
        .map(|block| match block {
            Birth => 1.0,
            Death => 0.5,
        })
        .unwrap_or(0.0)
}
