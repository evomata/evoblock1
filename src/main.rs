mod sim;

use sim::{Block::*, Brain, Cell, EvoBlock, Life};

const DIMS: (usize, usize) = (256, 144);
//const DIMS: (usize, usize) = (426, 240);
//const DIMS: (usize, usize) = (640, 360);
//const DIMS: (usize, usize) = (768, 432);
//const DIMS: (usize, usize) = (800, 450);
//const DIMS: (usize, usize) = (896, 504);
//const DIMS: (usize, usize) = (960, 540);
//const DIMS: (usize, usize) = (1280, 720);

fn main() {
    let ui_loop = gridsim_ui::Loop::new(|c| match c {
        Cell::Life(Life {
            brain: Brain { hiddens, .. },
            ..
        }) => [
            hiddens.output()[0] * 0.95 + 0.05,
            hiddens.output()[1] * 0.95 + 0.05,
            hiddens.output()[2] * 0.95 + 0.05,
        ],
        Cell::Block(Birth) => [0.0, 1.0, 0.0],
        Cell::Block(Death) => [1.0, 0.0, 0.0],
        Cell::None => [0.0, 0.0, 0.0],
    });
    ui_loop.run(gridsim::SquareGrid::<EvoBlock>::new(DIMS.0, DIMS.1));
}
