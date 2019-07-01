mod sim;

use sim::{Cell, EvoBlock};

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
        Cell::Brain { hiddens, .. } => [
            hiddens.output()[8] * 0.5 + 0.5,
            hiddens.output()[9] * 0.5 + 0.5,
            hiddens.output()[10] * 0.5 + 0.5,
            1.0,
        ],
        Cell::None => [0.0, 0.0, 0.0, 1.0],
    });
    ui_loop.run(gridsim::SquareGrid::<EvoBlock>::new(DIMS.0, DIMS.1));
}