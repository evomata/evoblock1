use boolinator::Boolinator;
use nalgebra as na;
use rand::{self, seq::SliceRandom, Rng};
use rand_distr::Poisson;
use std::sync::Arc;

pub type InputMatrix = na::MatrixMN<f32, OutLen, InLen>;
type HiddenMatrix = na::MatrixMN<f32, OutLen, OutLen>;
pub type InputVector = na::MatrixMN<f32, InLen, na::dimension::U1>;
pub type OutputVector = na::MatrixMN<f32, OutLen, na::dimension::U1>;

pub type OutLen = na::dimension::U32;
pub type InLen = na::dimension::U9;

// Number of extra layers
const EXTRA_LAYERS: usize = 2;
// Weight in network maximum
const MAX_WEIGHT: f32 = 2.0;
// Bias in network maximum (cant get output len as compile time constant)
const MAX_BIAS: f32 = MAX_WEIGHT * 32.0;

#[derive(Clone, Debug, PartialEq)]
pub struct Hiddens {
    pub input_hiddens: OutputVector,
    pub internal_hiddens: [OutputVector; EXTRA_LAYERS],
}

impl Default for Hiddens {
    fn default() -> Self {
        Self {
            input_hiddens: OutputVector::from_element(0.5),
            internal_hiddens: array_init::array_init(|_| OutputVector::from_element(0.5)),
        }
    }
}

impl Hiddens {
    pub fn output(&self) -> &OutputVector {
        self.internal_hiddens
            .iter()
            .rev()
            .next()
            .unwrap_or(&self.input_hiddens)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Network {
    pub input_gru: Arc<MGRUInput>,
    pub internal_grus: [Arc<MGRU>; EXTRA_LAYERS],
}

impl Default for Network {
    fn default() -> Self {
        Self {
            input_gru: Arc::new(MGRUInput::new()),
            internal_grus: array_init::array_init(|_| Arc::new(MGRU::new())),
        }
    }
}

impl Network {
    pub fn mutate(&mut self, lambda: f64) {
        if let Some(ngru) = self.input_gru.mutated(lambda) {
            self.input_gru = Arc::new(ngru);
        }
        for gru in &mut self.internal_grus {
            if let Some(ngru) = gru.mutated(lambda) {
                *gru = Arc::new(ngru);
            }
        }
    }

    pub fn apply(&self, inputs: &InputVector, mut hiddens: Hiddens) -> Hiddens {
        let mut hidden_accumulator = self.input_gru.apply(inputs, &hiddens.input_hiddens);
        hiddens.input_hiddens = hidden_accumulator;
        for (ix, gru) in self.internal_grus.iter().enumerate() {
            hidden_accumulator = gru.apply(&hidden_accumulator, &hiddens.internal_hiddens[ix]);
            hiddens.internal_hiddens[ix] = hidden_accumulator;
        }
        hiddens
    }
}

#[inline]
pub fn sigmoid(n: f32) -> f32 {
    (1.0 + (-n).exp()).recip()
}

#[inline]
fn mutate_weight(slice: &mut [f32], lambda: f64) -> bool {
    let mut rng = rand::thread_rng();
    let times =
        rng.sample(Poisson::new(lambda * slice.len() as f64).expect(
            "evoblock1::cell::brain::mutate_lambda(): created invalid poisson distribution",
        ));
    for _ in 0..times {
        *slice
            .choose_mut(&mut rng)
            .expect("evoblock1::cell::brain::mutate_lambda(): mutated empty slice") =
            (rng.gen::<f32>() * 2.0 - 1.0) * MAX_WEIGHT;
    }
    times != 0
}

#[inline]
fn mutate_bias(slice: &mut [f32], lambda: f64) -> bool {
    let mut rng = rand::thread_rng();
    let times =
        rng.sample(Poisson::new(lambda * slice.len() as f64).expect(
            "evoblock1::cell::brain::mutate_lambda(): created invalid poisson distribution",
        ));
    for _ in 0..times {
        *slice
            .choose_mut(&mut rng)
            .expect("evoblock1::cell::brain::mutate_lambda(): mutated empty slice") =
            (rng.gen::<f32>() * 2.0 - 1.0) * MAX_BIAS;
    }
    times != 0
}

#[derive(Clone, Debug, PartialEq)]
struct GRUNet {
    hidden_matrix: HiddenMatrix,
    input_matrix: HiddenMatrix,
    biases: OutputVector,
}

impl GRUNet {
    fn new_forget() -> GRUNet {
        GRUNet {
            hidden_matrix: HiddenMatrix::new_random().map(|n| (n * 2.0 - 1.0) * MAX_WEIGHT),
            input_matrix: HiddenMatrix::new_random().map(|n| (n * 2.0 - 1.0) * MAX_WEIGHT),
            // Make the gate initially completely ignore hidden state.
            biases: OutputVector::from_element(-1000.0),
        }
    }

    fn new_output() -> GRUNet {
        GRUNet {
            hidden_matrix: HiddenMatrix::new_random().map(|n| (n * 2.0 - 1.0) * MAX_WEIGHT),
            input_matrix: HiddenMatrix::new_random().map(|n| (n * 2.0 - 1.0) * MAX_WEIGHT),
            biases: OutputVector::new_random().map(|n| (n * 2.0 - 1.0) * MAX_BIAS),
        }
    }

    #[inline]
    fn apply_linear(&self, hiddens: &OutputVector, inputs: &OutputVector) -> OutputVector {
        &self.hidden_matrix * hiddens + &self.input_matrix * inputs + &self.biases
    }

    #[inline]
    fn apply_sigmoid(&self, hiddens: &OutputVector, inputs: &OutputVector) -> OutputVector {
        self.apply_linear(hiddens, inputs).map(sigmoid)
    }

    /// Mutate each matrix element with a probability lambda
    #[inline]
    fn mutate(&mut self, lambda: f64) -> bool {
        mutate_weight(self.hidden_matrix.as_mut_slice(), lambda)
            | mutate_weight(self.input_matrix.as_mut_slice(), lambda)
            | mutate_bias(self.biases.as_mut_slice(), lambda)
    }
}

#[derive(Clone, Debug, PartialEq)]
struct GRUNetInput {
    hidden_matrix: HiddenMatrix,
    input_matrix: InputMatrix,
    biases: OutputVector,
}

impl GRUNetInput {
    fn new_forget() -> GRUNetInput {
        GRUNetInput {
            hidden_matrix: HiddenMatrix::new_random().map(|n| (n * 2.0 - 1.0) * MAX_WEIGHT),
            input_matrix: InputMatrix::new_random().map(|n| (n * 2.0 - 1.0) * MAX_WEIGHT),
            // Make the gate initially completely ignore hidden state.
            biases: OutputVector::from_element(-1000.0),
        }
    }

    fn new_output() -> GRUNetInput {
        GRUNetInput {
            hidden_matrix: HiddenMatrix::new_random().map(|n| (n * 2.0 - 1.0) * MAX_WEIGHT),
            input_matrix: InputMatrix::new_random().map(|n| (n * 2.0 - 1.0) * MAX_WEIGHT),
            biases: OutputVector::new_random().map(|n| (n * 2.0 - 1.0) * MAX_BIAS),
        }
    }

    #[inline]
    fn apply_linear(&self, hiddens: &OutputVector, inputs: &InputVector) -> OutputVector {
        &self.hidden_matrix * hiddens + &self.input_matrix * inputs + &self.biases
    }

    #[inline]
    fn apply_sigmoid(&self, hiddens: &OutputVector, inputs: &InputVector) -> OutputVector {
        self.apply_linear(hiddens, inputs).map(sigmoid)
    }

    /// Mutate each matrix element with a probability lambda
    #[inline]
    fn mutate(&mut self, lambda: f64) -> bool {
        mutate_weight(self.hidden_matrix.as_mut_slice(), lambda)
            | mutate_weight(self.input_matrix.as_mut_slice(), lambda)
            | mutate_bias(self.biases.as_mut_slice(), lambda)
    }
}

/// A Minimal Gated Recurrent Unit
#[derive(Clone, Debug, PartialEq)]
pub struct MGRUInput {
    forget_gate: GRUNetInput,
    output_gate: GRUNetInput,
}

impl MGRUInput {
    pub fn new() -> MGRUInput {
        MGRUInput {
            forget_gate: GRUNetInput::new_forget(),
            output_gate: GRUNetInput::new_output(),
        }
    }

    pub fn apply(&self, inputs: &InputVector, hiddens: &OutputVector) -> OutputVector {
        // Compute forget coefficients.
        let f = self.forget_gate.apply_sigmoid(hiddens, inputs);

        let remebered = f.zip_map(hiddens, |f, h| f * h);

        remebered
            + f.zip_map(
                &self.output_gate.apply_sigmoid(&remebered, inputs),
                |f, o| (1.0 - f) * o,
            )
    }

    pub fn mutated(&self, lambda: f64) -> Option<Self> {
        let mut new = self.clone();
        let changed = new.forget_gate.mutate(lambda) | new.output_gate.mutate(lambda);
        changed.as_some(new)
    }
}

/// A Minimal Gated Recurrent Unit
#[derive(Clone, Debug, PartialEq)]
pub struct MGRU {
    forget_gate: GRUNet,
    output_gate: GRUNet,
}

impl MGRU {
    pub fn new() -> MGRU {
        MGRU {
            forget_gate: GRUNet::new_forget(),
            output_gate: GRUNet::new_output(),
        }
    }

    pub fn apply(&self, inputs: &OutputVector, hiddens: &OutputVector) -> OutputVector {
        // Compute forget coefficients.
        let f = self.forget_gate.apply_sigmoid(hiddens, inputs);

        let remebered = f.zip_map(hiddens, |f, h| f * h);

        remebered
            + f.zip_map(
                &self.output_gate.apply_sigmoid(&remebered, inputs),
                |f, o| (1.0 - f) * o,
            )
    }

    pub fn mutated(&self, lambda: f64) -> Option<Self> {
        let mut new = self.clone();
        let changed = new.forget_gate.mutate(lambda) | new.output_gate.mutate(lambda);
        changed.as_some(new)
    }
}
