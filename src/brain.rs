use ordered_float::OrderedFloat;
use ndarray::{Array2, Array1, s, stack, Axis, concatenate, ArrayView, ArrayView1};
use rand::prelude::*;

const CHARACTER_INPUT_SIZE: usize = 256;
const CHARACTER_OUTPUT_SIZE: usize = 256;
const TICK_INPUT_SIZE: usize = 1;
const NOOP_OUTPUT_SIZE: usize = 1;
const INPUT_RANDOM_SIZE: usize = 1;
const MEMORY_SIZE: usize = 16;

/// Trait for all layers.
/// Note that the forward() and backward() methods take a mutable reference to self.
trait Node {

    const NUM_INPUTS: usize;
    const NUM_OUTPUTS: usize;

    fn forward(&mut self, input: &[ArrayView1<f32>; Self::NUM_INPUTS]) -> [ArrayView1<f32>; Self::NUM_OUTPUTS];

}

struct Relu;

impl Node for Relu {
    const NUM_INPUTS: usize = 1;
    const NUM_OUTPUTS: usize = 1;

    fn forward(&mut self, input: &[ArrayView1<f32>; Self::NUM_INPUTS]) -> [ArrayView1<f32>; Self::NUM_OUTPUTS] {
        let input = input[0];
        let mut output = Array1::zeros(input.len());
        for i in 0..input.len() {
            output[i] = input[i].max(0.0);
        }
        [output]
    }
}

struct Dense {
    weights: Array2<f32>
}

impl Dense {
    fn new(input_size: usize, output_size: usize) -> Dense {
        let mut rng = rand::thread_rng();
        let weights = Array2::from_shape_fn((input_size, output_size), |_| rng.gen_range(-1.0, 1.0));
        Dense { weights }
    }
}

impl Node for Dense {
    const NUM_INPUTS: usize = 1;
    const NUM_OUTPUTS: usize = 1;

    fn forward(&mut self, input: &[ArrayView1<f32>; Self::NUM_INPUTS]) -> [ArrayView1<f32>; Self::NUM_OUTPUTS] {
        let input = input[0];
        // Add a bias of 1.0 to the input.
        let input = concatenate![Axis(0), input, Array1::ones(1)];

        let output = self.weights.dot(&input);

        [output.view()]
    }
}

struct Concat<N>;

impl<N:usize> Node for Concat<N> {
    const NUM_INPUTS: usize = N;
    const NUM_OUTPUTS: usize = 1;

    fn forward(&mut self, input: &[ArrayView1<f32>; Self::NUM_INPUTS]) -> [ArrayView1<f32>; Self::NUM_OUTPUTS] {
        let mut output = Array1::zeros(input[0].len() * N);
        for i in 0..N {
            output.slice_mut(s![i*input[0].len()..(i+1)*input[0].len()]).assign(&input[i]);
        }
        [output.view()]
    }
}

struct Split(u32);

impl Node for Split {
    const NUM_INPUTS: usize = 1;
    const NUM_OUTPUTS: usize = 2;

    fn forward(&mut self, input: &[ArrayView1<f32>; Self::NUM_INPUTS]) -> [ArrayView1<f32>; Self::NUM_OUTPUTS] {
        let input = input[0];
        let split = self.0 as usize;
        let output1 = input.slice(s![0..split]);
        let output2 = input.slice(s![split..]);
        [output1, output2]
    }
}






impl<L: NeuralModule> Brain<L> {

    pub fn new() -> Self<L> {

        let layers =
            MemoryLoop {
                memory: Array1::zeros(MEMORY_SIZE),
                inner: Stack {
                    above: Concat2Layer {
                        left: InputLayer {},
                        right: RandomLayer { size: MEMORY_SIZE }
                    },
                    below: dense_relu(CHARACTER_INPUT_SIZE + TICK_INPUT_SIZE + MEMORY_SIZE, 256),
                }
            };

        // Some mock-up code for a more declarative interface.

        model(|input|{

            let input_with_noise = concat(input, random(64));

            let output = memory(64, input_with_noise, |x| {
                let x = dense(64, x);
                let x = relu(x);
                x
            });

            softmax(output)

        });



        Brain {
            layers,
            input_log: Vec::new(),
            output_log: Vec::new()
        }

    }

    pub fn feedback(&mut self, reward: f32) {

        panic!("Not implemented");

    }

    pub fn input(&mut self, input: BrainInput) -> BrainOutput {

        let external_input = Self::input_to_ndarray(input);

        self.input_log.push(external_input.clone());

        // Append a 16x1 array of random values to the input.
        let random = Self::random_input_portion();

        let input = concatenate![Axis(0), external_input, random, self.memory];

        // Multiply the input by the weights (and turn it into an Array1)
        let output = self.weights.dot(&input).into_shape(CHARACTER_OUTPUT_SIZE + NOOP_OUTPUT_SIZE + MEMORY_SIZE).unwrap();

        // Grab the first 257 values of the output.
        let external_output = output.slice(s![..257]);

        // The rest is memory.
        let memory = output.slice(s![257..]);
        // ReLu it.
        let memory = memory.map(|x| x.max(0.0));

        // Store the rest in the memory.
        self.memory = memory;
        assert_eq!(self.memory.len(), 64);

        // Softmax.
        let mut external_output = output.mapv(|x| x.exp());
        external_output /= output.sum();

        self.output_log.push(external_output.clone());

        array_to_output(Self::ndarray_to_output(&mut external_output))

    }

    fn random_input_portion() -> Array1<f32> {
        let mut rng = rand::thread_rng();
        let mut random = ndarray::Array::from_shape_fn(16, |_| rng.gen_range(-1.0..1.0));
        random
    }

    fn ndarray_to_output(output: &mut Array1<f32>) -> [f32; 257] {
        // Convert the output to an array.
        let mut output_arr = [0.0; 257];
        output_arr.copy_from_slice(&output.as_slice().unwrap());
        output_arr
    }

    fn input_to_ndarray(input: BrainInput) -> Array1<f32> {
        let input = input_to_array(input);

        // convert to ndarray
        let input = ndarray::Array::from_shape_vec(257, input.to_vec()).unwrap();
        input
    }
}