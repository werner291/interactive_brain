
use ordered_float::OrderedFloat;

use rand::prelude::*;

pub struct Brain {



}

pub enum BrainInput {
    ChatCharacter(char),
    TimeTick
}

pub fn input_to_array(input: BrainInput) -> [f32; 257] {

    let mut array = [0.0; 257];

    match input {
        BrainInput::ChatCharacter(c) => {
            array[c as usize] = 1.0;
        },
        BrainInput::TimeTick => {
            array[256] = 1.0;
        }
    }

    array
}

pub fn array_to_output(array: [f32; 257]) -> BrainOutput {

    // Find the index of the highest value in the array and interpret as an output.

    let idx = array.iter().enumerate().max_by_key(|&(_, &v)| OrderedFloat(v)).unwrap().0;

    match idx {
        0..=255 => BrainOutput::ChatCharacter(idx as u8 as char),
        256 => BrainOutput::Nothing,
        _ => panic!("Invalid output index")
    }
}

pub enum BrainOutput {
    ChatCharacter(char),
    Nothing
}

 impl Brain {
     pub fn new() -> Self {
        Brain {}
    }

     pub fn input(&mut self, input: BrainInput) -> BrainOutput {
         let input = input_to_array(input);

         // convert to ndarray
         let input = ndarray::Array::from_shape_vec((1, 257), input.to_vec()).unwrap();

         // Create a 257x257 matrix of random values.
        let mut rng = rand::thread_rng();
        let mut weights = ndarray::Array::from_shape_fn((257, 257), |_| rng.gen_range(-1.0 .. 1.0));

        // Multiply the input by the weights.
        let output = input.dot(&weights);

         // Softmax.
            let mut output = output.mapv(|x| x.exp());
            output /= output.sum();

        // Convert the output to an array.
        let mut output_arr = [0.0; 257];
         output_arr.copy_from_slice(&output.as_slice().unwrap());

         array_to_output(output_arr)
    }
}