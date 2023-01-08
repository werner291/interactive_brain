extern crate core;

mod brain;

use std::iter::once;
use std::mem::take;
use std::sync::mpsc::Sender;
use druid::widget::{Button, Flex, Label, LensWrap, TextBox};
use druid::{AppLauncher, LocalizedString, PlatformError, Widget, WidgetExt, WindowDesc, Lens, Data};
use druid::lens::Field;
use crate::brain::{Brain, BrainInput, BrainOutput};

#[derive(Clone, Data, Lens, Default)]
struct UiState {
    pub input: String,
    pub output: String,
}

fn main() -> Result<(), PlatformError> {

    let (input_tx, input_rx) = std::sync::mpsc::channel();

    let main_window = WindowDesc::new(ui_builder(input_tx.clone())).title("External Event Demo");

    let launcher = AppLauncher::with_window(main_window);

    let ctx = launcher.get_external_handle();

    std::thread::spawn(move || {
        let mut brain = Brain::new();
        loop {
            let input = input_rx.recv().unwrap();
            let output = brain.input(input);

            ctx.add_idle_callback(move |data : &mut UiState| {
                match output {
                    BrainOutput::ChatCharacter(c) => {
                        data.output.push(c);
                    }
                    BrainOutput::Nothing => {}
                }
            });

        }
    });

    // Create a timer thread that sends a tick every second.
    std::thread::spawn(move || {
        loop {
            std::thread::sleep(std::time::Duration::from_secs(1));
            input_tx.send(BrainInput::TimeTick).unwrap();
        }
    });


    launcher.log_to_console()
        .launch(UiState::default())

}



fn ui_builder(input_tx: Sender<BrainInput>) -> impl Widget<UiState> {

    let input = TextBox::new()
        .with_placeholder("Input")
        .lens(UiState::input);

    let output = Label::new(|data: &UiState, _env: &_| data.output.clone())
        .with_text_size(12.0);

    let button = Button::new("Send")
        .on_click(move |ctx, data: &mut UiState, env| {

            for c in data.input.chars().chain(once('\n')) {
                input_tx.send(BrainInput::ChatCharacter(c)).unwrap();
            }

        });

    let reward_button = Button::new("Reward")
        .on_click(move |ctx, data: &mut UiState, env| {
        });

    let punish_button = Button::new("Punish")
        .on_click(move |ctx, data: &mut UiState, env| {
        });

    Flex::column()
        .with_child(input)
        .with_child(button)
        .with_child(output)

}