#![no_std]

pub mod console_contract;
pub mod console_history;
pub mod editor;
pub mod model_contract;
pub mod repl_contract;
pub mod status_bar;

#[cfg(test)]
extern crate std;
