#![deny(unsafe_op_in_unsafe_fn)]

mod bind_group;
mod buffer;
mod compute;
mod entry_point;
mod id;
mod instance;
mod math;
mod pipeline;

pub use bind_group::*;
pub use buffer::*;
pub use compute::*;
pub use entry_point::*;
pub use id::*;
pub use instance::*;
pub use math::*;
pub use pipeline::*;
pub use shatter_macro::*;

pub use wgpu;
