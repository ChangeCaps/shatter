#![deny(unsafe_op_in_unsafe_fn)]

mod bind_group;
mod buffer;
pub mod color;
mod compute;
mod id;
mod instance;
mod math;
mod pipeline;
mod render;
mod texture;

pub use bind_group::*;
pub use buffer::*;
#[doc(hidden)]
pub use color::*;
pub use compute::*;
pub use id::*;
pub use instance::*;
#[doc(hidden)]
pub use math::*;
pub use pipeline::*;
pub use render::*;
pub use shatter_macro::*;
pub use texture::*;
#[doc(hidden)]
pub use texture_format::*;

#[doc(hidden)]
pub use wgpu;
