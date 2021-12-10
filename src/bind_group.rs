use std::num::NonZeroU64;

use crate::{BindGroupLayoutId, Buffer, BufferData, BufferId, SamplerId, TextureViewId};

pub use wgpu::{
    BindGroupLayoutEntry, BindingType, BufferBindingType, ShaderStages, StorageTextureAccess,
    TextureFormat, TextureSampleType, TextureViewDimension,
};

pub trait Bindings {
    fn bind_group_layout_descriptors(&self) -> Vec<BindGroupLayoutDescriptor>;

    fn bind_group_descriptors(&self, layouts: &[BindGroupLayoutId]) -> Vec<BindGroupDescriptor>;

    fn upload(&mut self);

    fn download(&mut self);
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct BindGroupLayoutDescriptor {
    pub entries: Vec<wgpu::BindGroupLayoutEntry>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct BufferBinding {
    pub buffer: BufferId,
    pub offset: u64,
    pub size: Option<NonZeroU64>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum BindingResource {
    Buffer(BufferBinding),
    BufferArray(Vec<BufferBinding>),
    Sampler(SamplerId),
    TextureView(TextureViewId),
    TextureViewArray(Vec<TextureViewId>),
}

impl<T: BufferData + ?Sized> From<&Buffer<T>> for BindingResource {
    fn from(buffer: &Buffer<T>) -> Self {
        BindingResource::Buffer(BufferBinding {
            buffer: buffer.id(),
            offset: 0,
            size: None,
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct BindGroupEntry {
    pub binding: u32,
    pub resource: BindingResource,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct BindGroupDescriptor {
    pub layout: BindGroupLayoutId,
    pub entries: Vec<BindGroupEntry>,
}
