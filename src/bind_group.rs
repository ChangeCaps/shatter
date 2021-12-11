use std::num::NonZeroU64;

use crate::{BindGroupLayoutId, Buffer, BufferData, BufferId, SamplerId, TextureId};

pub use wgpu::{
    BindGroupLayoutEntry, BindingType, BufferBindingType, ShaderStages, StorageTextureAccess,
    TextureFormat, TextureSampleType, TextureViewDimension,
};

pub trait Binding<T: ?Sized> {
    fn binding_resource(&self) -> BindingResource;

    fn prepare(&self);

    fn read(&self);

    fn write(&mut self);
}

pub trait Bindings {
    fn bind_group_layout_descriptors(&self) -> Vec<BindGroupLayoutDescriptor>;

    fn bind_group_descriptors(&self, layouts: &[BindGroupLayoutId]) -> Vec<BindGroupDescriptor>;

    fn prepare(&self);

    fn read(&self);

    fn write(&mut self);
}

impl Bindings for () {
    fn bind_group_layout_descriptors(&self) -> Vec<BindGroupLayoutDescriptor> {
        Vec::new()
    }

    fn bind_group_descriptors(&self, _: &[BindGroupLayoutId]) -> Vec<BindGroupDescriptor> {
        Vec::new()
    }

    fn prepare(&self) {}

    fn read(&self) {}

    fn write(&mut self) {}
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
    TextureView(TextureId),
    TextureViewArray(Vec<TextureId>),
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
