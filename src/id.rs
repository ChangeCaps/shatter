use std::{
    hash::Hash,
    marker::PhantomData,
    ops::{Deref, DerefMut},
    sync::{
        atomic::{AtomicU32, AtomicU64, Ordering},
        Arc,
    },
};

use dashmap::DashMap;

pub type ShaderModuleId = Id<wgpu::ShaderModule>;
pub type BindGroupLayoutDescriptorId = Id<crate::BindGroupLayoutDescriptor>;
pub type BindGroupLayoutId = Id<wgpu::BindGroupLayout>;
pub type BindGroupDescriptorId = Id<crate::BindGroupDescriptor>;
pub type BindGroupId = Id<wgpu::BindGroup>;
pub type BufferId = Id<wgpu::Buffer>;
pub type SamplerId = Id<wgpu::Sampler>;
pub type TextureId = Id<wgpu::Texture>;
pub type TextureViewId = Id<wgpu::TextureView>;
pub type PipelineLayoutDescriptorId = Id<crate::PipelineLayoutDescriptor>;
pub type PipelineLayoutId = Id<wgpu::PipelineLayout>;
pub type ComputePipelineDescriptorId = Id<crate::ComputePipelineDescriptor>;
pub type ComputePipelineId = Id<wgpu::ComputePipeline>;
pub type RenderPipelineId = Id<wgpu::RenderPipeline>;

pub struct Id<T>(u64, Arc<AtomicU32>, PhantomData<fn() -> T>);

impl<T> Id<T> {
    pub(crate) fn ref_count(&self) -> u32 {
        self.1.load(Ordering::Acquire)
    }

    pub fn clone_untracked(&self) -> Self {
        Self(self.0, Arc::new(AtomicU32::new(0)), PhantomData)
    }
}

impl<T> Drop for Id<T> {
    fn drop(&mut self) {
        self.1.fetch_sub(1, Ordering::AcqRel);
    }
}

impl<T> Clone for Id<T> {
    fn clone(&self) -> Self {
        self.1.fetch_add(1, Ordering::AcqRel);

        Self(self.0, self.1.clone(), PhantomData)
    }
}

impl<T> std::fmt::Debug for Id<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Id({})", self.0)
    }
}

impl<T> PartialEq for Id<T> {
    fn eq(&self, other: &Self) -> bool {
        self.0.eq(&other.0)
    }
}

impl<T> Eq for Id<T> {}

impl<T> PartialOrd for Id<T> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.0.partial_cmp(&other.0)
    }
}

impl<T> Ord for Id<T> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.cmp(&other.0)
    }
}

impl<T> Hash for Id<T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.0.hash(state)
    }
}

pub struct IdMap<T> {
    map: DashMap<Id<T>, T>,
    next_id: AtomicU64,
}

impl<T> IdMap<T> {
    pub fn new() -> Self {
        Self {
            map: DashMap::new(),
            next_id: AtomicU64::new(0),
        }
    }

    pub fn next_id(&self) -> Id<T> {
        let id = self.next_id.fetch_add(1, Ordering::AcqRel);

        Id(id, Arc::new(AtomicU32::new(0)), PhantomData)
    }

    pub fn clean(&self) {
        self.map.retain(|id, _| id.ref_count() > 0)
    }
}

impl<T> Deref for IdMap<T> {
    type Target = DashMap<Id<T>, T>;

    fn deref(&self) -> &Self::Target {
        &self.map
    }
}

impl<T> DerefMut for IdMap<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.map
    }
}
