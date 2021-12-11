use std::borrow::Cow;

use dashmap::{mapref::one::Ref, DashMap};
use once_cell::sync::OnceCell;
use wgpu::Backends;

use crate::{
    BindGroupId, BindGroupLayoutId, BufferId, ComputePipelineId, IdMap, PipelineLayoutId,
    SamplerId, ShaderModuleId,
};

pub static GLOBAL_INSTANCE: OnceCell<Instance> = OnceCell::new();

#[derive(Default)]
pub struct InstanceDescriptor {
    pub features: wgpu::Features,
    pub limits: wgpu::Limits,
}

pub struct Instance {
    pub instance: wgpu::Instance,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub buffers: IdMap<wgpu::Buffer>,
    pub textures: IdMap<wgpu::Texture>,
    pub samplers: IdMap<wgpu::Sampler>,
    pub shader_module_sources: DashMap<Cow<'static, str>, ShaderModuleId>,
    pub shader_modules: IdMap<wgpu::ShaderModule>,
    pub bind_group_layout_descriptors: DashMap<crate::BindGroupLayoutDescriptor, BindGroupLayoutId>,
    pub bind_group_layouts: IdMap<wgpu::BindGroupLayout>,
    pub bind_group_descriptors: DashMap<crate::BindGroupDescriptor, BindGroupId>,
    pub bind_groups: IdMap<wgpu::BindGroup>,
    pub pipeline_layout_descriptors: DashMap<crate::PipelineLayoutDescriptor, PipelineLayoutId>,
    pub pipeline_layouts: IdMap<wgpu::PipelineLayout>,
    pub compute_pipeline_descriptors: DashMap<crate::ComputePipelineDescriptor, ComputePipelineId>,
    pub render_pipelines: IdMap<wgpu::RenderPipeline>,
    pub compute_pipelines: IdMap<wgpu::ComputePipeline>,
}

impl Instance {
    pub fn global<'a>() -> &'a Self {
        GLOBAL_INSTANCE.get_or_init(|| {
            pollster::block_on(Self::initialize(&InstanceDescriptor::default())).unwrap()
        })
    }

    pub fn init(desc: &InstanceDescriptor) {
        GLOBAL_INSTANCE.get_or_init(|| pollster::block_on(Self::initialize(desc)).unwrap());
    }

    pub async fn initialize(desc: &InstanceDescriptor) -> anyhow::Result<Self> {
        let instance = wgpu::Instance::new(Backends::all());

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
            })
            .await
            .unwrap();

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("shatter_default_device"),
                    features: desc.features,
                    limits: desc.limits.clone(),
                },
                None,
            )
            .await?;

        Ok(Self {
            instance,
            device,
            queue,
            buffers: IdMap::new(),
            textures: IdMap::new(),
            samplers: IdMap::new(),
            shader_module_sources: DashMap::new(),
            shader_modules: IdMap::new(),
            bind_group_layout_descriptors: DashMap::new(),
            bind_group_layouts: IdMap::new(),
            bind_group_descriptors: DashMap::new(),
            bind_groups: IdMap::new(),
            pipeline_layout_descriptors: DashMap::new(),
            pipeline_layouts: IdMap::new(),
            compute_pipeline_descriptors: DashMap::new(),
            compute_pipelines: IdMap::new(),
            render_pipelines: IdMap::new(),
        })
    }

    pub fn get_bind_group_layout(
        &self,
        desc: crate::BindGroupLayoutDescriptor,
    ) -> BindGroupLayoutId {
        if let Some(id) = self.bind_group_layout_descriptors.get(&desc) {
            return id.clone();
        }

        let wgpu_desc = wgpu::BindGroupLayoutDescriptor {
            label: Some("shatter_bind_group_layout"),
            entries: &desc.entries,
        };

        let bind_group = self.device.create_bind_group_layout(&wgpu_desc);

        let id = self.bind_group_layouts.next_id();

        self.bind_group_layout_descriptors
            .insert(desc, id.clone_untracked());
        self.bind_group_layouts.insert(id.clone(), bind_group);

        id
    }

    pub fn get_bind_group(&self, desc: crate::BindGroupDescriptor) -> BindGroupId {
        if let Some(id) = self.bind_group_descriptors.get(&desc) {
            return id.clone();
        }

        let layout = self.bind_group_layouts.get(&desc.layout).unwrap();

        #[allow(unused)]
        enum RefResource<'a> {
            Buffer(Ref<'a, BufferId, wgpu::Buffer>, &'a crate::BufferBinding),
            BufferArray(Vec<(Ref<'a, BufferId, wgpu::Buffer>, &'a crate::BufferBinding)>),
            Sampler(Ref<'a, SamplerId, wgpu::Sampler>),
            TextureView(wgpu::TextureView),
            TextureViewArray(Vec<wgpu::TextureView>),
        }

        let resources = desc
            .entries
            .iter()
            .map(|entry| match entry.resource {
                crate::BindingResource::Buffer(ref binding) => {
                    RefResource::Buffer(self.buffers.get(&binding.buffer).unwrap(), binding)
                }
                crate::BindingResource::TextureView(ref id) => {
                    let texture = self.textures.get(id).unwrap();

                    RefResource::TextureView(texture.create_view(&Default::default()))
                }
                _ => unimplemented!(),
            })
            .collect::<Vec<_>>();

        let entries = desc
            .entries
            .iter()
            .zip(&resources)
            .map(|(entry, resource)| {
                let resource = match resource {
                    RefResource::Buffer(buffer, binding) => {
                        wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: &buffer,
                            offset: binding.offset,
                            size: binding.size,
                        })
                    }
                    RefResource::TextureView(view) => wgpu::BindingResource::TextureView(view),
                    _ => unimplemented!(),
                };

                wgpu::BindGroupEntry {
                    binding: entry.binding,
                    resource,
                }
            })
            .collect::<Vec<_>>();

        let wgpu_desc = wgpu::BindGroupDescriptor {
            label: Some("shatter_bind_group"),
            layout: &layout,
            entries: &entries,
        };

        let bind_group = self.device.create_bind_group(&wgpu_desc);

        drop(wgpu_desc);
        drop(resources);

        let id = self.bind_groups.next_id();

        self.bind_group_descriptors
            .insert(desc, id.clone_untracked());
        self.bind_groups.insert(id.clone(), bind_group);

        id
    }

    pub fn get_shader_module(&self, source: impl Into<Cow<'static, str>>) -> ShaderModuleId {
        let source = source.into();

        if let Some(id) = self.shader_module_sources.get(&source) {
            return id.clone();
        }

        let wgpu_desc = wgpu::ShaderModuleDescriptor {
            label: Some("shatter_shader_module"),
            source: wgpu::ShaderSource::Wgsl(source.clone()),
        };

        let shader_module = self.device.create_shader_module(&wgpu_desc);

        let id = self.shader_modules.next_id();

        self.shader_module_sources
            .insert(source, id.clone_untracked());
        self.shader_modules.insert(id.clone(), shader_module);

        id
    }

    pub fn get_pipeline_layout(&self, desc: crate::PipelineLayoutDescriptor) -> PipelineLayoutId {
        if let Some(id) = self.pipeline_layout_descriptors.get(&desc) {
            return id.clone();
        }

        let refs = desc
            .bind_group_layouts
            .iter()
            .map(|layout| self.bind_group_layouts.get(layout).unwrap())
            .collect::<Vec<_>>();

        let bind_group_layouts = refs.iter().map(|layout| &**layout).collect::<Vec<_>>();

        let wgpu_desc = wgpu::PipelineLayoutDescriptor {
            label: Some("shatter_pipeline_layout"),
            bind_group_layouts: &bind_group_layouts,
            push_constant_ranges: &desc.push_constant_ranges,
        };

        let pipeline_layout = self.device.create_pipeline_layout(&wgpu_desc);

        let id = self.pipeline_layouts.next_id();

        self.pipeline_layout_descriptors
            .insert(desc, id.clone_untracked());
        self.pipeline_layouts.insert(id.clone(), pipeline_layout);

        id
    }

    pub fn get_compute_pipeline(
        &self,
        desc: crate::ComputePipelineDescriptor,
    ) -> ComputePipelineId {
        if let Some(id) = self.compute_pipeline_descriptors.get(&desc) {
            return id.clone();
        }

        let layout = desc
            .layout
            .as_ref()
            .map(|id| self.pipeline_layouts.get(id).unwrap());
        let module = &*self.shader_modules.get(&desc.module).unwrap();

        let wgpu_desc = wgpu::ComputePipelineDescriptor {
            label: Some("shatter_compute_pipeline_layout"),
            layout: layout.as_ref().map(|layout| &**layout),
            module,
            entry_point: desc.entry_point.as_ref(),
        };

        let compute_pipeline = self.device.create_compute_pipeline(&wgpu_desc);

        let id = self.compute_pipelines.next_id();

        self.compute_pipeline_descriptors
            .insert(desc, id.clone_untracked());
        self.compute_pipelines.insert(id.clone(), compute_pipeline);

        id
    }
}
