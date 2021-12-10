use std::borrow::Cow;

use crate::{BindGroupLayoutId, PipelineLayoutId, ShaderModuleId};

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct PipelineLayoutDescriptor {
    pub bind_group_layouts: Vec<BindGroupLayoutId>,
    pub push_constant_ranges: Vec<wgpu::PushConstantRange>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ComputePipelineDescriptor {
    pub layout: Option<PipelineLayoutId>,
    pub module: ShaderModuleId,
    pub entry_point: Cow<'static, str>,
}
