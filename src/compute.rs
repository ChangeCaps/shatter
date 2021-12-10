use crate::{Bindings, ComputePipelineDescriptor, Instance, PipelineLayoutDescriptor};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Dispatch {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

impl Dispatch {
    pub const fn new(x: u32, y: u32, z: u32) -> Self {
        Self { x, y, z }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct WorkGroupSize {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

impl WorkGroupSize {
    pub const fn new(x: u32, y: u32, z: u32) -> Self {
        Self { x, y, z }
    }
}

pub trait ComputeShader<'a> {
    type Bindings: Bindings;

    const SOURCE: &'static str;
    const ENTRY_POINT: &'static str;
}

pub struct ComputeShaderBuilder<'a, S: ComputeShader<'a>> {
    bindings: S::Bindings,
    encoder: Option<&'a mut wgpu::CommandEncoder>,
}

impl<'a, S: ComputeShader<'a>> ComputeShaderBuilder<'a, S> {
    #[inline]
    pub fn new(bindings: S::Bindings) -> Self {
        Self {
            bindings,
            encoder: None,
        }
    }

    #[inline]
    pub fn take_binding(self) -> S::Bindings {
        self.bindings
    }

    /// Set the command encoder for subsequent dispatches.
    ///
    /// # Note
    /// When the encoder is set, bindings must be *downloaded* manually.
    #[inline]
    pub fn encoder(&mut self, encoder: &'a mut wgpu::CommandEncoder) -> &mut Self {
        self.encoder = Some(encoder);
        self
    }

    /// Unsets the command encoder.
    ///
    /// This means that a command encoder will automatically be created
    /// on dispatch. Encoder is unset by default.
    #[inline]
    pub fn unset_encoder(&mut self) -> &mut Self {
        self.encoder = None;
        self
    }

    #[inline]
    pub fn dispatch(&mut self, dispatch: Dispatch) -> &mut Self {
        self.dispatch_multiple(&[dispatch]);
        self
    }

    #[inline]
    pub fn dispatch_multiple(&mut self, dispatches: &[Dispatch]) -> &mut Self {
        self.bindings.upload();

        let instance = Instance::global();

        let layout_descriptors = self.bindings.bind_group_layout_descriptors();
        let layouts = layout_descriptors
            .into_iter()
            .map(|desc| instance.get_bind_group_layout(desc))
            .collect::<Vec<_>>();

        let bind_group_descriptors = self.bindings.bind_group_descriptors(&layouts);
        let bind_group_ids = bind_group_descriptors
            .into_iter()
            .map(|desc| instance.get_bind_group(desc))
            .collect::<Vec<_>>();

        let bind_groups = bind_group_ids
            .iter()
            .map(|id| instance.bind_groups.get(id).unwrap())
            .collect::<Vec<_>>();

        let pipeline_layout_descriptor = PipelineLayoutDescriptor {
            bind_group_layouts: layouts,
            push_constant_ranges: Vec::new(),
        };

        let pipeline_layout = instance.get_pipeline_layout(pipeline_layout_descriptor);

        let shader_module = instance.get_shader_module(S::SOURCE);

        let compute_pipeline_descriptor = ComputePipelineDescriptor {
            layout: Some(pipeline_layout),
            module: shader_module,
            entry_point: S::ENTRY_POINT.into(),
        };

        let compute_pipeline_id = instance.get_compute_pipeline(compute_pipeline_descriptor);

        let compute_pipeline = instance
            .compute_pipelines
            .get(&compute_pipeline_id)
            .unwrap();

        let dispatch = |encoder: &mut wgpu::CommandEncoder| {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(&format!("shatter_compute_pass({})", S::ENTRY_POINT)),
            });

            compute_pass.set_pipeline(&compute_pipeline);

            for (i, bind_group) in bind_groups.iter().enumerate() {
                compute_pass.set_bind_group(i as u32, bind_group, &[]);
            }

            for dispatch in dispatches {
                compute_pass.dispatch(dispatch.x, dispatch.y, dispatch.z);
            }
        };

        if let Some(encoder) = &mut self.encoder {
            dispatch(encoder);
        } else {
            let mut encoder =
                instance
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some(&format!("shatter_command_encoder({})", S::ENTRY_POINT)),
                    });

            dispatch(&mut encoder);

            instance.queue.submit(std::iter::once(encoder.finish()));

            self.bindings.download();
        };

        self
    }
}
