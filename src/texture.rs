use std::{
    marker::PhantomData,
    num::NonZeroU32,
    ops::{Index, IndexMut},
    sync::atomic::{AtomicBool, Ordering},
};

use crate::{Binding, BindingResource, Instance, TextureId};

pub mod texture_sample_type {
    pub struct Float<const FILTERABLE: bool>;
    pub struct Depth;
    pub struct Sint;
    pub struct Uint;
}

pub mod texture_view_dimension {
    use std::{
        alloc::{self, Layout},
        mem,
        ptr::{self, NonNull},
        slice,
        sync::atomic::AtomicPtr,
    };

    use super::*;

    pub struct TextureStorageData<Data> {
        data: AtomicPtr<Data>,
        layout: Layout,
        marker: PhantomData<Data>,
    }

    impl<Data> TextureStorageData<Data> {
        pub const unsafe fn new(layout: Layout) -> Self {
            Self {
                data: AtomicPtr::new(ptr::null_mut()),
                layout,
                marker: PhantomData,
            }
        }

        pub fn allocate(&self) {
            if !self.data.load(Ordering::Acquire).is_null() {
                return;
            }

            let ptr = unsafe { alloc::alloc(self.layout) };

            unsafe { ptr::write_bytes(ptr, 0, self.layout.size()) };

            let ptr = match NonNull::new(ptr) {
                Some(ptr) => ptr,
                None => alloc::handle_alloc_error(self.layout),
            };

            self.data.store(ptr.cast().as_ptr(), Ordering::Release);
        }

        pub fn size(&self) -> usize {
            self.layout.size()
        }

        pub fn ptr(&self) -> *mut Data {
            self.allocate();

            self.data.load(Ordering::Acquire)
        }

        pub fn index(&self, extent: wgpu::Extent3d, x: usize, y: usize, z: usize) -> *mut Data
        where
            Data: TextureData,
        {
            assert!(
                x < extent.width as usize
                    && y < extent.height as usize
                    && z < extent.depth_or_array_layers as usize
            );

            let bytes_per_row = bytes_per_row::<Data>(extent.width as usize);
            let rows_per_image = extent.height as usize;
            let bytes_per_image = rows_per_image * bytes_per_row;

            let image_ptr = unsafe { (self.ptr() as *mut u8).add(z * bytes_per_image) };
            let row_ptr = unsafe { image_ptr.add(y * bytes_per_row) as *mut Data };
            let ptr = unsafe { row_ptr.add(x) };

            ptr
        }

        pub fn bytes(&self) -> &[u8] {
            self.allocate();

            unsafe { slice::from_raw_parts(self.ptr() as *const u8, self.layout.size()) }
        }
    }

    impl<Data> Drop for TextureStorageData<Data> {
        fn drop(&mut self) {
            assert!(!mem::needs_drop::<Data>());

            let ptr = self.ptr();
            if self.layout.size() > 0 && !ptr.is_null() {
                unsafe { alloc::dealloc(ptr as *mut u8, self.layout) };
            }
        }
    }

    pub struct TextureStorageD1<Data: TextureData> {
        width: usize,
        pub data: TextureStorageData<Data>,
    }

    impl<Data: TextureData> TextureStorageD1<Data> {
        pub fn new(width: usize) -> Self {
            let layout = Layout::array::<Data>(width).unwrap();

            Self {
                width,
                data: unsafe { TextureStorageData::new(layout) },
            }
        }
    }

    unsafe impl<Data: TextureData> TextureStorage for TextureStorageD1<Data> {
        fn extent(&self) -> wgpu::Extent3d {
            wgpu::Extent3d {
                width: self.width as u32,
                height: 1,
                depth_or_array_layers: 1,
            }
        }

        fn bytes_per_row(&self) -> Option<NonZeroU32> {
            NonZeroU32::new(bytes_per_row::<Data>(self.width) as u32)
        }

        fn size(&self) -> usize {
            self.data.size()
        }

        fn ptr(&self) -> *mut u8 {
            self.data.ptr() as *mut u8
        }

        fn bytes(&self) -> &[u8] {
            self.data.bytes()
        }
    }

    pub struct D1;

    impl<Format: TextureFormat> TextureDimension<Format> for D1 {
        type Storage = TextureStorageD1<Format::Data>;
    }

    fn bytes_per_row<Data: TextureData>(width: usize) -> usize {
        let row_layout = Layout::array::<Data>(width)
            .unwrap()
            .align_to(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT as usize)
            .unwrap()
            .pad_to_align();

        row_layout.size()
    }

    pub struct TextureStorageD2<Data: TextureData> {
        width: usize,
        height: usize,
        pub data: TextureStorageData<Data>,
    }

    impl<Data: TextureData> TextureStorageD2<Data> {
        pub fn new(width: usize, height: usize) -> Self {
            assert!(mem::size_of::<Data>() <= wgpu::COPY_BYTES_PER_ROW_ALIGNMENT as usize);

            let layout = Layout::from_size_align(
                bytes_per_row::<Data>(width) * height,
                mem::align_of::<Data>(),
            )
            .unwrap();

            Self {
                width,
                height,
                data: unsafe { TextureStorageData::new(layout) },
            }
        }
    }

    unsafe impl<Data: TextureData> TextureStorage for TextureStorageD2<Data> {
        fn extent(&self) -> wgpu::Extent3d {
            wgpu::Extent3d {
                width: self.width as u32,
                height: self.height as u32,
                depth_or_array_layers: 1,
            }
        }

        fn bytes_per_row(&self) -> Option<NonZeroU32> {
            NonZeroU32::new(bytes_per_row::<Data>(self.width) as u32)
        }

        fn size(&self) -> usize {
            self.data.size()
        }

        fn ptr(&self) -> *mut u8 {
            self.data.ptr() as *mut u8
        }

        fn bytes(&self) -> &[u8] {
            self.data.bytes()
        }
    }

    pub struct D2;

    impl<Format: TextureFormat> TextureDimension<Format> for D2 {
        type Storage = TextureStorageD2<Format::Data>;
    }

    pub struct D2Array;
    pub struct Cube;
    pub struct CubeArray;
    pub struct D3;
}

pub mod texel_format {
    macro_rules! texel_format {
        ($name:ident) => {
            pub struct $name;
        };
    }

    texel_format!(Rgba8Unorm);
    texel_format!(Rgba8Snorm);
    texel_format!(Rgba8Uint);
    texel_format!(Rgba8Sint);
    texel_format!(Rgba16Uint);
    texel_format!(Rgba16Sint);
    texel_format!(Rgba16Float);
    texel_format!(R32Uint);
    texel_format!(R32Sint);
    texel_format!(R32Float);
    texel_format!(Rg32Uint);
    texel_format!(Rg32Sint);
    texel_format!(Rg32Float);
    texel_format!(Rgba32Uint);
    texel_format!(Rgba32Sint);
    texel_format!(Rgba32Float);
}

pub mod texture_format {
    use crate::color::*;

    macro_rules! texture_format {
        ($name:ident, $sample_type:ident $(<$filterable:literal>)?, $data:path $(, $texel_format:ident)?) => {
            #[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
            pub struct $name;

            impl super::Sampled for $name {
                type SampleType = super::texture_sample_type::$sample_type$(<$filterable>)?;
            }

            $(
                impl super::Stored for $name {
                    type TexelFormat = super::texel_format::$texel_format;
                }
            )?

			impl super::TextureFormat for $name {
				type Data = $data;

				fn format(&self) -> wgpu::TextureFormat {
					wgpu::TextureFormat::$name
				}
			}
        };
    }

    texture_format!(Rgba8UnormSrgb, Float<true>, Rgba8U);
    texture_format!(Rgba8Unorm, Float<true>, Rgba8U, Rgba8Unorm);
    texture_format!(Rgba8Snorm, Float<true>, Rgba8I, Rgba8Snorm);
    texture_format!(Rgba8Uint, Uint, Rgba8U, Rgba8Uint);
    texture_format!(Rgba8Sint, Sint, Rgba8I, Rgba8Sint);
    texture_format!(Rgba16Uint, Uint, Rgba16U, Rgba16Uint);
    texture_format!(Rgba16Sint, Sint, Rgba16I, Rgba16Sint);
    texture_format!(Rgba16Float, Float<false>, Rgba16U, Rgba16Float);
    texture_format!(R32Uint, Uint, R32U, R32Uint);
    texture_format!(R32Sint, Sint, R32I, R32Sint);
    texture_format!(R32Float, Float<false>, R32, R32Float);
    texture_format!(Rg32Uint, Uint, Rg32U, Rg32Uint);
    texture_format!(Rg32Sint, Sint, Rg32I, Rg32Sint);
    texture_format!(Rg32Float, Float<false>, Rg32, Rg32Float);
    texture_format!(Rgba32Uint, Uint, Rgba32U, Rgba32Uint);
    texture_format!(Rgba32Sint, Sint, Rgba32I, Rgba32Sint);
    texture_format!(Rgba32Float, Float<false>, Rgba32, Rgba32Float);
}

pub trait Sampled {
    type SampleType;
}

pub trait Stored {
    type TexelFormat;
}

pub unsafe trait TextureStorage {
    fn extent(&self) -> wgpu::Extent3d;

    fn bytes_per_row(&self) -> Option<NonZeroU32>;

    fn size(&self) -> usize;

    fn ptr(&self) -> *mut u8;

    fn bytes(&self) -> &[u8];
}

pub trait TextureDimension<Format: TextureFormat> {
    type Storage: TextureStorage;
}

pub unsafe trait TextureData: Copy {}

pub trait TextureFormat {
    type Data: TextureData;

    fn format(&self) -> wgpu::TextureFormat;
}

pub struct TextureBinding<SampleType, ViewDimension, const MULTISAMPLED: bool>(
    PhantomData<(SampleType, ViewDimension)>,
);

pub struct StorageTextureBinding<TexelFormat, ViewDimension>(
    PhantomData<(TexelFormat, ViewDimension)>,
);

pub struct Texture<Format, Dimension, const MULTISAMPLED: bool>
where
    Format: TextureFormat,
    Dimension: TextureDimension<Format>,
{
    format: Format,
    storage: Dimension::Storage,
    id: TextureId,
    needs_upload: AtomicBool,
    needs_download: AtomicBool,
}

impl<Format, Dimension, const MULTISAMPLED: bool> Texture<Format, Dimension, MULTISAMPLED>
where
    Format: TextureFormat,
    Dimension: TextureDimension<Format>,
{
    pub fn needs_upload(&self) -> bool {
        self.needs_upload.load(Ordering::Acquire)
    }

    pub fn mark_needs_upload(&self) {
        self.needs_upload.store(true, Ordering::Release);
    }

    pub fn needs_download(&self) -> bool {
        self.needs_download.load(Ordering::Acquire)
    }

    pub fn mark_needs_download(&mut self) {
        self.needs_download.store(true, Ordering::Release);
    }

    pub fn wgpu_format(&self) -> wgpu::TextureFormat {
        self.format.format()
    }

    pub fn texture_id(&self) -> &TextureId {
        &self.id
    }

    pub fn bytes(&self) -> &[u8] {
        self.download();

        self.storage.bytes()
    }

    pub fn width(&self) -> usize {
        self.storage.extent().width as usize
    }

    pub fn height(&self) -> usize {
        self.storage.extent().height as usize
    }

    pub fn depth(&self) -> usize {
        self.storage.extent().depth_or_array_layers as usize
    }

    pub fn upload(&self) {
        if !self.needs_upload.swap(false, Ordering::AcqRel) {
            return;
        }

        let instance = Instance::global();

        let size = self.storage.size();

        if size == 0 {
            return;
        }

        let texture = instance.textures.get(&self.id).unwrap();

        instance.queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            self.storage.bytes(),
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: self.storage.bytes_per_row(),
                rows_per_image: None,
            },
            self.storage.extent(),
        );
    }

    pub fn download(&self) {
        if !self.needs_download.swap(false, Ordering::AcqRel) {
            return;
        }

        let instance = Instance::global();

        let size = self.storage.size();

        if size == 0 {
            return;
        }

        let size = size.max(4) as u64;

        let staging_buffer = instance.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("shatter_staging_buffer"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let texture = instance.textures.get(&self.id).unwrap();

        let mut encoder = instance.device.create_command_encoder(&Default::default());
        encoder.copy_texture_to_buffer(
            wgpu::ImageCopyTexture {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::ImageCopyBuffer {
                buffer: &staging_buffer,
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: self.storage.bytes_per_row(),
                    rows_per_image: None,
                },
            },
            self.storage.extent(),
        );
        instance.queue.submit(std::iter::once(encoder.finish()));

        let future = staging_buffer.slice(..).map_async(wgpu::MapMode::Read);
        instance.device.poll(wgpu::Maintain::Wait);
        pollster::block_on(future).unwrap();

        let slice: &[u8] = &staging_buffer.slice(..).get_mapped_range();

        assert_eq!(slice.len(), size as usize);

        unsafe {
            std::ptr::copy_nonoverlapping(
                slice as *const [u8] as *const u8,
                self.storage.ptr(),
                size as usize,
            )
        };
    }
}

impl<Format, Dimension, const MULTISAMPLED: bool>
    Binding<TextureBinding<Format::SampleType, Dimension, MULTISAMPLED>>
    for Texture<Format, Dimension, MULTISAMPLED>
where
    Format: TextureFormat + Sampled,
    Dimension: TextureDimension<Format>,
{
    fn binding_resource(&self) -> BindingResource {
        BindingResource::TextureView(self.id.clone())
    }

    fn prepare(&self) {}

    fn read(&self) {
        self.upload();
    }

    fn write(&mut self) {
        self.mark_needs_download();
    }
}

impl<Format, Dimension, const MULTISAMPLED: bool>
    Binding<StorageTextureBinding<Format::TexelFormat, Dimension>>
    for Texture<Format, Dimension, MULTISAMPLED>
where
    Format: TextureFormat + Stored,
    Dimension: TextureDimension<Format>,
{
    fn binding_resource(&self) -> BindingResource {
        BindingResource::TextureView(self.id.clone())
    }

    fn prepare(&self) {}

    fn read(&self) {
        self.upload();
    }

    fn write(&mut self) {
        self.mark_needs_download();
    }
}

pub type Texture2d<Format> = Texture<Format, texture_view_dimension::D2, false>;

impl<Format: TextureFormat + Default> Texture2d<Format> {
    pub fn new(width: usize, height: usize) -> Self {
        let format = Format::default();

        let instance = Instance::global();

        let texture = instance.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("shatter_texture"),
            size: wgpu::Extent3d {
                width: width as u32,
                height: height as u32,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: format.format(),
            usage: wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::RENDER_ATTACHMENT,
        });

        let id = instance.textures.next_id();
        instance.textures.insert(id.clone(), texture);

        Self {
            format,
            storage: texture_view_dimension::TextureStorageD2::new(width, height),
            id,
            needs_upload: AtomicBool::new(false),
            needs_download: AtomicBool::new(false),
        }
    }
}

impl<Format: TextureFormat + Default> Index<(usize, usize)> for Texture2d<Format> {
    type Output = Format::Data;

    fn index(&self, (x, y): (usize, usize)) -> &Self::Output {
        self.download();

        unsafe { &*self.storage.data.index(self.storage.extent(), x, y, 0) }
    }
}

impl<Format: TextureFormat + Default> IndexMut<(usize, usize)> for Texture2d<Format> {
    fn index_mut(&mut self, (x, y): (usize, usize)) -> &mut Self::Output {
        self.download();

        self.mark_needs_upload();

        unsafe { &mut *self.storage.data.index(self.storage.extent(), x, y, 0) }
    }
}
