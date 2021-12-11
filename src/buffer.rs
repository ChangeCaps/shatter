use std::{
    marker::PhantomData,
    ops::{Deref, DerefMut},
    ptr::NonNull,
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Mutex,
    },
};

use crate::{Binding, BindingResource, BufferBinding, BufferId, Instance};

/// Allows a struct to reside inside of a [`Buffer`].
///
/// # Safety
/// * It **must** be safe to cast the struct to a slice of bytes.
/// * Alignment must be safe.
/// * size **must** always return a valid size for ptr.
pub unsafe trait BufferData {
    type State;

    fn init() -> Self::State;

    fn size(state: &Self::State) -> usize;

    /// Allocates self and returns a pointer.
    unsafe fn alloc() -> NonNull<u8>;
    /// Deallocates self from a NonNull pointer.
    unsafe fn dealloc(ptr: NonNull<u8>, state: &Self::State);

    unsafe fn as_ptr(ptr: NonNull<u8>, state: &Self::State) -> *mut Self;
}

/// Allows a struct to
pub unsafe trait BufferVec: BufferData {
    type Item;

    fn len(state: &Self::State) -> usize;

    unsafe fn grow(ptr: &mut NonNull<u8>, state: &mut Self::State);
    unsafe fn push(ptr: &mut NonNull<u8>, state: &mut Self::State, item: Self::Item);
    unsafe fn pop(ptr: NonNull<u8>, state: &mut Self::State) -> Option<Self::Item>;
}

pub struct Buffer<T: BufferData + ?Sized> {
    value: NonNull<u8>,
    state: T::State,
    id: Mutex<BufferId>,
    buffer_size: AtomicU64,
    needs_download: AtomicBool,
    marker: PhantomData<T>,
}

impl<T: BufferData + ?Sized> Binding<T> for Buffer<T> {
    fn binding_resource(&self) -> BindingResource {
        BindingResource::Buffer(BufferBinding {
            buffer: self.id(),
            offset: 0,
            size: None,
        })
    }

    fn prepare(&self) {
        self.resize_buffer();
    }

    fn read(&self) {
        self.upload();
    }

    fn write(&mut self) {
        self.mark_needs_download();
    }
}

impl<T: BufferData + ?Sized> Default for Buffer<T> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl<T: BufferData + ?Sized> Deref for Buffer<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.download();

        let ptr = unsafe { T::as_ptr(self.value, &self.state) };

        unsafe { &*ptr }
    }
}

impl<T: BufferData + ?Sized> DerefMut for Buffer<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.download();

        let ptr = unsafe { T::as_ptr(self.value, &self.state) };

        unsafe { &mut *ptr }
    }
}

impl<T: BufferData + ?Sized> Buffer<T> {
    #[inline]
    pub fn new() -> Self {
        let value = unsafe { T::alloc() };
        let state = T::init();

        let size = T::size(&state).max(4) as u64;

        let device = &Instance::global().device;

        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("shatter_buffer"),
            size,
            usage: wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::UNIFORM,
            mapped_at_creation: false,
        });

        let id = Instance::global().buffers.next_id();
        Instance::global().buffers.insert(id.clone(), buffer);

        Self {
            value,
            state,
            id: Mutex::new(id),
            buffer_size: AtomicU64::new(size),
            needs_download: AtomicBool::new(false),
            marker: PhantomData,
        }
    }

    #[inline]
    pub fn resize_buffer(&self) {
        if self.needs_download() {
            self.download();
        }

        let size = T::size(&self.state).max(4) as u64;

        if self.buffer_size.load(Ordering::Acquire) < size {
            let device = &Instance::global().device;

            let buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("shatter_buffer"),
                size,
                usage: wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::UNIFORM,
                mapped_at_creation: false,
            });

            let id = Instance::global().buffers.next_id();
            Instance::global().buffers.insert(id.clone(), buffer);
            Instance::global().buffers.clean();

            *self.id.lock().unwrap() = id;
            self.buffer_size.store(size, Ordering::Release);
        }
    }

    #[inline]
    pub fn needs_download(&self) -> bool {
        self.needs_download.load(Ordering::Acquire)
    }

    #[inline]
    pub fn mark_needs_download(&mut self) {
        self.needs_download.store(true, Ordering::Release);
    }

    #[inline]
    pub fn upload(&self) {
        // if we haven't downloaded, there is no need to upload
        // we know that the data hasn't changed since both reading
        // and writing requires downloading
        if self.needs_download() {
            return;
        }

        self.resize_buffer();

        let size = T::size(&self.state);

        if size == 0 {
            return;
        }

        // SAFETY:
        // * BufferData ensures that size is valid.
        let slice = unsafe { std::slice::from_raw_parts(self.value.as_ptr(), size) };

        let id = self.id.lock().unwrap();
        let buffer = Instance::global().buffers.get(&id).unwrap();
        Instance::global().queue.write_buffer(&buffer, 0, slice);
    }

    #[inline]
    pub fn download(&self) {
        // if we don't need to download then don't
        if !self.needs_download.swap(false, Ordering::AcqRel) {
            return;
        }

        let device = &Instance::global().device;

        let size = T::size(&self.state);

        if size == 0 {
            return;
        } else if size < 4 {
            panic!("wtf");
        }

        let size = size.max(4) as u64;

        // TODO: cache the staging buffer
        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("shatter_buffer"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let id = self.id.lock().unwrap();
        let buffer = Instance::global().buffers.get(&id).unwrap();

        // copy data into the staging buffer
        let mut encoder = device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(&buffer, 0, &staging_buffer, 0, size);
        Instance::global()
            .queue
            .submit(std::iter::once(encoder.finish()));

        // map the staging buffer
        let future = staging_buffer.slice(..).map_async(wgpu::MapMode::Read);
        Instance::global().device.poll(wgpu::Maintain::Wait);
        pollster::block_on(future).unwrap();

        // get a mutable slice of the data
        let slice: &[u8] = &staging_buffer.slice(..).get_mapped_range();

        assert_eq!(slice.len(), size as usize);

        // SAFETY:
        // * BufferData ensures that size is valid.
        // * a mutable reference is needed to mark needs_download.
        //   any read or write to self.value requires a download.
        //   download marks itself as not needing download.
        //   therefore it's impossible to get here while a reference
        //   to self.value is held.
        // * self.value doesn't overlap with slice
        // * align of u8 is 1 so pointers will always be properly aligned.
        // * we have just asserted that the length if slice is equal to size.
        unsafe {
            std::ptr::copy_nonoverlapping(
                slice as *const [u8] as *const u8,
                self.value.as_ptr(),
                size as usize,
            )
        };
    }

    #[inline]
    pub fn id(&self) -> BufferId {
        self.id.lock().unwrap().clone()
    }
}

impl<T: BufferVec + ?Sized> Buffer<T> {
    #[inline]
    pub fn len(&self) -> usize {
        T::len(&self.state)
    }

    #[inline]
    pub fn push(&mut self, item: T::Item) {
        unsafe { T::push(&mut self.value, &mut self.state, item) };
    }

    #[inline]
    pub fn pop(&mut self) -> Option<T::Item> {
        unsafe { T::pop(self.value, &mut self.state) }
    }
}

impl<T: BufferData + ?Sized> Drop for Buffer<T> {
    #[inline]
    fn drop(&mut self) {
        unsafe { std::ptr::drop_in_place(T::as_ptr(self.value, &self.state)) };
        unsafe { T::dealloc(self.value, &self.state) };
    }
}
