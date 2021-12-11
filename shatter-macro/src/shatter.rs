use std::collections::HashMap;

use crate::wgsl::{Wgsl, WgslResult};
use naga::{
    proc::TypeResolution,
    valid::{
        Capabilities, ExpressionError, FunctionError, FunctionInfo, GlobalUse, ModuleInfo,
        ValidationError, ValidationFlags, Validator,
    },
    ArraySize, Constant, ConstantInner, EntryPoint, Handle, ImageClass, ImageDimension, Module,
    ScalarKind, ScalarValue, ShaderStage, StorageAccess, StorageClass, StorageFormat, Type,
    TypeInner, VectorSize,
};
use proc_macro2::{Ident, Span, TokenStream};
use proc_macro_error::{Diagnostic, Level};
use quote::quote;

fn expression_error_span(_module: &Module, err: &ExpressionError) -> Option<naga::Span> {
    match err {
        _ => return None,
    }
}

fn validation_error_span(module: &Module, err: &ValidationError) -> Option<naga::Span> {
    Some(match err {
        ValidationError::Layouter(ty) => module.types.get_span(ty.0),
        &ValidationError::Type { handle, .. } => module.types.get_span(handle),
        &ValidationError::Constant { handle, .. } => module.constants.get_span(handle),
        &ValidationError::GlobalVariable { handle, .. } => module.global_variables.get_span(handle),
        &ValidationError::Function {
            handle: func,
            ref error,
            ..
        } => match error {
            &FunctionError::Expression { handle, ref error } => {
                match expression_error_span(module, error) {
                    Some(span) => span,
                    None => module.functions[func].expressions.get_span(handle),
                }
            }
            _ => module.functions.get_span(func),
        },
        _ => return None,
    })
}

pub fn shatter(wgsl: &Wgsl) -> proc_macro::TokenStream {
    let module = naga::front::wgsl::parse_str(&wgsl.source).wgsl_unwrap(wgsl);

    let mut validator = Validator::new(ValidationFlags::all(), Capabilities::all());
    let info = validator.validate(&module).unwrap_or_else(|err| {
        let span = if let Some(span) = validation_error_span(&module, &err) {
            *wgsl.get_span(span.to_range().map_or(0, |range| range.start))
        } else {
            Span::call_site()
        };

        Diagnostic::spanned(span, Level::Error, format!("{}", err)).abort()
    });

    let consts = gen_consts(&module);
    let types = gen_types(&module);
    let entry_points = gen_entry_points(&module, &info, &wgsl.source);

    let expanded = quote! {
        #consts
        #types
        #entry_points
    };

    proc_macro::TokenStream::from(expanded)
}

fn gen_entry_points(module: &Module, info: &ModuleInfo, source: &str) -> TokenStream {
    let entry_points = module
        .entry_points
        .iter()
        .enumerate()
        .map(|(i, entry_point)| {
            let name = &entry_point.name;
            let ident = Ident::new(name, Span::call_site());

            let function_info = info.get_entry_point(i);

            match entry_point.stage {
                ShaderStage::Compute => gen_compute_entry_point(
                    module,
                    entry_point,
                    source,
                    name,
                    &ident,
                    function_info,
                ),
                _ => unimplemented!(),
            }
        });

    quote! {
        #(#entry_points)*
    }
}

fn gen_compute_entry_point(
    module: &Module,
    entry_point: &EntryPoint,
    source: &str,
    name: &str,
    ident: &Ident,
    function_info: &FunctionInfo,
) -> TokenStream {
    let bindings_ident = Ident::new("Bindings", Span::call_site());

    let bindings = gen_entry_point_bindings(module, &function_info, &bindings_ident);

    let bindings_param = if bindings.is_some() {
        Some(quote!(mut bindings: #ident::#bindings_ident<'a>,))
    } else {
        None
    };

    let bindings_build_var = if bindings.is_some() {
        quote!(bindings)
    } else {
        quote!(())
    };

    let bindings_var = if bindings.is_some() {
        quote!(bindings)
    } else {
        quote!()
    };

    let shader_bindings = if bindings.is_some() {
        quote!(#bindings_ident<'a>)
    } else {
        quote!(())
    };

    let work_group_size = {
        let x = entry_point.workgroup_size[0];
        let y = entry_point.workgroup_size[1];
        let z = entry_point.workgroup_size[2];

        quote!(::shatter::WorkGroupSize::new(
            #x as ::std::primitive::u32,
            #y as ::std::primitive::u32,
            #z as ::std::primitive::u32,
        ))
    };

    quote! {
        pub mod #ident {
            use super::*;

            pub const WORK_GROUP_SIZE: ::shatter::WorkGroupSize = #work_group_size;

            #bindings

            pub struct Shader;

            impl<'a> ::shatter::ComputeShader<'a> for Shader {
                type Bindings = #shader_bindings;

                const SOURCE: &'static ::std::primitive::str = #source;
                const ENTRY_POINT: &'static ::std::primitive::str = #name;
            }

            pub fn build<'a>(#bindings_param) -> ::shatter::ComputeShaderBuilder<'a, Shader> {
                ::shatter::ComputeShaderBuilder::new(#bindings_build_var)
            }
        }

        pub fn #ident<'a>(#bindings_param dispatch: ::shatter::Dispatch) {
            #ident::build(#bindings_var).dispatch(dispatch);
        }
    }
}

fn gen_entry_point_bindings(
    module: &Module,
    function: &FunctionInfo,
    ident: &Ident,
) -> Option<TokenStream> {
    let mut max_group = 0;
    let mut bind_group_layout_descriptors = HashMap::new();
    let mut bind_group_descriptors = HashMap::new();
    let mut prepare = Vec::new();
    let mut read = Vec::new();
    let mut write = Vec::new();

    let fields = module
        .global_variables
        .iter()
        .filter_map(|(handle, variable)| {
            let binding = variable.binding.as_ref()?;

            let var_use = function[handle];

            // terminate if variable is unused
            if var_use.is_empty() {
                return None;
            }

            let name = variable.name.as_ref()?;
            let ident = Ident::new(name, Span::call_site());

            max_group = max_group.max(binding.group);

            let ty = &module.types[variable.ty].inner;

            let binding_type = match ty {
                &TypeInner::Image {
                    ref dim,
                    arrayed,
                    ref class,
                } => {
                    let dimension = wgpu_view_dimension(dim, arrayed);

                    match class {
                        ImageClass::Storage { format, access } => {
                            let access = match access {
                                _ if access.contains(StorageAccess::LOAD)
                                    && access.contains(StorageAccess::STORE) =>
                                {
                                    quote!(::shatter::wgpu::StorageTextureAccess::ReadWrite)
                                }
                                _ if access.contains(StorageAccess::LOAD) => {
                                    quote!(::shatter::wgpu::StorageTextureAccess::ReadOnly)
                                }
                                _ if access.contains(StorageAccess::STORE) => {
                                    quote!(::shatter::wgpu::StorageTextureAccess::WriteOnly)
                                }
                                _ => unreachable!(),
                            };

                            let format = wgpu_texture_format(format);

                            quote!(::shatter::BindingType::StorageTexture {
                                access: #access,
                                format: #format,
                                view_dimension: #dimension,
                            })
                        }
                        _ => unimplemented!(),
                    }
                }
                _ => {
                    let buffer_binding_type = match variable.class {
                        StorageClass::Uniform => quote!(::shatter::BufferBindingType::Uniform),
                        StorageClass::Storage { access } => {
                            let read_only = !access.contains(StorageAccess::STORE);

                            quote!(::shatter::BufferBindingType::Storage { read_only: #read_only })
                        }
                        _ => unimplemented!(),
                    };

                    quote! {
                        ::shatter::BindingType::Buffer {
                            ty: #buffer_binding_type,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        }
                    }
                }
            };

            let layout_descriptor = bind_group_layout_descriptors
                .entry(binding.group)
                .or_insert_with(|| HashMap::new());

            let b = binding.binding;

            layout_descriptor.insert(
                binding.binding,
                quote! {
                    ::shatter::BindGroupLayoutEntry {
                        binding: #b,
                        visibility: ::shatter::ShaderStages::COMPUTE,
                        ty: #binding_type,
                        count: ::std::option::Option::None,
                    }
                },
            );

            let descriptor = bind_group_descriptors
                .entry(binding.group)
                .or_insert_with(|| HashMap::new());

            descriptor.insert(
                binding.binding,
                quote!(::shatter::Binding::binding_resource(self.#ident)),
            );

            let ty = rust_type(module, variable.ty, &mut None, false);

            // prepare binding
            prepare.push(quote!(::shatter::Binding::prepare(self.#ident)));

            // only read and write as necessary
            if var_use.contains(GlobalUse::READ) {
                read.push(quote!(::shatter::Binding::read(self.#ident)));
            }

            if var_use.contains(GlobalUse::WRITE) {
                write.push(quote!(::shatter::Binding::write(self.#ident)));
            }

            if var_use.contains(GlobalUse::WRITE) {
                return Some(quote!(pub #ident: &'a mut dyn ::shatter::Binding<#ty>));
            }

            if var_use.contains(GlobalUse::READ) {
                return Some(quote!(pub #ident: &'a dyn ::shatter::Binding<#ty>));
            }

            None
        })
        .collect::<Vec<_>>();

    let bind_group_layout_descriptors = (0..=max_group).into_iter().map(|group| {
        if let Some(descriptor) = bind_group_layout_descriptors.get(&group) {
            let mut entries = descriptor.iter().collect::<Vec<_>>();

            entries.sort_by(|(a, _), (b, _)| a.cmp(b));

            let entries = entries.into_iter().map(|(_binding, entry)| entry);

            quote! {
                ::shatter::BindGroupLayoutDescriptor {
                    entries: ::std::vec![#(#entries),*],
                }
            }
        } else {
            quote! {
                ::shatter::BindGroupLayoutDescriptor {
                    entries: ::std::vec::Vec::new(),
                }
            }
        }
    });

    let bind_group_descriptors = (0..=max_group).into_iter().map(|group| {
        if let Some(descriptor) = bind_group_descriptors.get(&group) {
            let mut resources = descriptor.iter().collect::<Vec<_>>();

            resources.sort_by(|(a, _), (b, _)| a.cmp(b));

            let resources = resources.into_iter().map(|(binding, resource)| {
                quote! {
                    ::shatter::BindGroupEntry {
                        binding: #binding as u32,
                        resource: #resource,
                    }
                }
            });

            quote! {
                ::shatter::BindGroupDescriptor {
                    layout: layouts.next().unwrap().clone(),
                    entries: ::std::vec![
                        #(#resources),*
                    ],
                }
            }
        } else {
            quote! {
                ::shatter::BindGroupDescriptor {
                    layout: layouts.next().unwrap().clone(),
                    entries: ::std::vec::Vec::new(),
                }
            }
        }
    });

    if !fields.is_empty() {
        Some(quote! {
            pub struct #ident<'a> {
                #(#fields),*
            }

            impl<'a> #ident<'a> {
                pub fn bind_group_layout_descriptors(
                    &self,
                ) -> ::std::vec::Vec<::shatter::BindGroupLayoutDescriptor> {
                    ::std::vec![#(#bind_group_layout_descriptors),*]
                }

                pub fn bind_group_descriptors(
                    &self,
                    layouts: &[::shatter::BindGroupLayoutId],
                ) -> ::std::vec::Vec<::shatter::BindGroupDescriptor> {
                    let mut layouts = layouts.iter();

                    ::std::vec![#(#bind_group_descriptors),*]
                }
            }

            impl<'a> ::shatter::Bindings for #ident<'a> {
                #[inline]
                fn bind_group_layout_descriptors(
                    &self,
                ) -> ::std::vec::Vec<::shatter::BindGroupLayoutDescriptor> {
                    self.bind_group_layout_descriptors()
                }

                #[inline]
                fn bind_group_descriptors(
                    &self,
                    layouts: &[::shatter::BindGroupLayoutId],
                ) -> ::std::vec::Vec<::shatter::BindGroupDescriptor> {
                    self.bind_group_descriptors(layouts)
                }

                #[inline]
                fn prepare(&self) {
                    #(#prepare;)*
                }

                #[inline]
                fn read(&self) {
                    #(#read;)*
                }

                #[inline]
                fn write(&mut self) {
                    #(#write;)*
                }
            }
        })
    } else {
        None
    }
}

fn wgpu_texture_format(format: &StorageFormat) -> TokenStream {
    match format {
        StorageFormat::Rgba8Unorm => quote!(::shatter::wgpu::TextureFormat::Rgba8Unorm),
        _ => todo!(),
    }
}

fn wgpu_view_dimension(dimension: &ImageDimension, arrayed: bool) -> TokenStream {
    match dimension {
        ImageDimension::D1 => quote!(::shatter::wgpu::TextureViewDimension::D1),
        ImageDimension::D2 => {
            if arrayed {
                quote!(::shatter::wgpu::TextureViewDimension::D2Array)
            } else {
                quote!(::shatter::wgpu::TextureViewDimension::D2)
            }
        }
        ImageDimension::D3 => quote!(::shatter::wgpu::TextureViewDimension::D3),
        ImageDimension::Cube => {
            if arrayed {
                quote!(::shatter::wgpu::TextureViewDimension::CubeArray)
            } else {
                quote!(::shatter::wgpu::TextureViewDimension::Cube)
            }
        }
    }
}

fn gen_consts(module: &Module) -> TokenStream {
    let consts = module
        .constants
        .iter()
        .map(|(_, constant)| gen_const(module, constant));

    quote! {
        #(#consts)*
    }
}

fn gen_const(module: &Module, constant: &Constant) -> Option<TokenStream> {
    let name = constant.name.as_ref()?;

    let ident = Ident::new(name, Span::call_site());

    let ty = constant.inner.resolve_type();

    let ty = match ty {
        TypeResolution::Value(ref inner) => rust_type_inner(module, inner, &mut None, false),
        TypeResolution::Handle(handle) => rust_type(module, handle, &mut None, false),
    };

    let value = const_value(module, constant);

    Some(quote! {
        pub const #ident: #ty = #value;
    })
}

fn const_value(_module: &Module, constant: &Constant) -> TokenStream {
    match constant.inner {
        ConstantInner::Scalar { width, value } => match value {
            ScalarValue::Bool(value) => quote!(#value),
            ScalarValue::Float(value) => match width {
                4 => quote!(#value as f32),
                8 => quote!(#value as f64),
                _ => unimplemented!("float of width '{}' not supported", width),
            },
            ScalarValue::Sint(value) => match width {
                1 => quote!(#value as i8),
                2 => quote!(#value as i16),
                4 => quote!(#value as i32),
                8 => quote!(#value as i64),
                _ => unimplemented!("signed integer of width '{}' not supported", width),
            },
            ScalarValue::Uint(value) => match width {
                1 => quote!(#value as i8),
                2 => quote!(#value as i16),
                4 => quote!(#value as i32),
                8 => quote!(#value as i64),
                _ => unimplemented!("unsigned integer of width '{}' not supported", width),
            },
        },
        _ => unimplemented!(),
    }
}

fn gen_types(module: &Module) -> TokenStream {
    let types = module.types.iter().map(|(_, ty)| gen_type(module, ty));

    quote! {
        #(#types)*
    }
}

fn gen_type(module: &Module, ty: &Type) -> Option<TokenStream> {
    let name = ty.name.as_ref()?;
    let name_sized = Ident::new(&format!("{}_Sized", name), Span::call_site());
    let name = Ident::new(name, Span::call_site());

    match ty.inner {
        TypeInner::Struct {
            top_level: false,
            ref members,
            ..
        } => {
            let members = members.iter().map(|member| {
                let ident = Ident::new(member.name.as_ref().unwrap(), Span::call_site());

                let ty = rust_type(module, member.ty, &mut None, false);

                quote! {
                    pub #ident: #ty
                }
            });

            Some(quote! {
                #[derive(Clone, Copy, Debug, PartialEq)]
                pub struct #name {
                    #(#members),*
                }
            })
        }
        TypeInner::Struct {
            top_level: true,
            ref members,
            ..
        } => {
            let mut buffer = None;

            let unsized_members = members
                .iter()
                .map(|member| {
                    let ident = Ident::new(member.name.as_ref().unwrap(), Span::call_site());

                    let ty = rust_type(module, member.ty, &mut buffer, false);

                    quote! {
                        pub #ident: #ty
                    }
                })
                .collect::<Vec<_>>();

            let sized_struct = if buffer.is_some() {
                let sized_members = members.iter().map(|member| {
                    let ident = Ident::new(member.name.as_ref().unwrap(), Span::call_site());

                    let ty = rust_type(module, member.ty, &mut None, true);

                    quote! {
                        pub #ident: #ty
                    }
                });

                Some(quote! {
                    #[repr(C)]
                    #[derive(Debug, Default, PartialEq)]
                    pub struct #name_sized {
                        #(#sized_members),*
                    }
                })
            } else {
                None
            };

            let derives = if buffer.is_some() {
                quote!(#[derive(Debug, PartialEq)])
            } else {
                quote!(#[derive(Debug, Default, PartialEq)])
            };

            let buffer_impl = if let Some(buffer_ty) = buffer {
                array_buffer_impl(&name, &name_sized, &buffer_ty)
            } else {
                buffer_impl(&name)
            };

            Some(quote! {
                #[repr(C)]
                #derives
                pub struct #name {
                    #(#unsized_members),*
                }

                #sized_struct

                #buffer_impl
            })
        }
        _ => None,
    }
}

fn buffer_impl(name: &Ident) -> TokenStream {
    quote! {
        unsafe impl ::shatter::BufferData for #name {
            type State = ();

            fn init() -> Self::State {}

            fn size(_: &Self::State) -> usize {
                ::std::mem::size_of::<#name>()
            }

            unsafe fn alloc() -> ::std::ptr::NonNull<u8> {
                if ::std::mem::size_of::<#name>() == 0 {
                    return ::std::ptr::NonNull::<#name>::dangling().cast();
                }

                let layout = ::std::alloc::Layout::new::<#name>();
                let ptr = unsafe { ::std::alloc::alloc(layout) };

                unsafe { ::std::ptr::write(ptr as *mut #name, ::std::default::Default::default()) };
                ::std::ptr::NonNull::new(ptr).unwrap()
            }

            unsafe fn dealloc(ptr: ::std::ptr::NonNull<u8>, _: &Self::State) {
                let layout = ::std::alloc::Layout::new::<#name>();

                if layout.size() == 0 {
                    return;
                }

                unsafe { ::std::alloc::dealloc(ptr.as_ptr(), layout) };
            }

            unsafe fn as_ptr(ptr: ::std::ptr::NonNull<u8>, _: &Self::State) -> *mut Self {
                ptr.as_ptr() as *mut Self
            }
        }
    }
}

fn array_buffer_impl(name: &Ident, name_sized: &Ident, buffer_ty: &TokenStream) -> TokenStream {
    quote! {
        unsafe impl ::shatter::BufferData for #name {
            type State = (usize, usize);

            fn init() -> Self::State {
                let cap = if ::std::mem::size_of::<#buffer_ty>() == 0 { !0 } else { 0 };

                (0, cap)
            }

            fn size(&(length, _capacity): &Self::State) -> usize {
                ::std::mem::size_of::<#name_sized>() + length * ::std::mem::size_of::<#buffer_ty>()
            }

            unsafe fn alloc() -> ::std::ptr::NonNull<u8> {
                if ::std::mem::size_of::<#name_sized>() == 0 {
                    return ::std::ptr::NonNull::<#name_sized>::dangling().cast();
                }

                let layout = ::std::alloc::Layout::new::<#name_sized>();
                let ptr = unsafe { ::std::alloc::alloc(layout) };

                unsafe { ::std::ptr::write(ptr as *mut #name_sized, ::std::default::Default::default()) };
                ::std::ptr::NonNull::new(ptr).unwrap()
            }

            unsafe fn dealloc(ptr: ::std::ptr::NonNull<u8>, &(_length, capacity): &Self::State) {
                let sized_layout = ::std::alloc::Layout::new::<#name_sized>();

                let layout = if ::std::mem::size_of::<#buffer_ty>() > 0 {
                    let array_layout = ::std::alloc::Layout::array::<#buffer_ty>(capacity).unwrap();

                    sized_layout.extend(array_layout).unwrap().0
                } else {
                    sized_layout
                };

                if layout.size() == 0 {
                    return;
                }

                unsafe { ::std::alloc::dealloc(ptr.as_ptr(), layout) };
            }

            unsafe fn as_ptr(ptr: ::std::ptr::NonNull<u8>, &(length, _capacity): &Self::State) -> *mut Self {
                let slice = unsafe { ::std::slice::from_raw_parts_mut(ptr.as_ptr(), length) };

                unsafe { ::std::mem::transmute(slice as *mut [u8]) }
            }
        }

        unsafe impl ::shatter::BufferVec for #name {
            type Item = #buffer_ty;

            fn len(&(length, _): &Self::State) -> usize {
                length
            }

            unsafe fn grow(
                ptr: &mut ::std::ptr::NonNull<u8>,
                (length, capacity): &mut Self::State,
            ) {
                assert!(::std::mem::size_of::<Self::Item>() != 0, "capacity overflow");

                let (new_cap, new_layout) = if *capacity == 0 {
                    let sized_layout = ::std::alloc::Layout::new::<#name_sized>();
                    let array_layout = ::std::alloc::Layout::array::<Self::Item>(1).unwrap();
                    let new_layout = sized_layout.extend(array_layout).unwrap().0.pad_to_align();

                    (1, new_layout)
                } else {
                    let new_cap = 2 * *capacity;

                    let sized_layout = ::std::alloc::Layout::new::<#name_sized>();
                    let array_layout = ::std::alloc::Layout::array::<Self::Item>(new_cap).unwrap();
                    let new_layout = sized_layout.extend(array_layout).unwrap().0.pad_to_align();
                    (new_cap, new_layout)
                };

                assert!(
                    new_layout.size() <= ::std::primitive::isize::MAX as usize,
                    "Allocation too large"
                );

                let new_ptr = if *capacity == 0 && ::std::mem::size_of::<#name_sized>() == 0 {
                    unsafe { ::std::alloc::alloc(new_layout) }
                } else {
                    let sized_layout = ::std::alloc::Layout::new::<#name_sized>();
                    let array_layout = ::std::alloc::Layout::array::<Self::Item>(*capacity).unwrap();
                    let old_layout = sized_layout.extend(array_layout).unwrap().0.pad_to_align();
                    let old_ptr = ptr.as_ptr();
                    unsafe { ::std::alloc::realloc(old_ptr, old_layout, new_layout.size()) }
                };

                *ptr = match ::std::ptr::NonNull::new(new_ptr) {
                    Some(ptr) => ptr,
                    None => ::std::alloc::handle_alloc_error(new_layout),
                };

                *capacity = new_cap;
            }

            unsafe fn push(
                ptr: &mut ::std::ptr::NonNull<u8>,
                state: &mut Self::State,
                item: Self::Item
            ) {
                if state.0 == state.1 {
                    Self::grow(ptr, state);
                }

                let layout = ::std::alloc::Layout::new::<#name_sized>();

                unsafe {
                    ::std::ptr::write(
                        (ptr.as_ptr().add(layout.size()) as *mut Self::Item).add(state.0),
                        item,
                    );
                }

                state.0 += 1;
            }

            unsafe fn pop(
                ptr: ::std::ptr::NonNull<u8>,
                (length, _capacity): &mut Self::State,
            ) -> ::std::option::Option<Self::Item> {
                if *length == 0 {
                    None
                } else {
                    *length -= 1;

                    let layout = ::std::alloc::Layout::new::<#name_sized>();

                    unsafe {
                        Some(
                            ::std::ptr::read(
                                (ptr.as_ptr().add(layout.size()) as *mut Self::Item).add(*length)
                            )
                        )
                    }
                }
            }
        }
    }
}

fn rust_type(
    module: &Module,
    ty: Handle<Type>,
    buffer: &mut Option<TokenStream>,
    force_sized: bool,
) -> TokenStream {
    let ty = module.types.get_handle(ty).unwrap();

    match ty.inner {
        TypeInner::Struct { .. } => {
            let name = ty.name.as_ref().unwrap();

            let ident = Ident::new(name, Span::call_site());

            quote! { #ident }
        }
        ref inner => rust_type_inner(module, inner, buffer, force_sized),
    }
}

fn rust_type_inner(
    module: &Module,
    inner: &TypeInner,
    buffer: &mut Option<TokenStream>,
    force_sized: bool,
) -> TokenStream {
    match *inner {
        TypeInner::Scalar { kind, width } => rust_scalar(kind, width),
        TypeInner::Vector { size, kind, width } => {
            let scalar = rust_scalar(kind, width);

            match size {
                VectorSize::Bi => quote!(::shatter::Vec2<#scalar>),
                VectorSize::Tri => quote!(::shatter::Vec3<#scalar>),
                VectorSize::Quad => quote!(::shatter::Vec4<#scalar>),
            }
        }
        TypeInner::Matrix {
            columns,
            rows,
            width,
        } => {
            let scalar = rust_scalar(ScalarKind::Float, width);

            match columns {
                VectorSize::Bi => match rows {
                    VectorSize::Bi => quote!([[#scalar; 2]; 2]),
                    VectorSize::Tri => quote!([[#scalar; 3]; 2]),
                    VectorSize::Quad => quote!([[#scalar; 4]; 2]),
                },
                VectorSize::Tri => match rows {
                    VectorSize::Bi => quote!([[#scalar; 2]; 3]),
                    VectorSize::Tri => quote!([[#scalar; 3]; 3]),
                    VectorSize::Quad => quote!([[#scalar; 4]; 3]),
                },
                VectorSize::Quad => match rows {
                    VectorSize::Bi => quote!([[#scalar; 2]; 4]),
                    VectorSize::Tri => quote!([[#scalar; 3]; 4]),
                    VectorSize::Quad => quote!([[#scalar; 4]; 4]),
                },
            }
        }
        TypeInner::Atomic { kind, width } => rust_scalar(kind, width),
        TypeInner::Array { base, size, .. } => {
            let base = rust_type(module, base, buffer, force_sized);

            match size {
                ArraySize::Constant(size) => {
                    let size = rust_const(module, size);

                    quote!([#base; #size as ::std::primitive::usize])
                }
                ArraySize::Dynamic => {
                    if force_sized {
                        quote!([#base; 0])
                    } else {
                        *buffer = Some(base.clone());

                        quote!([#base])
                    }
                }
            }
        }
        TypeInner::Image {
            dim,
            arrayed,
            class,
        } => {
            let dimension = match dim {
                ImageDimension::D1 => quote!(::shatter::texture_view_dimension::D1),
                ImageDimension::D2 => {
                    if arrayed {
                        quote!(::shatter::texture_view_dimension::D2Array)
                    } else {
                        quote!(::shatter::texture_view_dimension::D2)
                    }
                }
                ImageDimension::D3 => quote!(::shatter::texture_View_dimension::D3),
                ImageDimension::Cube => {
                    if arrayed {
                        quote!(::shatter::texture_view_dimension::CubeArray)
                    } else {
                        quote!(::shatter::texture_view_dimension::Cube)
                    }
                }
            };

            match class {
                ImageClass::Sampled { kind, multi } => {
                    let sample_type = match kind {
                        ScalarKind::Float => quote!(::shatter::texture_sample_type::Float<true>),
                        ScalarKind::Sint => quote!(::shatter::texture_sample_type::Sint),
                        ScalarKind::Uint => quote!(::shatter::texture_sample_type::Uint),
                        ScalarKind::Bool => panic!(),
                    };

                    quote!(::shatter::TextureBinding<#sample_type, #dimension, #multi>)
                }
                ImageClass::Storage { format, .. } => {
                    let texel_format = match format {
                        StorageFormat::Rgba8Unorm => quote!(::shatter::texel_format::Rgba8Unorm),
                        _ => unimplemented!(),
                    };

                    quote!(::shatter::StorageTextureBinding<#texel_format, #dimension>)
                }
                _ => unimplemented!(),
            }
        }
        _ => unimplemented!("type cannot be resolved"),
    }
}

fn rust_const(module: &Module, constant: Handle<Constant>) -> TokenStream {
    let constant = module.constants.try_get(constant).unwrap();

    match constant.name {
        Some(ref name) => {
            let ident = Ident::new(name, Span::call_site());

            quote!(#ident)
        }
        None => const_value(module, constant),
    }
}

fn rust_scalar(kind: ScalarKind, width: u8) -> TokenStream {
    match kind {
        ScalarKind::Bool => quote!(bool),
        ScalarKind::Sint => match width {
            1 => quote!(::std::primitive::i8),
            2 => quote!(::std::primitive::i16),
            4 => quote!(::std::primitive::i32),
            8 => quote!(::std::primitive::i64),
            width => unreachable!("scalar with of '{}' not supported", width),
        },
        ScalarKind::Uint => match width {
            1 => quote!(::std::primitive::u8),
            2 => quote!(::std::primitive::u16),
            4 => quote!(::std::primitive::u32),
            8 => quote!(::std::primitive::u64),
            width => unreachable!("scalar with of '{}' not supported", width),
        },
        ScalarKind::Float => match width {
            4 => quote!(::std::primitive::f32),
            8 => quote!(::std::primitive::f64),
            width => unreachable!("scalar with of '{}' not supported", width),
        },
    }
}
