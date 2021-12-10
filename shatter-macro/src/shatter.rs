use std::collections::HashMap;

use crate::wgsl::{Wgsl, WgslResult};
use naga::{
    proc::TypeResolution,
    valid::{Capabilities, FunctionInfo, GlobalUse, ModuleInfo, ValidationFlags, Validator},
    ArraySize, Constant, ConstantInner, Handle, Module, ScalarKind, ScalarValue, StorageAccess,
    StorageClass, Type, TypeInner, VectorSize,
};
use proc_macro2::{Ident, Span, TokenStream};
use quote::quote;

pub fn shatter(wgsl: &Wgsl) -> proc_macro::TokenStream {
    let module = naga::front::wgsl::parse_str(&wgsl.source).wgsl_unwrap(wgsl);

    let mut validator = Validator::new(ValidationFlags::all(), Capabilities::all());
    let info = validator.validate(&module).unwrap();

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

            let bindings_ident =
                Ident::new("Bindings", Span::call_site());

            let bindings = gen_entry_point_bindings(module, &function_info, &bindings_ident);

            let bindings_param = if bindings.is_some() {
                Some(quote!(mut bindings: #ident::#bindings_ident<'a>,))
            } else {
                None
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
                        type Bindings = #bindings_ident<'a>;

                        const SOURCE: &'static ::std::primitive::str = #source;
                        const ENTRY_POINT: &'static ::std::primitive::str = #name;
                    }

                    pub fn build<'a>(#bindings_param) -> ::shatter::ComputeShaderBuilder<'a, Shader> {
                        ::shatter::ComputeShaderBuilder::new(bindings)
                    }
                }

                pub fn #ident<'a>(#bindings_param dispatch: ::shatter::Dispatch) {
                    #ident::build(bindings).dispatch(dispatch);
                }
            }
        });

    quote! {
        #(#entry_points)*
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
    let mut upload = Vec::new();
    let mut download = Vec::new();

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
                quote!(::shatter::BindingResource::from(&*self.#ident)),
            );

            let ty = rust_type(module, variable.ty, &mut None, false);

            if var_use.contains(GlobalUse::WRITE) {
                // only download if the buffer has been written to
                download.push(quote!(self.#ident.mark_needs_download()));
            }

            if var_use.contains(GlobalUse::READ) {
                // only upload if the buffer will be read
                upload.push(quote!(self.#ident.upload()));
            } else {
                upload.push(quote!(self.#ident.resize_buffer()));
            }

            if var_use.contains(GlobalUse::WRITE) {
                return Some(quote!(pub #ident: &'a mut ::shatter::Buffer<#ty>));
            }

            if var_use.contains(GlobalUse::READ) {
                return Some(quote!(pub #ident: &'a ::shatter::Buffer<#ty>));
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

                pub fn upload(&mut self) {
                    #(#upload;)*
                }

                pub fn download(&mut self) {
                    #(#download;)*
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
                fn upload(&mut self) {
                    self.upload();
                }

                #[inline]
                fn download(&mut self) {
                    self.download();
                }
            }
        })
    } else {
        None
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
                assert!(::std::mem::size_of::<#buffer_ty>() != 0, "capacity overflow");

                let (new_cap, new_layout) = if *capacity == 0 {
                    (1, ::std::alloc::Layout::array::<#buffer_ty>(1).unwrap())
                } else {
                    let new_cap = 2 * *capacity;

                    let new_layout = ::std::alloc::Layout::array::<#buffer_ty>(new_cap).unwrap();
                    (new_cap, new_layout)
                };

                assert!(
                    new_layout.size() <= ::std::primitive::isize::MAX as usize,
                    "Allocation too large"
                );

                let new_ptr = if *capacity == 0 {
                    unsafe { ::std::alloc::alloc(new_layout) }
                } else {
                    let old_layout = ::std::alloc::Layout::array::<#buffer_ty>(*capacity).unwrap();
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
