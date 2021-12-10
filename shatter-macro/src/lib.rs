mod shatter;
mod wgsl;

#[proc_macro_error::proc_macro_error]
#[proc_macro]
pub fn wgsl(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let wgsl = wgsl::Wgsl::new(&input.into());

    shatter::shatter(&wgsl)
}
