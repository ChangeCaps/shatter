use std::collections::BTreeMap;

use proc_macro2::{Delimiter, Spacing, TokenTree};
use proc_macro_error::{Diagnostic, Level};

pub trait WgslResult {
    type Ok;

    fn wgsl_unwrap(self, wgsl: &Wgsl) -> Self::Ok;
}

impl<T> WgslResult for Result<T, naga::front::wgsl::ParseError> {
    type Ok = T;

    fn wgsl_unwrap(self, wgsl: &Wgsl) -> <Self as WgslResult>::Ok {
        match self {
            Self::Ok(value) => value,
            Self::Err(error) => {
                let (_line, column) = error.location(&wgsl.source);

                let span = wgsl.get_span(column);

                Diagnostic::spanned(*span, Level::Error, error.to_string()).abort()
            }
        }
    }
}

#[derive(Default)]
pub struct Wgsl {
    pub spans: BTreeMap<usize, proc_macro2::Span>,
    pub source: String,
}

impl Wgsl {
    pub fn get_span(&self, start: usize) -> &proc_macro2::Span {
        if let Some(span) = self.spans.get(&start) {
            return span;
        }

        for (span_start, span) in self.spans.iter().rev() {
            if start >= *span_start {
                return span;
            }
        }

        proc_macro_error::abort_call_site! {
            "span not found"
        }
    }

    #[inline]
    pub fn new(source: &proc_macro2::TokenStream) -> Self {
        let mut wgsl = Self::default();

        for tree in source.clone() {
            wgsl.add_tree(tree);
        }

        wgsl
    }

    pub fn add_tree(&mut self, tree: TokenTree) {
        match tree {
            TokenTree::Group(group) => {
                match group.delimiter() {
                    Delimiter::Parenthesis => {
                        self.add_string("(", group.span_open());
                    }
                    Delimiter::Brace => {
                        self.add_string("{", group.span_open());
                    }
                    Delimiter::Bracket => {
                        self.add_string("[", group.span_open());
                    }
                    _ => {}
                }

                for tree in group.stream() {
                    self.add_tree(tree);
                }

                match group.delimiter() {
                    Delimiter::Parenthesis => {
                        self.add_string(")", group.span_open());
                    }
                    Delimiter::Brace => {
                        self.add_string("}", group.span_open());
                    }
                    Delimiter::Bracket => {
                        self.add_string("]", group.span_open());
                    }
                    _ => {}
                }
            }
            TokenTree::Ident(ident) => {
                self.add_string(&ident.to_string(), ident.span());
                self.source.push(' ');
            }
            TokenTree::Literal(lit) => {
                self.add_string(&lit.to_string(), lit.span());
                self.source.push(' ');
            }
            TokenTree::Punct(punct) => {
                self.add_string(&punct.to_string(), punct.span());

                match punct.spacing() {
                    Spacing::Alone => self.source.push(' '),
                    Spacing::Joint => {}
                }
            }
        }
    }

    pub fn add_string(&mut self, string: &str, span: proc_macro2::Span) {
        let start = self.source.len();

        self.source += string;

        self.spans.insert(start, span);
    }
}
