use crate::TextureData;

macro_rules! color {
    {
		colors: [$d1:ident, $d2:ident, $d3: ident, $d4:ident],
		data: $data:path,
		zero: $zero:expr,
		one: $one:expr,
	} => {
		#[repr(C)]
		#[derive(Clone, Copy, Debug, Default, PartialEq, PartialOrd)]
		pub struct $d1 {
			pub r: $data,
		}

		unsafe impl TextureData for $d1 {}

		impl $d1 {
			pub const BLACK: Self = Self::r($zero);
			pub const WHITE: Self = Self::r($one);

			#[inline]
			pub const fn r(r: $data) -> Self {
				Self { r }
			}
		}

		#[repr(C)]
		#[derive(Clone, Copy, Debug, Default, PartialEq, PartialOrd)]
		pub struct $d2 {
			pub r: $data,
			pub g: $data,
		}

		unsafe impl TextureData for $d2 {}

		impl $d2 {
			pub const BLACK: Self = Self::rg($zero, $zero);
			pub const WHITE: Self = Self::rg($one, $one);

			#[inline]
			pub const fn r(r: $data) -> Self {
				Self { r, g: $zero }
			}

			#[inline]
			pub const fn rg(r: $data, g: $data) -> Self {
				Self { r, g }
			}
		}

		#[repr(C)]
		#[derive(Clone, Copy, Debug, Default, PartialEq, PartialOrd)]
		pub struct $d3 {
			pub r: $data,
			pub g: $data,
			pub b: $data,
		}

		unsafe impl TextureData for $d3 {}

		impl $d3 {
			pub const BLACK: Self = Self::rgb($zero, $zero, $zero);
			pub const WHITE: Self = Self::rgb($one, $one, $one);

			#[inline]
			pub const fn r(r: $data) -> Self {
				Self { r, g: $zero, b: $zero }
			}

			#[inline]
			pub const fn rg(r: $data, g: $data) -> Self {
				Self { r, g, b: $zero }
			}

			#[inline]
			pub const fn rgb(r: $data, g: $data, b: $data) -> Self {
				Self { r, g, b }
			}
		}

		#[repr(C)]
		#[derive(Clone, Copy, Debug, Default, PartialEq, PartialOrd)]
		pub struct $d4 {
			pub r: $data,
			pub g: $data,
			pub b: $data,
			pub a: $data,
		}

		unsafe impl TextureData for $d4 {}

		impl $d4 {
			pub const TRANSPARENT: Self = Self::rgba($zero, $zero, $zero, $zero);
			pub const BLACK: Self = Self::rgb($zero, $zero, $zero);
			pub const WHITE: Self = Self::rgb($one, $one, $one);

			#[inline]
			pub const fn r(r: $data) -> Self {
				Self { r, g: $zero, b: $zero, a: $one }
			}

			#[inline]
			pub const fn rg(r: $data, g: $data) -> Self {
				Self { r, g, b: $zero, a: $one }
			}

			#[inline]
			pub const fn rgb(r: $data, g: $data, b: $data) -> Self {
				Self { r, g, b, a: $one }
			}

			#[inline]
			pub const fn rgba(r: $data, g: $data, b: $data, a: $data) -> Self {
				Self { r, g, b, a }
			}
		}

		impl Into<[$data; 4]> for $d4 {
			fn into(self) -> [$data; 4] {
				[self.r, self.g, self.b, self.a]
			}
		}
	};
}

color! {
    colors: [R8U, Rg8U, Rgb8U, Rgba8U],
    data: u8,
    zero: u8::MIN,
    one: u8::MAX,
}

color! {
    colors: [R8I, Rg8I, Rgb8I, Rgba8I],
    data: i8,
    zero: i8::MIN,
    one: i8::MAX,
}

color! {
    colors: [R16U, Rg16U, Rgb16U, Rgba16U],
    data: u16,
    zero: u16::MIN,
    one: u16::MAX,
}

color! {
    colors: [R16I, Rg16I, Rgb16I, Rgba16I],
    data: i16,
    zero: i16::MIN,
    one: i16::MAX,
}

color! {
    colors: [R32U, Rg32U, Rgb32U, Rgba32U],
    data: u32,
    zero: 0,
    one: 1,
}

color! {
    colors: [R32I, Rg32I, Rgb32I, Rgba32I],
    data: i32,
    zero: 0,
    one: 1,
}

color! {
    colors: [R32, Rg32, Rgb32, Rgba32],
    data: f32,
    zero: 0.0,
    one: 1.0,
}
