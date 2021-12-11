#[repr(C, align(8))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Vec2<T> {
    pub x: T,
    pub y: T,
}

impl<T> Vec2<T> {
    pub const fn new(x: T, y: T) -> Self {
        Self { x, y }
    }
}

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Vec3<T> {
    pub x: T,
    pub y: T,
    pub z: T,
}

impl<T> Vec3<T> {
    pub const fn new(x: T, y: T, z: T) -> Self {
        Self { x, y, z }
    }
}

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Vec4<T> {
    pub x: T,
    pub y: T,
    pub z: T,
    pub w: T,
}

impl<T> Vec4<T> {
    pub const fn new(x: T, y: T, z: T, w: T) -> Self {
        Self { x, y, z, w }
    }
}

macro_rules! impl_vec {
    ($ty:ty, zero: $zero:expr) => {
        impl Vec2<$ty> {
            pub const ZERO: Self = Self::new($zero, $zero);
        }

        impl Vec3<$ty> {
            pub const ZERO: Self = Self::new($zero, $zero, $zero);
        }

        impl Vec4<$ty> {
            pub const ZERO: Self = Self::new($zero, $zero, $zero, $zero);
        }
    };
}

impl_vec!(f32, zero: 0.0);
impl_vec!(i32, zero: 0);
impl_vec!(u32, zero: 0);
