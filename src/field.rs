use core::fmt::Debug;
use std::ops::Add;
use std::ops::Mul;

trait Field:
    Clone + Copy + Debug + Sized + Add<Output = Self> + Mul<Output = Self> + From<u32>
{
}

// /// Source: https://arxiv.org/ftp/arxiv/papers/1407/1407.3383.pdf (function 16)
// macro_rules! mul_mod_fma {
//     ($t:ty) => {};
// }

pub mod numeric {
    use super::*;

    pub mod double_precision {
        use super::*;

        #[repr(transparent)]
        #[derive(Clone, Copy, Debug)]
        pub struct Fp(f64);

        impl Field for Fp {}

        impl Add for Fp {
            type Output = Self;

            fn add(self, rhs: Self) -> Self {
                todo!()
            }
        }

        impl Mul for Fp {
            type Output = Self;

            fn mul(self, rhs: Self) -> Self {
                todo!()
            }
        }

        impl From<u32> for Fp {
            fn from(value: u32) -> Self {
                todo!()
            }
        }
    }

    pub mod single_precision {
        use super::*;

        #[repr(transparent)]
        #[derive(Clone, Copy, Debug)]
        pub struct Fp([f32; 2]);

        impl Field for Fp {}

        impl Add for Fp {
            type Output = Self;

            fn add(self, rhs: Self) -> Self {
                todo!()
            }
        }

        impl Mul for Fp {
            type Output = Self;

            fn mul(self, rhs: Self) -> Self {
                todo!()
            }
        }

        impl From<u32> for Fp {
            fn from(value: u32) -> Self {
                todo!()
            }
        }
    }

    pub mod half_precision {
        use super::*;
        use crate::arithmetic::half_precision::U32;

        #[repr(transparent)]
        #[derive(Clone, Copy, Debug)]
        pub struct Fp(U32);

        impl Field for Fp {}

        impl Add for Fp {
            type Output = Self;

            fn add(self, rhs: Self) -> Self {
                todo!()
            }
        }

        impl Mul for Fp {
            type Output = Self;

            fn mul(self, rhs: Self) -> Self {
                todo!()
            }
        }

        impl From<u32> for Fp {
            fn from(value: u32) -> Self {
                todo!()
            }
        }
    }
}

mod integer {}
