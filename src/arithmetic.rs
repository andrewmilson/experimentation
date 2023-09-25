pub mod half_precision {
    use half::f16;
    use num_traits::Float;
    use std::ops::Add;
    use std::ops::Mul;

    // u8 arithmetic simulated with half precision
    #[repr(transparent)]
    #[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
    struct U8(f16);

    impl U8 {
        const MODULUS: f16 = f16::from_f32_const(256.0);
        const MODULUS_INV: f16 = f16::from_f32_const(1.0 / 256.0);
        const ONE: f16 = f16::ONE;

        #[inline]
        #[must_use]
        pub const fn new(v: u8) -> Self {
            Self(f16::from_f32_const(v as f32))
        }
    }

    impl From<U8> for u8 {
        #[inline]
        fn from(v: U8) -> u8 {
            v.0.to_f32() as u8
        }
    }

    impl Add for U8 {
        type Output = Self;

        fn add(self, rhs: Self) -> Self {
            let a = self.0 + rhs.0;
            Self(if a >= Self::MODULUS {
                a - Self::MODULUS
            } else {
                a
            })
        }
    }

    impl Mul for U8 {
        type Output = Self;

        fn mul(self, rhs: Self) -> Self {
            let h = self.0 * rhs.0;
            let l = mad_f16(self.0, rhs.0, -h);
            let b = h * Self::MODULUS_INV;
            let c = b.trunc();
            let d = mad_f16(-c, Self::MODULUS, h);
            let e = d + l;
            Self(if e >= Self::MODULUS {
                e - Self::MODULUS
            } else if e < f16::ZERO {
                e + Self::MODULUS
            } else {
                e
            })
        }
    }

    /// Calculates the fused multiply add: `a * b + c`
    #[inline]
    #[must_use]
    fn mad_f16(a: f16, b: f16, c: f16) -> f16 {
        let a = f16::to_f32(a);
        let b = f16::to_f32(b);
        let c = f16::to_f32(c);
        f16::from_f32(a.mul_add(b, c))
    }

    /// u32 arithmetic simulated with half precision
    #[repr(transparent)]
    #[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
    pub struct U32([U8; 4]);

    impl U32 {
        #[inline]
        #[must_use]
        pub const fn new(v: u32) -> Self {
            Self([
                U8::new(v as u8),
                U8::new((v >> 8) as u8),
                U8::new((v >> 16) as u8),
                U8::new((v >> 24) as u8),
            ])
        }
    }

    impl Add for U32 {
        type Output = Self;

        fn add(self, rhs: Self) -> Self::Output {
            let mut l0 = self.0[0].0 + rhs.0[0].0;
            let mut l1 = self.0[1].0 + rhs.0[1].0;
            let mut l2 = self.0[2].0 + rhs.0[2].0;
            let mut l3 = self.0[3].0 + rhs.0[3].0;

            if l0 >= U8::MODULUS {
                l0 -= U8::MODULUS;
                l1 += U8::ONE;
            }

            if l1 >= U8::MODULUS {
                l1 -= U8::MODULUS;
                l2 += U8::ONE;
            }

            if l2 >= U8::MODULUS {
                l2 -= U8::MODULUS;
                l3 += U8::ONE;
            }

            if l3 >= U8::MODULUS {
                l3 -= U8::MODULUS;
            }

            Self([U8(l0), U8(l1), U8(l2), U8(l3)])
        }
    }

    impl Mul for U32 {
        type Output = Self;

        fn mul(self, rhs: Self) -> Self::Output {
            todo!()
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        const U32_EDGE_CASES: [u32; 10] =
            [0, 1, 2, 3, 5, 0xFF, 0xFF00, 0xFF0000, 0xFF000000, u32::MAX];

        #[test]
        fn simulated_u8_multiplication() {
            for a in 0..=u8::MAX {
                for b in 0..=u8::MAX {
                    let expected = U8::new(a * b);
                    let actual = U8::new(a) * U8::new(b);
                    assert_eq!(expected, actual, "mismatch: `{a} * {b}`");
                }
            }
        }

        #[test]
        #[ignore = "unimplemented"]
        fn simulated_u32_multiplication() {
            for a in U32_EDGE_CASES {
                for b in U32_EDGE_CASES {
                    let expected = U32::new(a * b);
                    let actual = U32::new(a) * U32::new(b);
                    assert_eq!(expected, actual, "mismatch: `{a} + {b}`");
                }
            }
        }

        #[test]
        fn simulated_u8_addition() {
            for a in 0..=u8::MAX {
                for b in 0..=u8::MAX {
                    let expected = U8::new(a + b);
                    let actual = U8::new(a) + U8::new(b);
                    assert_eq!(expected, actual, "mismatch: `{a} + {b}`");
                }
            }
        }

        #[test]
        fn simulated_u32_addition() {
            for a in U32_EDGE_CASES {
                for b in U32_EDGE_CASES {
                    let expected = U32::new(a + b);
                    let actual = U32::new(a) + U32::new(b);
                    assert_eq!(expected, actual, "mismatch: `{a} + {b}`");
                }
            }
        }
    }
}

pub mod single_precision {
    use rand::distributions::Standard;
    use rand::prelude::Distribution;
    use std::ops::Add;
    use std::ops::Mul;

    #[repr(transparent)]
    #[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
    pub struct U16(f32);

    impl U16 {
        const MODULUS: f32 = 65536.0;
        const MODULUS_INV: f32 = 1.0 / Self::MODULUS;
        const ZERO: Self = U16(0.0);
        const ONE: Self = U16(1.0);

        #[inline]
        #[must_use]
        pub const fn new(v: u16) -> Self {
            Self(v as f32)
        }
    }

    impl Add for U16 {
        type Output = Self;

        fn add(self, rhs: Self) -> Self {
            let a = self.0 + rhs.0;
            Self(if a >= Self::MODULUS {
                a - Self::MODULUS
            } else {
                a
            })
        }
    }

    impl Mul for U16 {
        type Output = Self;

        fn mul(self, rhs: Self) -> Self {
            let h = self.0 * rhs.0;
            let l = self.0.mul_add(rhs.0, -h);
            let b = h * Self::MODULUS_INV;
            let c = b.trunc();
            let d = (-c).mul_add(Self::MODULUS, h);
            let e = d + l;
            Self(if e >= Self::MODULUS {
                e - Self::MODULUS
            } else if e < 0.0 {
                e + Self::MODULUS
            } else {
                e
            })
        }
    }

    /// Stores a u32 across two f32s as `x1 * 2^16 + x0`
    /// Where `x0 ∈ [0, 2^16)` and `x1 ∈ [0, 2^16)`
    #[repr(transparent)]
    #[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
    pub struct U32([U16; 2]);

    impl U32 {
        #[inline]
        #[must_use]
        pub const fn new(v: u32) -> Self {
            Self([U16::new(v as u16), U16::new((v >> 16) as u16)])
        }
    }

    impl Add for U32 {
        type Output = Self;

        fn add(self, rhs: Self) -> Self::Output {
            let mut l0 = self.0[0].0 + rhs.0[0].0;
            let mut l1 = self.0[1].0 + rhs.0[1].0;

            if l0 >= U16::MODULUS {
                l0 -= U16::MODULUS;
                l1 += 1.0;
            }

            if l1 >= U16::MODULUS {
                l1 -= U16::MODULUS;
            }

            Self([U16(l0), U16(l1)])
        }
    }

    impl Mul for U32 {
        type Output = Self;
        /// Adapted from: https://github.com/calccrypto/uint128_t/blob/master/uint128_t.cpp
        fn mul(self, rhs: Self) -> Self::Output {
            // // split values into 4 32-bit parts
            // let top = {
            //     let v0 = self.0[0].0 / 256.0;
            //     let v1 = self.0[1].0 / 256.0;
            //     [
            //         v1.trunc(),
            //         v1.fract() * 256.0,
            //         v0.trunc(),
            //         v0.fract() * 256.0,
            //     ]
            //     .map(U16)
            // };
            // let bottom = {
            //     let v0 = rhs.0[0].0 / 256.0;
            //     let v1 = rhs.0[1].0 / 256.0;
            //     [
            //         v1.trunc(),
            //         v1.fract() * 256.0,
            //         v0.trunc(),
            //         v0.fract() * 256.0,
            //     ]
            //     .map(U16)
            // };

            // let mut prod = [[0.0; 4]; 4];

            // // multiply each component of the values
            // for y in (0..4).rev() {
            //     for x in (0..4).rev() {
            //         prod[3 - x][y] = (top[x] * bottom[y]).0;
            //     }
            // }

            // split values into 4 32-bit parts
            let top = {
                let v0 = self.0[0].0 / 256.0;
                let v1 = self.0[1].0 / 256.0;
                [
                    v1.trunc() / 16.0,
                    v1.fract() * 16.0,
                    v0.trunc() / 16.0,
                    v0.fract() * 16.0,
                ]
            };
            let bottom = {
                let v0 = rhs.0[0].0 / 256.0;
                let v1 = rhs.0[1].0 / 256.0;
                [
                    v1.trunc() / 16.0,
                    v1.fract() * 16.0,
                    v0.trunc() / 16.0,
                    v0.fract() * 16.0,
                ]
            };

            let mut prod = [[0.0; 4]; 4];

            // // multiply each component of the values
            // for y in (0..4).rev() {
            //     for x in (0..4).rev() {
            //         // prod[3 - x][y] = (top[x] * bottom[y]).0;
            //         let h = top[x].0 * bottom[y].0;
            //         let l = top[x].0.mul_add(bottom[y].0, -h);
            //         let b = h * U16::MODULUS_INV;
            //         let c = b.trunc();
            //         let d = (-c).mul_add(U16::MODULUS, h);
            //         prod[3 - x][y] = d + l;
            //     }
            // }

            #[inline]
            fn mul(a: f32, b: f32) -> f32 {
                let h = a * b;
                let l = a.mul_add(b, -h);
                let b = h * U16::MODULUS_INV;
                let c = b.trunc();
                let d = (-c).mul_add(U16::MODULUS, h);
                d + l
            }

            prod[0][3] = mul(top[3], bottom[3]);
            prod[0][2] = mul(top[3], bottom[2]);
            prod[0][1] = mul(top[3], bottom[1]);
            prod[0][0] = mul(top[3], bottom[0]);

            prod[1][3] = mul(top[2], bottom[3]);
            prod[1][2] = mul(top[2], bottom[2]);
            prod[1][1] = mul(top[2], bottom[1]);

            prod[2][3] = mul(top[1], bottom[3]);
            prod[2][2] = mul(top[1], bottom[2]);

            prod[3][3] = mul(top[0], bottom[3]);

            // for row in &mut prod {
            //     for v in row {
            //         *v /= 256.0;
            //     }
            // }

            // first row
            let mut fourth32 = prod[0][3].fract() * 256.0;
            let mut third32 = prod[0][2].fract().mul_add(256.0, prod[0][3].trunc());
            let mut second32 = prod[0][1].fract().mul_add(256.0, prod[0][2].trunc());
            let mut first32 = prod[0][0].fract().mul_add(256.0, prod[0][1].trunc());

            // second row
            third32 = prod[1][3].fract().mul_add(256.0, third32);
            second32 += prod[1][2].fract().mul_add(256.0, prod[1][3].trunc());
            first32 += prod[1][1].fract().mul_add(256.0, prod[1][2].trunc());

            // third row
            second32 = prod[2][3].fract().mul_add(256.0, second32);
            first32 += prod[2][2].fract().mul_add(256.0, prod[2][3].trunc());

            // fourth row
            first32 = prod[3][3].fract().mul_add(256.0, first32);

            let v0 = third32.mul_add(256.0, fourth32) / 65536.0;
            let v1 = first32.mul_add(256.0, second32) / 65536.0;

            let l0 = v0.fract() * 65536.0;
            let mut l1 = v1.fract().mul_add(65536.0, v0.trunc());

            if l1 >= 65536.0 {
                l1 -= 65536.0;
            }

            // combine components
            U32([U16(l0), U16(l1)])
        }
    }

    impl Distribution<U32> for Standard {
        fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> U32 {
            U32::new(self.sample(rng))
        }
    }

    /// Stores a u32 across two f32s as `x1 * 2^11 + x0`
    /// Where `x0 ∈ [0, 2^11)` and `x1 ∈ [0, 2^21)`
    #[repr(transparent)]
    #[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
    pub struct U31([f32; 2]);

    impl U31 {
        const MASK_11_BITS: f32 = 0b11111111111 as f32;
        const MASK_20_BITS: f32 = 0b11111111111111111111 as f32;

        #[inline]
        #[must_use]
        pub const fn new(v: u32) -> Self {
            Self([
                (v & Self::MASK_11_BITS as u32) as f32,
                ((v >> 11) & Self::MASK_20_BITS as u32) as f32,
            ])
        }
    }

    impl Add for U31 {
        type Output = Self;

        fn add(self, rhs: Self) -> Self::Output {
            let mut l0 = self.0[0] + rhs.0[0];
            let mut l1 = self.0[1] + rhs.0[1];

            if l0 >= (Self::MASK_11_BITS + 1.0) {
                l0 -= Self::MASK_11_BITS + 1.0;
                l1 += 1.0;
            }

            if l1 >= (Self::MASK_20_BITS + 1.0) {
                l1 -= Self::MASK_20_BITS + 1.0;
            }

            Self([l0, l1])
        }
    }

    impl Mul for U31 {
        type Output = Self;

        /// Adapted from: https://github.com/calccrypto/uint128_t/blob/master/uint128_t.cpp
        fn mul(self, rhs: Self) -> Self {
            let tmp0 = (self.0[0] * rhs.0[0]) / (Self::MASK_11_BITS + 1.0);

            let l0 = tmp0.fract() * (Self::MASK_11_BITS + 1.0);
            let mut l1 = tmp0.trunc();

            /// Multiplication `a * b % 2^20`
            /// Where `a, b ∈ [0, 2^20)`
            #[inline]
            fn mul<const MODULUS: u32>(a: f32, b: f32) -> f32 {
                let h = a * b;
                let l = a.mul_add(b, -h);
                let b = h * (1.0 / MODULUS as f32);
                let c = b.trunc();
                let d = (-c).mul_add(MODULUS as f32, h);
                d + l
            }

            l1 += mul::<{ Self::MASK_20_BITS as u32 + 1 }>(self.0[0], rhs.0[1]);
            l1 += mul::<{ Self::MASK_20_BITS as u32 + 1 }>(self.0[1], rhs.0[0]);

            if l1 >= (Self::MASK_20_BITS + 1.0) {
                l1 -= Self::MASK_20_BITS + 1.0;
            }

            let tmp_lha = (self.0[1] / 512.0).fract() * (512.0 * 64.0);
            let tmp_rhs = (rhs.0[1] / 512.0).fract() * (512.0 * 32.0);
            l1 += mul::<{ Self::MASK_20_BITS as u32 + 1 }>(tmp_lha, tmp_rhs);

            if l1 >= (Self::MASK_20_BITS + 1.0) {
                l1 -= Self::MASK_20_BITS + 1.0;
            }

            U31([l0, l1])
        }
    }

    impl Distribution<U31> for Standard {
        fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> U31 {
            U31::new(self.sample(rng))
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use rand::rngs::StdRng;
        use rand::Rng;
        use rand::SeedableRng;

        const MASK_31_BITS: u32 = 0b1111111111111111111111111111111;

        const U16_EDGE_CASES: [u16; 10] = [0, 1, 2, 3, 5, 7, 8, 0xFF, 0xFF00, 0xFFFF];

        const U32_EDGE_CASES: [u32; 10] =
            [0, 1, 2, 3, 5, 0xFF, 0xFF00, 0xFF0000, 0xFF000000, u32::MAX];

        const U31_EDGE_CASES: [u32; 10] = [
            0,
            1,
            2,
            3,
            5,
            0xFF,
            0xFF00,
            0xFF0000,
            0xFF000000 >> 1,
            u32::MAX >> 1,
        ];

        #[test]
        fn simulated_u16_multiplication() {
            for a in U16_EDGE_CASES {
                for b in U16_EDGE_CASES {
                    let expected = U16::new(a.overflowing_mul(b).0);
                    let actual = U16::new(a) * U16::new(b);
                    assert_eq!(expected, actual, "mismatch: `{a} * {b}`");
                }
            }
        }

        #[test]
        fn simulated_u32_multiplication() {
            let mut rng = StdRng::from_seed([1; 32]);
            let edge_cases = (0..1024).map(|_| rng.gen()).collect::<Vec<u32>>();
            for &a in &edge_cases {
                for &b in &edge_cases {
                    let expected = U32::new(a.overflowing_mul(b).0);
                    let actual = U32::new(a) * U32::new(b);
                    assert_eq!(expected, actual, "mismatch: `{a} * {b}`");
                }
            }
        }

        #[test]
        fn simulated_u31_multiplication() {
            let mut rng = StdRng::from_seed([1; 32]);
            let edge_cases = (0..2048)
                .map(|_| rng.gen::<u32>() & MASK_31_BITS)
                .collect::<Vec<u32>>();
            for &a in &edge_cases {
                for &b in &edge_cases {
                    let expected = U31::new(a.overflowing_mul(b).0 & MASK_31_BITS);
                    let actual = U31::new(a) * U31::new(b);
                    assert_eq!(expected, actual, "mismatch: `{a} * {b}`");
                }
            }
        }

        #[test]
        fn simulated_u16_addition() {
            for a in U16_EDGE_CASES {
                for b in U16_EDGE_CASES {
                    let expected = U16::new(a.overflowing_add(b).0);
                    let actual = U16::new(a) + U16::new(b);
                    assert_eq!(expected, actual, "mismatch: `{a} + {b}`");
                }
            }
        }

        #[test]
        fn simulated_u32_addition() {
            for a in U32_EDGE_CASES {
                for b in U32_EDGE_CASES {
                    let expected = U32::new(a.overflowing_add(b).0);
                    let actual = U32::new(a) + U32::new(b);
                    assert_eq!(expected, actual, "mismatch: `{a} + {b}`");
                }
            }
        }

        #[test]
        fn simulated_u31_addition() {
            for a in U31_EDGE_CASES {
                for b in U31_EDGE_CASES {
                    let expected = U31::new(a.overflowing_add(b).0 & MASK_31_BITS);
                    let actual = U31::new(a) + U31::new(b);
                    assert_eq!(expected, actual, "mismatch: `{a} + {b}`");
                }
            }
        }
    }
}

pub mod double_precision {
    use rand::distributions::Standard;
    use rand::prelude::Distribution;
    use std::ops::Add;
    use std::ops::Mul;

    #[repr(transparent)]
    #[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
    pub struct U32(f64);

    impl U32 {
        const MODULUS: f64 = 4294967296.0;
        const MODULUS_INV: f64 = 1.0 / Self::MODULUS;
        const ZERO: Self = U32(0.0);
        const ONE: Self = U32(1.0);

        #[inline]
        #[must_use]
        pub const fn new(v: u32) -> Self {
            Self(v as f64)
        }
    }

    impl Add for U32 {
        type Output = Self;

        fn add(self, rhs: Self) -> Self {
            let a = self.0 + rhs.0;
            Self(if a >= Self::MODULUS {
                a - Self::MODULUS
            } else {
                a
            })
        }
    }

    impl Mul for U32 {
        type Output = Self;

        fn mul(self, rhs: Self) -> Self {
            let h = self.0 * rhs.0;
            let l = self.0.mul_add(rhs.0, -h);
            let b = h * Self::MODULUS_INV;
            let c = b.trunc();
            let d = (-c).mul_add(Self::MODULUS, h);
            let e = d + l;
            Self(if e >= Self::MODULUS {
                e - Self::MODULUS
            } else if e < 0.0 {
                e + Self::MODULUS
            } else {
                e
            })
        }
    }

    impl Distribution<U32> for Standard {
        fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> U32 {
            U32::new(self.sample(rng))
        }
    }
}
