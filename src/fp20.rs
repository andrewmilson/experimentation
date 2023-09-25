pub mod single_precision {
    use rand::distributions::Standard;
    use rand::prelude::Distribution;
    use std::fmt::Display;
    use std::ops::Add;
    use std::ops::Mul;

    #[repr(transparent)]
    #[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
    pub struct Fp(f32);

    impl Fp {
        const MODULUS: f32 = 0b111111111111111110111 as f32;
        const MODULUS_INV: f32 = 1.0 / Self::MODULUS;
        const ZERO: Self = Fp(0.0);
        const ONE: Self = Fp(1.0);

        #[inline]
        #[must_use]
        pub const fn new(v: u32) -> Self {
            debug_assert!(v < Self::MODULUS as u32);
            Self(v as f32)
        }
    }

    impl Display for Fp {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            self.0.fmt(f)
        }
    }

    impl Add for Fp {
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

    impl Mul for Fp {
        type Output = Self;

        fn mul(self, rhs: Self) -> Self {
            let h = self.0 * rhs.0;
            let l = self.0.mul_add(rhs.0, -h);
            let b = h * Self::MODULUS_INV;
            let c = b.floor();
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

    impl Distribution<Fp> for Standard {
        fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> Fp {
            // TODO: make sure sampling is done correctly
            let v: u32 = self.sample(rng);
            Fp::new(v % (Fp::MODULUS as u32))
        }
    }

    impl From<Fp> for super::integer::Fp {
        #[inline]
        fn from(value: Fp) -> Self {
            Self::new(value.0 as u32)
        }
    }

    impl From<Fp> for u32 {
        #[inline]
        fn from(value: Fp) -> Self {
            value.0 as u32
        }
    }

    #[cfg(test)]
    mod tests {
        use super::super::integer;
        use super::*;
        use rand::rngs::StdRng;
        use rand::Rng;
        use rand::SeedableRng;

        const MODULUS: u32 = Fp::MODULUS as u32;

        #[test]
        fn multiplication() {
            let mut rng = StdRng::from_seed([1; 32]);
            let edge_cases = (0..2048)
                .map(|_| rng.gen::<u32>() % MODULUS)
                .collect::<Vec<u32>>();
            for &a in &edge_cases {
                for &b in &edge_cases {
                    let expected = Fp::new((a as u64 * b as u64 % MODULUS as u64) as u32);
                    let actual = Fp::new(a) * Fp::new(b);
                    assert_eq!(expected, actual, "mismatch: `{a} * {b}`");
                }
            }
        }

        #[test]
        fn addition() {
            let mut rng = StdRng::from_seed([1; 32]);
            let edge_cases = (0..1024)
                .map(|_| rng.gen::<u32>() % MODULUS)
                .collect::<Vec<u32>>();
            for &a in &edge_cases {
                for &b in &edge_cases {
                    let expected = Fp::new((a + b) % MODULUS);
                    let actual = Fp::new(a) + Fp::new(b);
                    assert_eq!(expected, actual, "mismatch: `{a} + {b}`");
                }
            }
        }
    }
}

pub mod integer {
    use rand::distributions::Standard;
    use rand::prelude::Distribution;
    use std::ops::Add;
    use std::ops::Mul;

    /// Pseudo-Mersenne prime field modulus `p = 2097143`
    const MODULUS: u32 = 0b111111111111111110111;

    /// Pseudo-Mersenne Reduction:
    /// <https://hal.sorbonne-universite.fr/hal-02883333/file/BaDueprintversion.pdf>
    const fn reduce(a: u64) -> u32 {
        const C: u32 = MODULUS.next_power_of_two() - MODULUS;
        let b = ((a >> 21) as u32) * C + (a as u32 & 0x1FFFFF);
        let mut r = (b >> 21) * C + (b & 0x1FFFFF);
        let mut r_prime = r + C;
        if r_prime >= (1 << 21) {
            r = r_prime - (1 << 21);
            r_prime = r + C;
            if r_prime >= (1 << 21) {
                return r_prime - (1 << 21);
            }
        }
        r
    }

    #[repr(transparent)]
    #[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
    pub struct Fp(u32);

    impl Fp {
        #[inline]
        #[must_use]
        pub const fn new(v: u32) -> Self {
            debug_assert!(v < MODULUS);
            Self(v)
        }
    }

    impl Add for Fp {
        type Output = Self;

        fn add(self, rhs: Self) -> Self {
            let a = self.0 + rhs.0;
            Self(if a >= MODULUS { a - MODULUS } else { a })
        }
    }

    impl Mul for Fp {
        type Output = Self;

        fn mul(self, rhs: Self) -> Self {
            Self(reduce(self.0 as u64 * rhs.0 as u64))
        }
    }

    impl Distribution<Fp> for Standard {
        fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> Fp {
            let v: u32 = self.sample(rng);
            Fp::new(v % MODULUS)
        }
    }

    impl From<Fp> for super::single_precision::Fp {
        #[inline]
        fn from(value: Fp) -> Self {
            Self::new(value.0)
        }
    }

    impl From<Fp> for super::double_precision::Fp {
        #[inline]
        fn from(value: Fp) -> Self {
            Self::new(value.0)
        }
    }

    impl From<Fp> for u32 {
        #[inline]
        fn from(value: Fp) -> Self {
            value.0
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use rand::rngs::StdRng;
        use rand::Rng;
        use rand::SeedableRng;

        #[test]
        fn multiplication() {
            let mut rng = StdRng::from_seed([1; 32]);
            let edge_cases = (0..1024)
                .map(|_| rng.gen::<u32>() % MODULUS)
                .collect::<Vec<u32>>();
            for &a in &edge_cases {
                for &b in &edge_cases {
                    let expected = Fp::new((a as u64 * b as u64 % MODULUS as u64) as u32);
                    let actual = Fp::new(a) * Fp::new(b);
                    assert_eq!(expected, actual, "mismatch: `{a} * {b}`");
                }
            }
        }

        #[test]
        fn multiply_by_zero() {
            let res = Fp::new(1) * Fp::new(0);

            assert_eq!(Fp::new(0), res);
        }

        #[test]
        fn addition() {
            let mut rng = StdRng::from_seed([1; 32]);
            let edge_cases = (0..1024)
                .map(|_| rng.gen::<u32>() % MODULUS)
                .collect::<Vec<u32>>();
            for &a in &edge_cases {
                for &b in &edge_cases {
                    let expected = Fp::new((a + b) % MODULUS);
                    let actual = Fp::new(a) + Fp::new(b);
                    assert_eq!(expected, actual, "mismatch: `{a} + {b}`");
                }
            }
        }
    }
}

pub mod double_precision {
    use rand::distributions::Standard;
    use rand::prelude::Distribution;
    use std::fmt::Display;
    use std::ops::Add;
    use std::ops::Mul;

    #[repr(transparent)]
    #[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
    pub struct Fp(f64);

    impl Fp {
        const MODULUS: f64 = 0b111111111111111110111 as f64;
        const MODULUS_INV: f64 = 1.0 / Self::MODULUS;
        const ZERO: Self = Fp(0.0);
        const ONE: Self = Fp(1.0);

        #[inline]
        #[must_use]
        pub const fn new(v: u32) -> Self {
            debug_assert!(v < Self::MODULUS as u32);
            Self(v as f64)
        }
    }

    impl Display for Fp {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            self.0.fmt(f)
        }
    }

    impl Add for Fp {
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

    impl Mul for Fp {
        type Output = Self;

        /// Source: https://arxiv.org/ftp/arxiv/papers/1407/1407.3383.pdf (function 14)
        fn mul(self, rhs: Self) -> Self {
            let a = self.0 * rhs.0;
            let b = a * Self::MODULUS_INV;
            let c = b.trunc();
            let d = a - c * Self::MODULUS;
            Self(d)
            // TODO: depending on the rounding mode
            // Self(if d >= Self::MODULUS {
            //     d - Self::MODULUS
            // } else if d < 0.0 {
            //     d + Self::MODULUS
            // } else {
            //     d
            // })
        }
    }

    impl Distribution<Fp> for Standard {
        fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> Fp {
            // TODO: make sure sampling is done correctly
            let v: u32 = self.sample(rng);
            Fp::new(v % (Fp::MODULUS as u32))
        }
    }

    impl From<Fp> for super::integer::Fp {
        #[inline]
        fn from(value: Fp) -> Self {
            Self::new(value.0 as u32)
        }
    }

    impl From<Fp> for u32 {
        #[inline]
        fn from(value: Fp) -> Self {
            value.0 as u32
        }
    }

    #[cfg(test)]
    mod tests {
        use super::super::integer;
        use super::*;
        use rand::rngs::StdRng;
        use rand::Rng;
        use rand::SeedableRng;

        const MODULUS: u32 = Fp::MODULUS as u32;

        #[test]
        fn multiplication() {
            let mut rng = StdRng::from_seed([1; 32]);
            let edge_cases = (0..1024)
                .map(|_| rng.gen::<u32>() % MODULUS)
                .collect::<Vec<u32>>();
            for &a in &edge_cases {
                for &b in &edge_cases {
                    let expected = Fp::new((a as u64 * b as u64 % MODULUS as u64) as u32);
                    let actual = Fp::new(a) * Fp::new(b);
                    assert_eq!(expected, actual, "mismatch: `{a} * {b}`");
                }
            }
        }

        #[test]
        fn addition() {
            let mut rng = StdRng::from_seed([1; 32]);
            let edge_cases = (0..1024)
                .map(|_| rng.gen::<u32>() % MODULUS)
                .collect::<Vec<u32>>();
            for &a in &edge_cases {
                for &b in &edge_cases {
                    let expected = Fp::new((a + b) % MODULUS);
                    let actual = Fp::new(a) + Fp::new(b);
                    assert_eq!(expected, actual, "mismatch: `{a} + {b}`");
                }
            }

            for a in ((1 << 16) - 1)..MODULUS {
                for b in 0..MODULUS {
                    let expected = Fp::from(integer::Fp::new(a) * integer::Fp::new(b));
                    let actual = Fp::new(a) * Fp::new(b);
                    assert_eq!(expected, actual, "mismatch: `{a} * {b}`");
                }
            }
        }
    }
}
