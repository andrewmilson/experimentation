pub mod arithmetic;
pub mod field;
pub mod fp20;
pub mod utils;

#[cfg(not(all(target_arch = "aarch64", target_os = "macos")))]
compile_error!("library only supported on apple silicon devices");

const MODULUS: f32 = 65537.0;
const U: f32 = 1.0 / MODULUS;

pub fn reduce_numeric_half(a: f32) -> f32 {
    let b = a * U;
    let c = b.trunc();
    let d = a - c * MODULUS;
    if d >= MODULUS {
        d - MODULUS
    } else if d < 0.0 {
        d + MODULUS
    } else {
        d
    }
}

/// Computes `(a_1 * a_2) mod p`
fn mul_mod_fma(a1: f32, a2: f32) -> f32 {
    let h = a1 * a2;
    let l = a1.mul_add(a2, -h);
    let b = h * U;
    let c = b.trunc();
    let d = (-c).mul_add(MODULUS, h);
    let e = d + l;
    if e >= MODULUS {
        e - MODULUS
    } else if e < 0.0 {
        e + MODULUS
    } else {
        e
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_float() {
        let v = 0.15625f32.to_bits();
        println!("v: {:032b}", v);

        // for i in 0..65537u64 {
        //     for j in 0..65537u64 {
        //         let expected = (i * j) % 65537;
        //         let actual = mul_mod_fma(i as f32, j as f32);
        //         assert_eq!(
        //             expected as f32, actual,
        //             "mismatch at `{i} * {j} mod {MODULUS}`"
        //         );
        //     }
        // }
    }
}
