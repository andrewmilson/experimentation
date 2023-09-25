/// Returns GCD and bezout coefficients such that `ax + by = gcd(a, b)`
/// Output is of the form: `(gcd, x, y)`
const fn xgcd(a: u64, b: u64) -> (u64, u64, u64) {
    if a == 0 {
        (b, 0, 1)
    } else {
        let (gcd, x, y) = xgcd(b % a, a);
        (gcd, y - (b / a) * x, x)
    }
}
