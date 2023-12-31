#![feature(array_chunks)]

use ark_experimentation::arithmetic::double_precision;
use ark_experimentation::arithmetic::single_precision;
use ark_experimentation::fp20;
use criterion::black_box;
use criterion::criterion_group;
use criterion::criterion_main;
use criterion::Criterion;
use p3_mersenne_31::Mersenne31;
use rand::distributions::Standard;
use rand::prelude::Distribution;
use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;
use std::ops::Add;
use std::ops::Mul;

fn bench_multiplication<T: Mul<Output = T> + Copy>(c: &mut Criterion, id: &str)
where
    Standard: Distribution<T>,
{
    let mut rng = StdRng::from_seed([1; 32]);
    let values = (0..256).map(|_| rng.gen()).collect::<Vec<T>>();
    c.bench_function(&format!("{id}/multiplication"), |b| {
        b.iter(|| {
            for &[a, b, c, d, e, f, g, h] in values.array_chunks() {
                black_box([a * b, c * d, e * f, g * h]);
            }
        });
    });
}

fn bench_addition<T: Add<Output = T> + Copy>(c: &mut Criterion, id: &str)
where
    Standard: Distribution<T>,
{
    let mut rng = StdRng::from_seed([1; 32]);
    let values = (0..256).map(|_| rng.gen()).collect::<Vec<T>>();
    c.bench_function(&format!("{id}/addition"), |b| {
        b.iter(|| {
            for &[a, b, c, d, e, f, g, h] in values.array_chunks() {
                black_box([a + b, c + d, e + f, g + h]);
            }
        });
    });
}

fn multiplication_benches(c: &mut Criterion) {
    bench_multiplication::<u32>(c, "native_u32");
    bench_multiplication::<Mersenne31>(c, "plonky3_mersenne_31");
    bench_multiplication::<single_precision::U31>(c, "f32_sim_u31");
    bench_multiplication::<single_precision::U32>(c, "f32_sim_u32");
    bench_multiplication::<double_precision::U32>(c, "f64_sim_u32");
    bench_multiplication::<fp20::single_precision::Fp>(c, "f32_pseudo_mersene_21");
    bench_multiplication::<fp20::double_precision::Fp>(c, "f64_pseudo_mersene_21");
    bench_multiplication::<fp20::integer::Fp>(c, "u32_pseudo_mersene_21");
}

fn addition_benches(c: &mut Criterion) {
    bench_addition::<u32>(c, "native_u32");
    bench_addition::<Mersenne31>(c, "plonky3_mersenne_31");
    bench_addition::<single_precision::U31>(c, "f32_sim_u31");
    bench_addition::<single_precision::U32>(c, "f32_sim_u32");
    bench_addition::<double_precision::U32>(c, "f64_sim_u32");
    bench_addition::<fp20::single_precision::Fp>(c, "f32_pseudo_mersene_21");
    bench_addition::<fp20::double_precision::Fp>(c, "f64_pseudo_mersene_21");
    bench_addition::<fp20::integer::Fp>(c, "u32_pseudo_mersene_21");
}

criterion_group!(benches, multiplication_benches, addition_benches);
criterion_main!(benches);
