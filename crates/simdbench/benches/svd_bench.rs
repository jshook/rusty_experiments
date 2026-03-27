// SPDX-License-Identifier: MIT

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use simdbench::distance;
use simdbench::svd2x2;
use simsimd::SpatialSimilarity;

fn bench_svd(c: &mut Criterion) {
    let m = [1.0_f64, 2.0, 3.0, 4.0];

    let mut group = c.benchmark_group("svd_2x2");
    group.bench_function("inline_asm_simd", |b| {
        b.iter(|| svd2x2::svd_2x2_simd(black_box(&m)))
    });
    group.bench_function("pure_scalar", |b| {
        b.iter(|| svd2x2::svd_2x2_scalar(black_box(&m)))
    });
    group.finish();
}

fn bench_dot_product(c: &mut Criterion) {
    let mut group = c.benchmark_group("dot_product_f32");

    for size in [64, 256, 1024, 4096] {
        let a: Vec<f32> = (0..size).map(|i| (i as f32) * 0.001).collect();
        let b: Vec<f32> = (0..size).map(|i| (i as f32) * 0.002 + 0.5).collect();

        #[cfg(target_arch = "aarch64")]
        {
            group.bench_with_input(BenchmarkId::new("sve_asm", size), &size, |bench, _| {
                bench.iter(|| unsafe { distance::dot_f32_sve(black_box(&a), black_box(&b)) })
            });

            group.bench_with_input(BenchmarkId::new("neon_asm", size), &size, |bench, _| {
                bench.iter(|| unsafe { distance::dot_f32_neon(black_box(&a), black_box(&b)) })
            });
        }

        #[cfg(target_arch = "x86_64")]
        {
            group.bench_with_input(BenchmarkId::new("avx512_asm", size), &size, |bench, _| {
                bench.iter(|| unsafe { distance::dot_f32_avx512(black_box(&a), black_box(&b)) })
            });

            group.bench_with_input(BenchmarkId::new("avx_asm", size), &size, |bench, _| {
                bench.iter(|| unsafe { distance::dot_f32_avx(black_box(&a), black_box(&b)) })
            });
        }

        group.bench_with_input(BenchmarkId::new("simsimd", size), &size, |bench, _| {
            bench.iter(|| <f32 as SpatialSimilarity>::dot(black_box(&a), black_box(&b)))
        });

        group.bench_with_input(BenchmarkId::new("scalar", size), &size, |bench, _| {
            bench.iter(|| distance::dot_f32_scalar(black_box(&a), black_box(&b)))
        });
    }
    group.finish();
}

fn bench_cosine(c: &mut Criterion) {
    let mut group = c.benchmark_group("cosine_f32");

    for size in [64, 256, 1024, 4096] {
        let a: Vec<f32> = (0..size).map(|i| (i as f32) * 0.001).collect();
        let b: Vec<f32> = (0..size).map(|i| (i as f32) * 0.002 + 0.5).collect();

        #[cfg(target_arch = "aarch64")]
        {
            group.bench_with_input(BenchmarkId::new("sve_asm", size), &size, |bench, _| {
                bench.iter(|| unsafe { distance::cosine_f32_sve(black_box(&a), black_box(&b)) })
            });

            group.bench_with_input(BenchmarkId::new("neon_asm", size), &size, |bench, _| {
                bench.iter(|| unsafe { distance::cosine_f32_neon(black_box(&a), black_box(&b)) })
            });
        }

        #[cfg(target_arch = "x86_64")]
        {
            group.bench_with_input(BenchmarkId::new("avx512_asm", size), &size, |bench, _| {
                bench.iter(|| unsafe { distance::cosine_f32_avx512(black_box(&a), black_box(&b)) })
            });

            group.bench_with_input(BenchmarkId::new("avx_asm", size), &size, |bench, _| {
                bench.iter(|| unsafe { distance::cosine_f32_avx(black_box(&a), black_box(&b)) })
            });
        }

        group.bench_with_input(BenchmarkId::new("simsimd", size), &size, |bench, _| {
            bench.iter(|| <f32 as SpatialSimilarity>::cos(black_box(&a), black_box(&b)))
        });

        group.bench_with_input(BenchmarkId::new("scalar", size), &size, |bench, _| {
            bench.iter(|| distance::cosine_f32_scalar(black_box(&a), black_box(&b)))
        });
    }
    group.finish();
}

fn bench_sqeuclidean(c: &mut Criterion) {
    let mut group = c.benchmark_group("sqeuclidean_f32");

    for size in [64, 256, 1024, 4096] {
        let a: Vec<f32> = (0..size).map(|i| (i as f32) * 0.001).collect();
        let b: Vec<f32> = (0..size).map(|i| (i as f32) * 0.002 + 0.5).collect();

        #[cfg(target_arch = "aarch64")]
        {
            group.bench_with_input(BenchmarkId::new("sve_asm", size), &size, |bench, _| {
                bench.iter(|| unsafe { distance::sqeuclidean_f32_sve(black_box(&a), black_box(&b)) })
            });

            group.bench_with_input(BenchmarkId::new("neon_asm", size), &size, |bench, _| {
                bench.iter(|| unsafe { distance::sqeuclidean_f32_neon(black_box(&a), black_box(&b)) })
            });
        }

        #[cfg(target_arch = "x86_64")]
        {
            group.bench_with_input(BenchmarkId::new("avx512_asm", size), &size, |bench, _| {
                bench.iter(|| unsafe { distance::sqeuclidean_f32_avx512(black_box(&a), black_box(&b)) })
            });

            group.bench_with_input(BenchmarkId::new("avx_asm", size), &size, |bench, _| {
                bench.iter(|| unsafe { distance::sqeuclidean_f32_avx(black_box(&a), black_box(&b)) })
            });
        }

        group.bench_with_input(BenchmarkId::new("simsimd", size), &size, |bench, _| {
            bench.iter(|| <f32 as SpatialSimilarity>::l2sq(black_box(&a), black_box(&b)))
        });

        group.bench_with_input(BenchmarkId::new("scalar", size), &size, |bench, _| {
            bench.iter(|| distance::sqeuclidean_f32_scalar(black_box(&a), black_box(&b)))
        });
    }
    group.finish();
}

criterion_group!(benches, bench_svd, bench_dot_product, bench_cosine, bench_sqeuclidean);
criterion_main!(benches);
