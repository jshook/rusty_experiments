// SPDX-License-Identifier: MIT

use std::f64::consts::PI;

use simdbench::svd2x2::{svd_2x2_simd, svd_2x2_scalar, Svd2x2};

fn mat2_mul(a: &[f64; 4], b: &[f64; 4]) -> [f64; 4] {
    [
        a[0] * b[0] + a[1] * b[2],
        a[0] * b[1] + a[1] * b[3],
        a[2] * b[0] + a[3] * b[2],
        a[2] * b[1] + a[3] * b[3],
    ]
}

fn verify_svd(m: &[f64; 4], svd: &Svd2x2) -> f64 {
    let s_mat = [svd.s[0], 0.0, 0.0, svd.s[1]];
    let us = mat2_mul(&svd.u, &s_mat);
    let recon = mat2_mul(&us, &svd.vt);
    let err: f64 = m.iter().zip(recon.iter()).map(|(a, b)| (a - b).powi(2)).sum();
    err.sqrt()
}

fn main() {
    #[cfg(target_arch = "aarch64")]
    let isa_label = "NEON";
    #[cfg(target_arch = "x86_64")]
    let isa_label = "SSE2";

    println!("=== 2x2 SVD via inline {} assembly ===\n", isa_label);

    let matrices: Vec<(&str, [f64; 4])> = vec![
        ("Symmetric", [3.0, 2.0, 2.0, 3.0]),
        ("Non-symmetric", [1.0, 2.0, 3.0, 4.0]),
        ("Rotation (30 deg)", {
            let a = PI / 6.0;
            [a.cos(), -a.sin(), a.sin(), a.cos()]
        }),
        ("Near-singular", [1.0, 2.0, 2.0, 4.0]),
    ];

    for (name, m) in &matrices {
        let svd = svd_2x2_simd(m);
        let svd_s = svd_2x2_scalar(m);
        println!("{}: S=[{:.4}, {:.4}]  err={:.2e}  scalar_check=[{:.4}, {:.4}]",
            name, svd.s[0], svd.s[1], verify_svd(m, &svd), svd_s.s[0], svd_s.s[1]);
    }

    use simdbench::distance::*;
    use simsimd::SpatialSimilarity;

    let a: Vec<f32> = (0..256).map(|i| i as f32).collect();
    let b: Vec<f32> = (0..256).map(|i| (255 - i) as f32).collect();

    println!("\n=== Vector distance: asm vs simsimd vs scalar ===\n");
    println!("  Vectors: a=[0..255], b=[255..0], len=256\n");

    #[cfg(target_arch = "aarch64")]
    unsafe {
        println!("  dot:  sve={:.1}  neon={:.1}  simsimd={:?}  scalar={:.1}",
            dot_f32_sve(&a, &b), dot_f32_neon(&a, &b),
            <f32 as SpatialSimilarity>::dot(&a, &b), dot_f32_scalar(&a, &b));
        println!("  cos:  sve={:.6}  neon={:.6}  simsimd={:?}  scalar={:.6}",
            cosine_f32_sve(&a, &b), cosine_f32_neon(&a, &b),
            <f32 as SpatialSimilarity>::cos(&a, &b), cosine_f32_scalar(&a, &b));
        println!("  l2sq: sve={:.1}  neon={:.1}  simsimd={:?}  scalar={:.1}",
            sqeuclidean_f32_sve(&a, &b), sqeuclidean_f32_neon(&a, &b),
            <f32 as SpatialSimilarity>::l2sq(&a, &b), sqeuclidean_f32_scalar(&a, &b));
    }

    #[cfg(target_arch = "x86_64")]
    unsafe {
        println!("  dot:  avx512={:.1}  avx={:.1}  simsimd={:?}  scalar={:.1}",
            dot_f32_avx512(&a, &b), dot_f32_avx(&a, &b),
            <f32 as SpatialSimilarity>::dot(&a, &b), dot_f32_scalar(&a, &b));
        println!("  cos:  avx512={:.6}  avx={:.6}  simsimd={:?}  scalar={:.6}",
            cosine_f32_avx512(&a, &b), cosine_f32_avx(&a, &b),
            <f32 as SpatialSimilarity>::cos(&a, &b), cosine_f32_scalar(&a, &b));
        println!("  l2sq: avx512={:.1}  avx={:.1}  simsimd={:?}  scalar={:.1}",
            sqeuclidean_f32_avx512(&a, &b), sqeuclidean_f32_avx(&a, &b),
            <f32 as SpatialSimilarity>::l2sq(&a, &b), sqeuclidean_f32_scalar(&a, &b));
    }

    println!("\n  Run benchmarks: RUSTFLAGS='-C target-cpu=native' cargo bench");
}
