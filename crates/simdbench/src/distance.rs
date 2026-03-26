// SPDX-License-Identifier: MIT

use std::arch::asm;

// =============================================================================
// AVX-512 implementations (512-bit ZMM registers, 16 f32s per op)
// =============================================================================

/// Horizontal sum of 16 f32s in a ZMM register to scalar.
/// Pattern embedded in each asm block:
///   vextractf32x8 ymm_hi, zmm, 1     // upper 256 bits
///   vaddps        ymm_lo, ymm_lo, ymm_hi  // 8 floats (but needs ymm alias of zmm lower)
///   vextractf128  xmm_hi, ymm_lo, 1  // upper 128 of that
///   vaddps        xmm_lo, xmm_lo, xmm_hi  // 4 floats
///   vmovhlps      xmm_hi, xmm_hi, xmm_lo
///   vaddps        xmm_lo, xmm_lo, xmm_hi  // 2 floats
///   vpshufd       xmm_hi, xmm_lo, 1
///   vaddss        xmm_lo, xmm_lo, xmm_hi  // scalar

/// Dot product using AVX-512F inline assembly.
/// Processes 16 floats at a time with 512-bit ZMM registers.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub unsafe fn dot_f32_avx512(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let n = a.len();
    let chunks = n / 16;
    let remainder = n % 16;

    let mut result: f32 = 0.0;

    if chunks > 0 {
        unsafe {
            asm!(
                // Zero accumulator
                "vpxord zmm0, zmm0, zmm0",

                "2:",
                "vmovups zmm1, [{pa}]",
                "vmovups zmm2, [{pb}]",
                "vfmadd231ps zmm0, zmm1, zmm2",
                "add {pa}, 64",
                "add {pb}, 64",
                "dec {count}",
                "jnz 2b",

                // Horizontal sum: zmm0 → scalar
                // 512 → 256
                "vextractf32x8 ymm1, zmm0, 1",
                "vaddps ymm0, ymm0, ymm1",
                // 256 → 128
                "vextractf128 xmm1, ymm0, 1",
                "vaddps xmm0, xmm0, xmm1",
                // 128 → scalar
                "vmovhlps xmm1, xmm1, xmm0",
                "vaddps xmm0, xmm0, xmm1",
                "vpshufd xmm1, xmm0, 0x01",
                "vaddss xmm0, xmm0, xmm1",

                "vmovss [{pout}], xmm0",

                pa = inout(reg) a.as_ptr() => _,
                pb = inout(reg) b.as_ptr() => _,
                count = inout(reg) chunks => _,
                pout = in(reg) &mut result as *mut f32,
                out("zmm0") _,
                out("zmm1") _,
                out("zmm2") _,
                options(nostack),
            );
        }
    }

    let tail = chunks * 16;
    for i in 0..remainder {
        result += a[tail + i] * b[tail + i];
    }
    result
}

/// Cosine distance using AVX-512F inline assembly.
/// Returns 1 - cos(a, b).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub unsafe fn cosine_f32_avx512(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let n = a.len();
    let chunks = n / 16;
    let remainder = n % 16;

    let mut dot: f32 = 0.0;
    let mut norm_a: f32 = 0.0;
    let mut norm_b: f32 = 0.0;

    if chunks > 0 {
        unsafe {
            asm!(
                // Zero three accumulators: zmm0=dot, zmm3=norm_a, zmm4=norm_b
                "vpxord zmm0, zmm0, zmm0",
                "vpxord zmm3, zmm3, zmm3",
                "vpxord zmm4, zmm4, zmm4",

                "2:",
                "vmovups zmm1, [{pa}]",
                "vmovups zmm2, [{pb}]",
                "vfmadd231ps zmm0, zmm1, zmm2",   // dot += a*b
                "vfmadd231ps zmm3, zmm1, zmm1",   // norm_a += a*a
                "vfmadd231ps zmm4, zmm2, zmm2",   // norm_b += b*b
                "add {pa}, 64",
                "add {pb}, 64",
                "dec {count}",
                "jnz 2b",

                // Horizontal sum for dot (zmm0)
                "vextractf32x8 ymm1, zmm0, 1",
                "vaddps ymm0, ymm0, ymm1",
                "vextractf128 xmm1, ymm0, 1",
                "vaddps xmm0, xmm0, xmm1",
                "vmovhlps xmm1, xmm1, xmm0",
                "vaddps xmm0, xmm0, xmm1",
                "vpshufd xmm1, xmm0, 0x01",
                "vaddss xmm0, xmm0, xmm1",
                "vmovss [{pdot}], xmm0",

                // Horizontal sum for norm_a (zmm3)
                "vextractf32x8 ymm1, zmm3, 1",
                "vaddps ymm3, ymm3, ymm1",
                "vextractf128 xmm1, ymm3, 1",
                "vaddps xmm3, xmm3, xmm1",
                "vmovhlps xmm1, xmm1, xmm3",
                "vaddps xmm3, xmm3, xmm1",
                "vpshufd xmm1, xmm3, 0x01",
                "vaddss xmm3, xmm3, xmm1",
                "vmovss [{pna}], xmm3",

                // Horizontal sum for norm_b (zmm4)
                "vextractf32x8 ymm1, zmm4, 1",
                "vaddps ymm4, ymm4, ymm1",
                "vextractf128 xmm1, ymm4, 1",
                "vaddps xmm4, xmm4, xmm1",
                "vmovhlps xmm1, xmm1, xmm4",
                "vaddps xmm4, xmm4, xmm1",
                "vpshufd xmm1, xmm4, 0x01",
                "vaddss xmm4, xmm4, xmm1",
                "vmovss [{pnb}], xmm4",

                pa = inout(reg) a.as_ptr() => _,
                pb = inout(reg) b.as_ptr() => _,
                count = inout(reg) chunks => _,
                pdot = in(reg) &mut dot as *mut f32,
                pna = in(reg) &mut norm_a as *mut f32,
                pnb = in(reg) &mut norm_b as *mut f32,
                out("zmm0") _,
                out("zmm1") _,
                out("zmm2") _,
                out("zmm3") _,
                out("zmm4") _,
                options(nostack),
            );
        }
    }

    let tail = chunks * 16;
    for i in 0..remainder {
        let ai = a[tail + i];
        let bi = b[tail + i];
        dot += ai * bi;
        norm_a += ai * ai;
        norm_b += bi * bi;
    }

    let denom = (norm_a * norm_b).sqrt();
    if denom == 0.0 { 1.0 } else { 1.0 - dot / denom }
}

/// Squared Euclidean distance using AVX-512F inline assembly.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub unsafe fn sqeuclidean_f32_avx512(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let n = a.len();
    let chunks = n / 16;
    let remainder = n % 16;

    let mut result: f32 = 0.0;

    if chunks > 0 {
        unsafe {
            asm!(
                "vpxord zmm0, zmm0, zmm0",

                "2:",
                "vmovups zmm1, [{pa}]",
                "vmovups zmm2, [{pb}]",
                "vsubps  zmm1, zmm1, zmm2",
                "vfmadd231ps zmm0, zmm1, zmm1",
                "add {pa}, 64",
                "add {pb}, 64",
                "dec {count}",
                "jnz 2b",

                // Horizontal sum
                "vextractf32x8 ymm1, zmm0, 1",
                "vaddps ymm0, ymm0, ymm1",
                "vextractf128 xmm1, ymm0, 1",
                "vaddps xmm0, xmm0, xmm1",
                "vmovhlps xmm1, xmm1, xmm0",
                "vaddps xmm0, xmm0, xmm1",
                "vpshufd xmm1, xmm0, 0x01",
                "vaddss xmm0, xmm0, xmm1",
                "vmovss [{pout}], xmm0",

                pa = inout(reg) a.as_ptr() => _,
                pb = inout(reg) b.as_ptr() => _,
                count = inout(reg) chunks => _,
                pout = in(reg) &mut result as *mut f32,
                out("zmm0") _,
                out("zmm1") _,
                out("zmm2") _,
                options(nostack),
            );
        }
    }

    let tail = chunks * 16;
    for i in 0..remainder {
        let d = a[tail + i] - b[tail + i];
        result += d * d;
    }
    result
}

// =============================================================================
// AVX (256-bit YMM) implementations — kept for comparison
// =============================================================================

/// Dot product using AVX+FMA inline assembly (256-bit, 8 floats per op).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,fma")]
pub unsafe fn dot_f32_avx(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let n = a.len();
    let chunks = n / 8;
    let remainder = n % 8;

    let mut result: f32 = 0.0;

    if chunks > 0 {
        unsafe {
            asm!(
                "vxorps {acc}, {acc}, {acc}",

                "2:",
                "vmovups {tmp0}, [{pa}]",
                "vmovups {tmp1}, [{pb}]",
                "vfmadd231ps {acc}, {tmp0}, {tmp1}",
                "add {pa}, 32",
                "add {pb}, 32",
                "dec {count}",
                "jnz 2b",

                "vextractf128 {lo}, {acc}, 0",
                "vextractf128 {hi}, {acc}, 1",
                "vaddps {lo}, {lo}, {hi}",
                "vmovhlps {hi}, {hi}, {lo}",
                "vaddps {lo}, {lo}, {hi}",
                "vpshufd {hi}, {lo}, 0x01",
                "vaddss {lo}, {lo}, {hi}",
                "vmovss [{pout}], {lo}",

                pa = inout(reg) a.as_ptr() => _,
                pb = inout(reg) b.as_ptr() => _,
                count = inout(reg) chunks => _,
                pout = in(reg) &mut result as *mut f32,
                acc = out(ymm_reg) _,
                tmp0 = out(ymm_reg) _,
                tmp1 = out(ymm_reg) _,
                lo = out(xmm_reg) _,
                hi = out(xmm_reg) _,
                options(nostack),
            );
        }
    }

    let tail_start = chunks * 8;
    for i in 0..remainder {
        result += a[tail_start + i] * b[tail_start + i];
    }
    result
}

/// Cosine distance using AVX+FMA inline assembly (256-bit).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,fma")]
pub unsafe fn cosine_f32_avx(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let n = a.len();
    let chunks = n / 8;
    let remainder = n % 8;

    let mut dot: f32 = 0.0;
    let mut norm_a: f32 = 0.0;
    let mut norm_b: f32 = 0.0;

    if chunks > 0 {
        unsafe {
            asm!(
                "vxorps {acc_dot}, {acc_dot}, {acc_dot}",
                "vxorps {acc_na}, {acc_na}, {acc_na}",
                "vxorps {acc_nb}, {acc_nb}, {acc_nb}",

                "2:",
                "vmovups {va}, [{pa}]",
                "vmovups {vb}, [{pb}]",
                "vfmadd231ps {acc_dot}, {va}, {vb}",
                "vfmadd231ps {acc_na}, {va}, {va}",
                "vfmadd231ps {acc_nb}, {vb}, {vb}",
                "add {pa}, 32",
                "add {pb}, 32",
                "dec {count}",
                "jnz 2b",

                "vextractf128 {lo}, {acc_dot}, 0",
                "vextractf128 {hi}, {acc_dot}, 1",
                "vaddps {lo}, {lo}, {hi}",
                "vmovhlps {hi}, {hi}, {lo}",
                "vaddps {lo}, {lo}, {hi}",
                "vpshufd {hi}, {lo}, 0x01",
                "vaddss {lo}, {lo}, {hi}",
                "vmovss [{pdot}], {lo}",

                "vextractf128 {lo}, {acc_na}, 0",
                "vextractf128 {hi}, {acc_na}, 1",
                "vaddps {lo}, {lo}, {hi}",
                "vmovhlps {hi}, {hi}, {lo}",
                "vaddps {lo}, {lo}, {hi}",
                "vpshufd {hi}, {lo}, 0x01",
                "vaddss {lo}, {lo}, {hi}",
                "vmovss [{pna}], {lo}",

                "vextractf128 {lo}, {acc_nb}, 0",
                "vextractf128 {hi}, {acc_nb}, 1",
                "vaddps {lo}, {lo}, {hi}",
                "vmovhlps {hi}, {hi}, {lo}",
                "vaddps {lo}, {lo}, {hi}",
                "vpshufd {hi}, {lo}, 0x01",
                "vaddss {lo}, {lo}, {hi}",
                "vmovss [{pnb}], {lo}",

                pa = inout(reg) a.as_ptr() => _,
                pb = inout(reg) b.as_ptr() => _,
                count = inout(reg) chunks => _,
                pdot = in(reg) &mut dot as *mut f32,
                pna = in(reg) &mut norm_a as *mut f32,
                pnb = in(reg) &mut norm_b as *mut f32,
                acc_dot = out(ymm_reg) _,
                acc_na = out(ymm_reg) _,
                acc_nb = out(ymm_reg) _,
                va = out(ymm_reg) _,
                vb = out(ymm_reg) _,
                lo = out(xmm_reg) _,
                hi = out(xmm_reg) _,
                options(nostack),
            );
        }
    }

    let tail = chunks * 8;
    for i in 0..remainder {
        let ai = a[tail + i];
        let bi = b[tail + i];
        dot += ai * bi;
        norm_a += ai * ai;
        norm_b += bi * bi;
    }

    let denom = (norm_a * norm_b).sqrt();
    if denom == 0.0 { 1.0 } else { 1.0 - dot / denom }
}

/// Squared Euclidean distance using AVX+FMA inline assembly (256-bit).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,fma")]
pub unsafe fn sqeuclidean_f32_avx(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let n = a.len();
    let chunks = n / 8;
    let remainder = n % 8;

    let mut result: f32 = 0.0;

    if chunks > 0 {
        unsafe {
            asm!(
                "vxorps {acc}, {acc}, {acc}",

                "2:",
                "vmovups {va}, [{pa}]",
                "vmovups {vb}, [{pb}]",
                "vsubps  {va}, {va}, {vb}",
                "vfmadd231ps {acc}, {va}, {va}",
                "add {pa}, 32",
                "add {pb}, 32",
                "dec {count}",
                "jnz 2b",

                "vextractf128 {lo}, {acc}, 0",
                "vextractf128 {hi}, {acc}, 1",
                "vaddps {lo}, {lo}, {hi}",
                "vmovhlps {hi}, {hi}, {lo}",
                "vaddps {lo}, {lo}, {hi}",
                "vpshufd {hi}, {lo}, 0x01",
                "vaddss {lo}, {lo}, {hi}",
                "vmovss [{pout}], {lo}",

                pa = inout(reg) a.as_ptr() => _,
                pb = inout(reg) b.as_ptr() => _,
                count = inout(reg) chunks => _,
                pout = in(reg) &mut result as *mut f32,
                acc = out(ymm_reg) _,
                va = out(ymm_reg) _,
                vb = out(ymm_reg) _,
                lo = out(xmm_reg) _,
                hi = out(xmm_reg) _,
                options(nostack),
            );
        }
    }

    let tail = chunks * 8;
    for i in 0..remainder {
        let d = a[tail + i] - b[tail + i];
        result += d * d;
    }
    result
}

// =============================================================================
// Scalar baselines
// =============================================================================

/// Pure scalar dot product for baseline comparison.
pub fn dot_f32_scalar(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Pure scalar cosine distance for baseline comparison.
pub fn cosine_f32_scalar(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0_f32;
    let mut na = 0.0_f32;
    let mut nb = 0.0_f32;
    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    let denom = (na * nb).sqrt();
    if denom == 0.0 { 1.0 } else { 1.0 - dot / denom }
}

/// Pure scalar squared Euclidean distance for baseline comparison.
pub fn sqeuclidean_f32_scalar(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum()
}
