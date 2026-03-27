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

// =============================================================================
// AArch64 SVE implementations (scalable vector length, VL-agnostic)
//
// Uses WHILELT + predication for clean tail handling — no scalar cleanup loop.
// Adapts automatically to any SVE vector length (128-bit on Graviton 3/4,
// 256-bit on A64FX/Graviton future, etc.).
// =============================================================================

/// Dot product using SVE inline assembly.
/// VL-agnostic: works with any SVE vector width, handles arbitrary lengths
/// via predication (no scalar tail loop needed).
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "sve")]
pub unsafe fn dot_f32_sve(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let n = a.len();
    let mut result: f32 = 0.0;

    if n > 0 {
        unsafe {
            asm!(
                "mov x8, #0",
                "whilelt p0.s, x8, {n}",
                // Two accumulators for ILP across dual SVE pipes
                "mov z0.s, #0",
                "mov z3.s, #0",
                // Check if we have enough for 2x unrolled
                "cntw x9",
                "lsl  x10, x9, #1",
                "cmp  {n}, x10",
                "b.lt 3f",

                // 2x unrolled main loop
                "ptrue p1.s",
                "2:",
                "ld1w {{z1.s}}, p1/z, [{pa}, x8, lsl #2]",
                "ld1w {{z2.s}}, p1/z, [{pb}, x8, lsl #2]",
                "fmla z0.s, p1/m, z1.s, z2.s",
                "add  x11, x8, x9",
                "ld1w {{z4.s}}, p1/z, [{pa}, x11, lsl #2]",
                "ld1w {{z5.s}}, p1/z, [{pb}, x11, lsl #2]",
                "fmla z3.s, p1/m, z4.s, z5.s",
                "add  x8, x8, x10",
                "sub  x12, {n}, x10",
                "cmp  x8, x12",
                "b.ls 2b",

                // Predicated tail loop for remaining elements
                "3:",
                "whilelt p0.s, x8, {n}",
                "b.none 4f",
                "5:",
                "ld1w {{z1.s}}, p0/z, [{pa}, x8, lsl #2]",
                "ld1w {{z2.s}}, p0/z, [{pb}, x8, lsl #2]",
                "fmla z0.s, p0/m, z1.s, z2.s",
                "incw x8",
                "whilelt p0.s, x8, {n}",
                "b.first 5b",

                "4:",
                // Merge accumulators and horizontal sum
                "fadd z0.s, z0.s, z3.s",
                "ptrue p0.s",
                "faddv s0, p0, z0.s",
                "str s0, [{pout}]",

                n = in(reg) n as i64,
                pa = in(reg) a.as_ptr(),
                pb = in(reg) b.as_ptr(),
                pout = in(reg) &mut result as *mut f32,
                out("x8") _, out("x9") _, out("x10") _, out("x11") _, out("x12") _,
                out("v0") _, out("v1") _, out("v2") _,
                out("v3") _, out("v4") _, out("v5") _,
                options(nostack),
            );
        }
    }
    result
}

/// Cosine distance using SVE inline assembly. Returns 1 - cos(a, b).
/// VL-agnostic with dual-accumulator unrolling.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "sve")]
pub unsafe fn cosine_f32_sve(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let n = a.len();
    let mut dot: f32 = 0.0;
    let mut norm_a: f32 = 0.0;
    let mut norm_b: f32 = 0.0;

    if n > 0 {
        unsafe {
            asm!(
                "mov x8, #0",
                // Zero accumulators: z0=dot, z3=norm_a, z6=norm_b
                "mov z0.s, #0",
                "mov z3.s, #0",
                "mov z6.s, #0",

                "whilelt p0.s, x8, {n}",
                "1:",
                "ld1w {{z1.s}}, p0/z, [{pa}, x8, lsl #2]",
                "ld1w {{z2.s}}, p0/z, [{pb}, x8, lsl #2]",
                "fmla z0.s, p0/m, z1.s, z2.s",
                "fmla z3.s, p0/m, z1.s, z1.s",
                "fmla z6.s, p0/m, z2.s, z2.s",
                "incw x8",
                "whilelt p0.s, x8, {n}",
                "b.first 1b",

                // Horizontal sums
                "ptrue p0.s",
                "faddv s0, p0, z0.s",
                "str s0, [{pdot}]",
                "faddv s3, p0, z3.s",
                "str s3, [{pna}]",
                "faddv s6, p0, z6.s",
                "str s6, [{pnb}]",

                n = in(reg) n as i64,
                pa = in(reg) a.as_ptr(),
                pb = in(reg) b.as_ptr(),
                pdot = in(reg) &mut dot as *mut f32,
                pna = in(reg) &mut norm_a as *mut f32,
                pnb = in(reg) &mut norm_b as *mut f32,
                out("x8") _,
                out("v0") _, out("v1") _, out("v2") _,
                out("v3") _, out("v6") _,
                options(nostack),
            );
        }
    }

    let denom = (norm_a * norm_b).sqrt();
    if denom == 0.0 { 1.0 } else { 1.0 - dot / denom }
}

/// Squared Euclidean distance using SVE inline assembly.
/// VL-agnostic with predicated tail handling.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "sve")]
pub unsafe fn sqeuclidean_f32_sve(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let n = a.len();
    let mut result: f32 = 0.0;

    if n > 0 {
        unsafe {
            asm!(
                "mov x8, #0",
                "mov z0.s, #0",
                "whilelt p0.s, x8, {n}",

                "1:",
                "ld1w {{z1.s}}, p0/z, [{pa}, x8, lsl #2]",
                "ld1w {{z2.s}}, p0/z, [{pb}, x8, lsl #2]",
                "fsub z1.s, p0/m, z1.s, z2.s",
                "fmla z0.s, p0/m, z1.s, z1.s",
                "incw x8",
                "whilelt p0.s, x8, {n}",
                "b.first 1b",

                "ptrue p0.s",
                "faddv s0, p0, z0.s",
                "str s0, [{pout}]",

                n = in(reg) n as i64,
                pa = in(reg) a.as_ptr(),
                pb = in(reg) b.as_ptr(),
                pout = in(reg) &mut result as *mut f32,
                out("x8") _,
                out("v0") _, out("v1") _, out("v2") _,
                options(nostack),
            );
        }
    }
    result
}

// =============================================================================
// AArch64 NEON implementations (128-bit, 4x unrolled for pipeline saturation)
//
// Processes 16 f32s per loop iteration via 4 independent accumulators to
// hide FMA latency on Neoverse V2's dual ASIMD pipes. Uses LDP for 256-bit
// loads per cycle and FADDP for efficient horizontal reduction.
// =============================================================================

/// Dot product using NEON inline assembly (128-bit, 4x unrolled).
#[cfg(target_arch = "aarch64")]
pub unsafe fn dot_f32_neon(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let n = a.len();
    let chunks = n / 16;
    let remainder = n % 16;
    let mut result: f32 = 0.0;

    if chunks > 0 {
        unsafe {
            asm!(
                "movi v0.4s, #0",
                "movi v1.4s, #0",
                "movi v2.4s, #0",
                "movi v3.4s, #0",

                "1:",
                "ldp q4, q5, [{pa}]",
                "ldp q6, q7, [{pa}, #32]",
                "ldp q16, q17, [{pb}]",
                "ldp q18, q19, [{pb}, #32]",
                "fmla v0.4s, v4.4s, v16.4s",
                "fmla v1.4s, v5.4s, v17.4s",
                "fmla v2.4s, v6.4s, v18.4s",
                "fmla v3.4s, v7.4s, v19.4s",
                "add {pa}, {pa}, #64",
                "add {pb}, {pb}, #64",
                "subs {count}, {count}, #1",
                "b.ne 1b",

                // Reduce 4 accumulators
                "fadd v0.4s, v0.4s, v1.4s",
                "fadd v2.4s, v2.4s, v3.4s",
                "fadd v0.4s, v0.4s, v2.4s",
                // Horizontal sum: v0.4s → s0
                "faddp v0.4s, v0.4s, v0.4s",
                "faddp s0, v0.2s",
                "str s0, [{pout}]",

                pa = inout(reg) a.as_ptr() => _,
                pb = inout(reg) b.as_ptr() => _,
                count = inout(reg) chunks as i64 => _,
                pout = in(reg) &mut result as *mut f32,
                out("v0") _, out("v1") _, out("v2") _, out("v3") _,
                out("v4") _, out("v5") _, out("v6") _, out("v7") _,
                out("v16") _, out("v17") _, out("v18") _, out("v19") _,
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

/// Cosine distance using NEON inline assembly (128-bit, 4x unrolled).
/// Returns 1 - cos(a, b).
#[cfg(target_arch = "aarch64")]
pub unsafe fn cosine_f32_neon(a: &[f32], b: &[f32]) -> f32 {
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
                // 3 sets of 4 accumulators: dot(v0-v3), norm_a(v20-v23), norm_b(v24-v27)
                "movi v0.4s, #0",
                "movi v1.4s, #0",
                "movi v2.4s, #0",
                "movi v3.4s, #0",
                "movi v20.4s, #0",
                "movi v21.4s, #0",
                "movi v22.4s, #0",
                "movi v23.4s, #0",
                "movi v24.4s, #0",
                "movi v25.4s, #0",
                "movi v26.4s, #0",
                "movi v27.4s, #0",

                "1:",
                "ldp q4, q5, [{pa}]",
                "ldp q6, q7, [{pa}, #32]",
                "ldp q16, q17, [{pb}]",
                "ldp q18, q19, [{pb}, #32]",
                // dot += a*b
                "fmla v0.4s, v4.4s, v16.4s",
                "fmla v1.4s, v5.4s, v17.4s",
                "fmla v2.4s, v6.4s, v18.4s",
                "fmla v3.4s, v7.4s, v19.4s",
                // norm_a += a*a
                "fmla v20.4s, v4.4s, v4.4s",
                "fmla v21.4s, v5.4s, v5.4s",
                "fmla v22.4s, v6.4s, v6.4s",
                "fmla v23.4s, v7.4s, v7.4s",
                // norm_b += b*b
                "fmla v24.4s, v16.4s, v16.4s",
                "fmla v25.4s, v17.4s, v17.4s",
                "fmla v26.4s, v18.4s, v18.4s",
                "fmla v27.4s, v19.4s, v19.4s",

                "add {pa}, {pa}, #64",
                "add {pb}, {pb}, #64",
                "subs {count}, {count}, #1",
                "b.ne 1b",

                // Reduce dot
                "fadd v0.4s, v0.4s, v1.4s",
                "fadd v2.4s, v2.4s, v3.4s",
                "fadd v0.4s, v0.4s, v2.4s",
                "faddp v0.4s, v0.4s, v0.4s",
                "faddp s0, v0.2s",
                "str s0, [{pdot}]",
                // Reduce norm_a
                "fadd v20.4s, v20.4s, v21.4s",
                "fadd v22.4s, v22.4s, v23.4s",
                "fadd v20.4s, v20.4s, v22.4s",
                "faddp v20.4s, v20.4s, v20.4s",
                "faddp s20, v20.2s",
                "str s20, [{pna}]",
                // Reduce norm_b
                "fadd v24.4s, v24.4s, v25.4s",
                "fadd v26.4s, v26.4s, v27.4s",
                "fadd v24.4s, v24.4s, v26.4s",
                "faddp v24.4s, v24.4s, v24.4s",
                "faddp s24, v24.2s",
                "str s24, [{pnb}]",

                pa = inout(reg) a.as_ptr() => _,
                pb = inout(reg) b.as_ptr() => _,
                count = inout(reg) chunks as i64 => _,
                pdot = in(reg) &mut dot as *mut f32,
                pna = in(reg) &mut norm_a as *mut f32,
                pnb = in(reg) &mut norm_b as *mut f32,
                out("v0") _, out("v1") _, out("v2") _, out("v3") _,
                out("v4") _, out("v5") _, out("v6") _, out("v7") _,
                out("v16") _, out("v17") _, out("v18") _, out("v19") _,
                out("v20") _, out("v21") _, out("v22") _, out("v23") _,
                out("v24") _, out("v25") _, out("v26") _, out("v27") _,
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

/// Squared Euclidean distance using NEON inline assembly (128-bit, 4x unrolled).
#[cfg(target_arch = "aarch64")]
pub unsafe fn sqeuclidean_f32_neon(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let n = a.len();
    let chunks = n / 16;
    let remainder = n % 16;
    let mut result: f32 = 0.0;

    if chunks > 0 {
        unsafe {
            asm!(
                "movi v0.4s, #0",
                "movi v1.4s, #0",
                "movi v2.4s, #0",
                "movi v3.4s, #0",

                "1:",
                "ldp q4, q5, [{pa}]",
                "ldp q6, q7, [{pa}, #32]",
                "ldp q16, q17, [{pb}]",
                "ldp q18, q19, [{pb}, #32]",
                "fsub v4.4s, v4.4s, v16.4s",
                "fsub v5.4s, v5.4s, v17.4s",
                "fsub v6.4s, v6.4s, v18.4s",
                "fsub v7.4s, v7.4s, v19.4s",
                "fmla v0.4s, v4.4s, v4.4s",
                "fmla v1.4s, v5.4s, v5.4s",
                "fmla v2.4s, v6.4s, v6.4s",
                "fmla v3.4s, v7.4s, v7.4s",
                "add {pa}, {pa}, #64",
                "add {pb}, {pb}, #64",
                "subs {count}, {count}, #1",
                "b.ne 1b",

                "fadd v0.4s, v0.4s, v1.4s",
                "fadd v2.4s, v2.4s, v3.4s",
                "fadd v0.4s, v0.4s, v2.4s",
                "faddp v0.4s, v0.4s, v0.4s",
                "faddp s0, v0.2s",
                "str s0, [{pout}]",

                pa = inout(reg) a.as_ptr() => _,
                pb = inout(reg) b.as_ptr() => _,
                count = inout(reg) chunks as i64 => _,
                pout = in(reg) &mut result as *mut f32,
                out("v0") _, out("v1") _, out("v2") _, out("v3") _,
                out("v4") _, out("v5") _, out("v6") _, out("v7") _,
                out("v16") _, out("v17") _, out("v18") _, out("v19") _,
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
