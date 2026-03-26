// SPDX-License-Identifier: MIT

use std::arch::asm;

/// Result of a 2x2 Singular Value Decomposition.
/// For matrix M, the decomposition is: M = U * S * V^T
/// where U and V are orthogonal and S is diagonal with non-negative entries.
#[derive(Debug, Clone)]
pub struct Svd2x2 {
    /// Left singular vectors (2x2 orthogonal matrix, row-major)
    pub u: [f64; 4],
    /// Singular values (2 values, descending order)
    pub s: [f64; 2],
    /// Right singular vectors (2x2 orthogonal matrix V^T, row-major)
    pub vt: [f64; 4],
}

/// Compute the 2x2 SVD using inline assembly with SSE2 SIMD instructions.
///
/// The algorithm uses the closed-form decomposition:
///   1. Compute E = (m00+m11)/2, F = (m00-m11)/2, G = (m10+m01)/2, H = (m10-m01)/2
///   2. Singular values: s1 = sqrt(E^2 + H^2) + sqrt(F^2 + G^2)
///                       s2 = sqrt(E^2 + H^2) - sqrt(F^2 + G^2)
///   3. Rotation angles via atan2, then M = Rot(angle_u) * diag(s1,s2) * Rot(angle_v)
///
/// The SIMD portions accelerate the paired arithmetic (E,F,G,H computation,
/// squaring, horizontal add) while atan2/sqrt use libm scalars.
#[cfg(target_arch = "x86_64")]
pub fn svd_2x2_simd(m: &[f64; 4]) -> Svd2x2 {
    let mut e: f64 = 0.0;
    let mut f: f64 = 0.0;
    let mut g: f64 = 0.0;
    let mut h: f64 = 0.0;

    // Step 1: Compute E, F, G, H using SSE2 packed double operations.
    unsafe {
        asm!(
            "movsd  {tmp0}, [{src}]",
            "movhpd {tmp0}, [{src} + 16]",   // tmp0 = [m00, m10]
            "movsd  {tmp1}, [{src} + 24]",
            "movhpd {tmp1}, [{src} + 8]",    // tmp1 = [m11, m01]
            "movapd {tmp2}, {tmp0}",
            "addpd  {tmp2}, {tmp1}",         // [m00+m11, m10+m01] = [2E, 2G]
            "subpd  {tmp0}, {tmp1}",         // [m00-m11, m10-m01] = [2F, 2H]
            "movsd  {tmp1}, [{half}]",
            "unpcklpd {tmp1}, {tmp1}",       // [0.5, 0.5]
            "mulpd  {tmp2}, {tmp1}",         // [E, G]
            "mulpd  {tmp0}, {tmp1}",         // [F, H]
            "movsd  [{out_e}], {tmp2}",
            "movhpd [{out_g}], {tmp2}",
            "movsd  [{out_f}], {tmp0}",
            "movhpd [{out_h}], {tmp0}",
            src = in(reg) m.as_ptr(),
            half = in(reg) &0.5_f64,
            out_e = in(reg) &mut e as *mut f64,
            out_f = in(reg) &mut f as *mut f64,
            out_g = in(reg) &mut g as *mut f64,
            out_h = in(reg) &mut h as *mut f64,
            tmp0 = out(xmm_reg) _,
            tmp1 = out(xmm_reg) _,
            tmp2 = out(xmm_reg) _,
            options(nostack),
        );
    }

    // Step 2: Compute squared magnitudes using SSE2 horizontal add pattern.
    let mut q1: f64 = 0.0; // E^2 + H^2
    let mut q2: f64 = 0.0; // F^2 + G^2

    unsafe {
        asm!(
            "movsd  {tmp0}, [{pe}]",
            "movhpd {tmp0}, [{ph}]",         // [E, H]
            "mulpd  {tmp0}, {tmp0}",         // [E^2, H^2]
            "movsd  {tmp1}, [{pf}]",
            "movhpd {tmp1}, [{pg}]",         // [F, G]
            "mulpd  {tmp1}, {tmp1}",         // [F^2, G^2]
            "movapd {tmp2}, {tmp0}",
            "unpckhpd {tmp2}, {tmp2}",
            "addsd  {tmp0}, {tmp2}",         // E^2 + H^2
            "movapd {tmp2}, {tmp1}",
            "unpckhpd {tmp2}, {tmp2}",
            "addsd  {tmp1}, {tmp2}",         // F^2 + G^2
            "movsd  [{pq1}], {tmp0}",
            "movsd  [{pq2}], {tmp1}",
            pe = in(reg) &e as *const f64,
            pf = in(reg) &f as *const f64,
            pg = in(reg) &g as *const f64,
            ph = in(reg) &h as *const f64,
            pq1 = in(reg) &mut q1 as *mut f64,
            pq2 = in(reg) &mut q2 as *mut f64,
            tmp0 = out(xmm_reg) _,
            tmp1 = out(xmm_reg) _,
            tmp2 = out(xmm_reg) _,
            options(nostack),
        );
    }

    let sq1 = q1.sqrt();
    let sq2 = q2.sqrt();
    let s1 = sq1 + sq2;
    let s2 = sq1 - sq2;

    // Step 3: Rotation angles
    let theta = g.atan2(f);
    let phi = h.atan2(e);
    let angle_u = (phi + theta) / 2.0;
    let angle_v = (phi - theta) / 2.0;

    // Step 4: Build rotation matrices
    let (sin_u, cos_u) = angle_u.sin_cos();
    let (sin_v, cos_v) = angle_v.sin_cos();

    let mut u = [0.0_f64; 4];
    let mut vt = [0.0_f64; 4];

    unsafe {
        asm!(
            // U = Rot(angle_u): [cos_u, -sin_u, sin_u, cos_u]
            "movsd   {tmp0}, [{pcos_u}]",
            "movsd   {tmp1}, [{psin_u}]",
            "movsd   {tmp2}, [{pneg}]",
            "xorpd   {tmp2}, {tmp1}",        // -sin_u
            "movsd   [{pu}],      {tmp0}",
            "movsd   [{pu} + 8],  {tmp2}",
            "movsd   [{pu} + 16], {tmp1}",
            "movsd   [{pu} + 24], {tmp0}",
            // V^T = Rot(angle_v): [cos_v, -sin_v, sin_v, cos_v]
            "movsd   {tmp0}, [{pcos_v}]",
            "movsd   {tmp1}, [{psin_v}]",
            "movsd   {tmp2}, [{pneg}]",
            "xorpd   {tmp2}, {tmp1}",        // -sin_v
            "movsd   [{pvt}],      {tmp0}",
            "movsd   [{pvt} + 8],  {tmp2}",
            "movsd   [{pvt} + 16], {tmp1}",
            "movsd   [{pvt} + 24], {tmp0}",
            pcos_u = in(reg) &cos_u,
            psin_u = in(reg) &sin_u,
            pcos_v = in(reg) &cos_v,
            psin_v = in(reg) &sin_v,
            pneg = in(reg) &(-0.0_f64),
            pu = in(reg) u.as_mut_ptr(),
            pvt = in(reg) vt.as_mut_ptr(),
            tmp0 = out(xmm_reg) _,
            tmp1 = out(xmm_reg) _,
            tmp2 = out(xmm_reg) _,
            options(nostack),
        );
    }

    // Sign fixup: make singular values non-negative
    let mut s = [s1, s2];
    if s[0] < 0.0 {
        s[0] = -s[0];
        u[0] = -u[0];
        u[2] = -u[2];
    }
    if s[1] < 0.0 {
        s[1] = -s[1];
        u[1] = -u[1];
        u[3] = -u[3];
    }

    Svd2x2 { u, s, vt }
}

/// Pure Rust (no inline asm) 2x2 SVD for comparison benchmarking.
pub fn svd_2x2_scalar(m: &[f64; 4]) -> Svd2x2 {
    let (a, b, c, d) = (m[0], m[1], m[2], m[3]);

    let e = (a + d) / 2.0;
    let f = (a - d) / 2.0;
    let g = (c + b) / 2.0;
    let h = (c - b) / 2.0;

    let q1 = (e * e + h * h).sqrt();
    let q2 = (f * f + g * g).sqrt();

    let s1 = q1 + q2;
    let s2 = q1 - q2;

    let theta = g.atan2(f);
    let phi = h.atan2(e);
    let angle_u = (phi + theta) / 2.0;
    let angle_v = (phi - theta) / 2.0;

    let (sin_u, cos_u) = angle_u.sin_cos();
    let (sin_v, cos_v) = angle_v.sin_cos();

    let mut u = [cos_u, -sin_u, sin_u, cos_u];
    let vt = [cos_v, -sin_v, sin_v, cos_v];
    let mut s = [s1, s2];

    if s[0] < 0.0 {
        s[0] = -s[0];
        u[0] = -u[0];
        u[2] = -u[2];
    }
    if s[1] < 0.0 {
        s[1] = -s[1];
        u[1] = -u[1];
        u[3] = -u[3];
    }

    Svd2x2 { u, s, vt }
}
