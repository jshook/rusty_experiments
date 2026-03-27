# SIMD Microbenchmark Results — AArch64 (Neoverse V2 / Graviton 4)

## System Under Test

| Property | Value |
|----------|-------|
| **Instance** | AWS Graviton 4 (aarch64) |
| **CPU** | ARM Neoverse V2 r0p1 (MIDR `0xd4f`) |
| **Cores** | 8 (1 socket, 1 thread/core — no SMT) |
| **Clock** | ~2.0 GHz (BogoMIPS 2000.00) |
| **L1d / L1i** | 64 KiB per core (512 KiB total) |
| **L2** | 2 MiB per core (16 MiB total) |
| **L3** | 36 MiB shared |
| **Memory** | 61 GiB |
| **SVE vector length** | 128 bits (16 bytes) |
| **ISA extensions** | ASIMD, SVE, SVE2, SVE-AES, SVE-BitPerm, SVE-SHA3, I8MM, BF16, AES, SHA-512, CRC32, Atomics (LSE), BTI, PAC, DIT, FlagM2, FRINT, RNG |
| **Kernel** | Linux 6.17.0-1007-aws aarch64 |
| **Toolchain** | Rust 1.94.1 stable, `RUSTFLAGS='-C target-cpu=native'` |
| **Benchmark harness** | Criterion 0.5 |

## Implementations

| Label | Description |
|-------|-------------|
| **neon_asm** | Hand-written NEON inline asm, 4× unrolled with LDP load pairs and 4 independent FMA accumulators to saturate dual ASIMD/FP pipes |
| **sve_asm** | Hand-written SVE inline asm, VL-agnostic WHILELT/predication loop (no scalar tail), FADDV horizontal reduction |
| **simsimd** | [SimSIMD](https://github.com/ashvardanian/SimSIMD) v6 — portable SIMD distance library |
| **scalar** | Pure Rust iterator chains, no intrinsics or inline asm |

---

## Dot Product (`f32`)

| Size | NEON 4× (ns) | SVE (ns) | simsimd (ns) | scalar (ns) | NEON vs scalar |
|-----:|--------------:|---------:|-------------:|-----------:|:--------------:|
|   64 |       **7.0** |      7.9 |         10.0 |       40.0 |       **5.7×** |
|  256 |      **20.7** |     26.1 |         37.5 |        157 |       **7.6×** |
| 1024 |      **68.1** |     95.0 |          165 |        639 |       **9.4×** |
| 4096 |       **308** |      373 |          716 |       2559 |       **8.3×** |

## Cosine Distance (`f32`, returns `1 − cos(a,b)`)

| Size | NEON 4× (ns) | SVE (ns) | simsimd (ns) | scalar (ns) | NEON vs scalar |
|-----:|--------------:|---------:|-------------:|-----------:|:--------------:|
|   64 |      **17.4** |     27.3 |         17.8 |        104 |       **6.0×** |
|  256 |      **40.1** |     53.1 |         55.7 |        418 |      **10.4×** |
| 1024 |       **112** |      192 |          193 |       1669 |      **14.9×** |
| 4096 |       **385** |      742 |          744 |       6674 |      **17.3×** |

## Squared Euclidean Distance (`f32`)

| Size | NEON 4× (ns) | SVE (ns) | simsimd (ns) | scalar (ns) | NEON vs scalar |
|-----:|--------------:|---------:|-------------:|-----------:|:--------------:|
|   64 |       **7.8** |      9.9 |         10.6 |       39.6 |       **5.1×** |
|  256 |      **25.4** |     38.8 |         39.8 |        158 |       **6.2×** |
| 1024 |      **82.7** |      174 |          175 |        640 |       **7.7×** |
| 4096 |       **329** |      725 |          725 |       2559 |       **7.8×** |

## 2×2 SVD (`f64`)

| Variant | Time (ns) |
|---------|----------:|
| NEON inline asm | 64.5 |
| Pure scalar     | 59.1 |

The SVD is dominated by transcendentals (`atan2`, `sin_cos`, `sqrt`) — the NEON asm for the linear-algebra portion doesn't offset the extra load/store overhead at this tiny matrix size.

---

## Analysis

### Why NEON 4× wins

Neoverse V2 has **two 128-bit ASIMD/FP pipes**, each capable of issuing one FMLA per cycle with 4-cycle latency.
To keep both pipes fully occupied:

- **4 independent accumulators** (`v0`–`v3`) break the dependency chain, giving the out-of-order engine 4 × 2 = 8 in-flight FMAs — enough to hide the 4-cycle latency on both pipes.
- **`LDP` (load pair)** feeds 256 bits (8 floats) per cycle from L1, matching the two pipes' consumption rate.
- **`FADDP` horizontal reduction** collapses the final 4×4 = 16 partial sums in 3 instructions.

### Why SVE is slower here

On this hardware the SVE vector length is **128 bits** — identical to NEON. SVE's advantages (VL-agnostic loops, predicated tails, single-instruction `FADDV` reduction) don't offset:

- Per-iteration overhead of `WHILELT` + `INCW` vs a simple counted loop.
- Single-accumulator loop body (no unroll) — one dependency chain limits throughput to 1 FMA per 4 cycles instead of 2 per cycle.

On wider SVE hardware (256-bit Neoverse V1, 512-bit A64FX), the same SVE code would process 2–4× more data per iteration and likely match or beat the fixed-width NEON version.

### NEON vs simsimd

Our hand-rolled NEON asm beats simsimd by **1.4–2×** across the board. simsimd's generic NEON path likely uses fewer accumulators and doesn't exploit `LDP` for paired 128-bit loads, leaving throughput on the table.
