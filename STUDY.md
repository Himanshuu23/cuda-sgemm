# STUDY.md — understanding the SGEMM ladder

Goal: `C = A * B`, all matrices N x N, fp32, **row-major** (element (r, c) lives at `r*N + c`).
Work: `2*N^3` FLOPs (one multiply + one add per (i, j, k)). Every kernel does the same work; they
differ only in *how data moves*.

No measured numbers appear here (they live in your own `results/`). Where I quote numbers they are
arithmetic derived from the code, or clearly labelled illustrations.

## 0. Vocabulary you need

* **Warp**: 32 threads that execute in lockstep. Threads are numbered inside a block with a flat
  id `tid = threadIdx.x + threadIdx.y*blockDim.x`; consecutive 32 ids form a warp. So in a 2-D block
  **`threadIdx.x` varies fastest across a warp**.
* **Coalescing**: when the 32 lanes of a warp load addresses that fall in few 32-byte sectors /
  128-byte lines, the hardware serves the whole warp with few memory transactions. 32 lanes x 4 B
  contiguous = 128 B = ideal.
* **Shared memory**: fast on-chip memory shared by a block, split in **32 banks** (4 bytes wide). Lanes of a
  warp hitting *different* addresses in the *same* bank are serialised (**bank conflict**). Many lanes reading the
  *same* address is a free broadcast.
* **Arithmetic intensity** = FLOPs per byte moved. Low → memory-bound; high → compute-bound.
* **Occupancy**: resident warps per SM / max warps per SM. Limited by registers/thread, shared memory/block,
  and threads/block. More warps = better latency hiding, but not always faster (see kernel 4).
* **ILP (instruction-level parallelism)**: independent instructions a *single thread* can have in flight.
  A thread with one accumulator does `acc += x*y` serially (each FMA waits for the last); 64 accumulators
  give 64 independent FMAs.
* **Roofline thinking**: attainable perf = min(peak FLOPs, intensity x bandwidth).

---

## Kernel 1 — naive

### What it does
Each thread computes one `C[row][col]` with a loop over k, reading a row of A and a column of B straight from
global memory.

### Thread-index math, line by line
Launch: block `(32,32)`, grid `(N/32, N/32)`.
```cpp
const int row = blockIdx.x * 32 + threadIdx.x;   // (1)
const int col = blockIdx.y * 32 + threadIdx.y;   // (2)
```
1. `blockIdx.x*32` is the first row owned by this block (blocks tile C in 32x32 patches);
   `+ threadIdx.x` picks one of 32 rows inside the patch. **x is the fast thread index, and here it selects the row.**
2. Same for columns using y.

Worked example, block (0,0): lanes 0..31 of warp 0 have `threadIdx.y = 0`, `threadIdx.x = 0..31` → rows 0..31, column 0.
```cpp
acc += A[row * N + k] * B[k * N + col];
```
* `A[row*N + k]`: lane i reads address `i*N + k` → addresses `N` floats (e.g. 8 KB at N=2048) apart →
  **32 different sectors for one load instruction** (uncoalesced).
* `B[k*N + col]`: all lanes read the same address → broadcast, one sector.
* `C[row*N + col] = acc`: stride-N store → uncoalesced.

### Why it is slow / the bottleneck
**Memory-bound, and badly so.** Per FMA the thread issues 2 global loads (8 B for 2 FLOPs = 0.25 FLOP/B before
caching). Caches (L1/L2) rescue some of it — consecutive `k` reuse the same A sector — but the access pattern wastes
transactions, and there is no deliberate reuse. No occupancy or ILP trick can fix a bad access pattern.

### Interview questions
1. **Why is this kernel slow even though every thread does useful work?**
   Because performance is set by memory transactions, not FLOPs: the warp's A loads hit 32 separate sectors, nothing is
   reused across threads of the block, and arithmetic intensity at the global-memory level is ~0.25 FLOP/B.
2. **Which thread dimension is "fast", and why does it matter?**
   `threadIdx.x`: lanes are numbered x-first, so lanes of a warp differ in x. If x maps to the contiguous (column)
   dimension, accesses coalesce; here x maps to rows, so they don't.
3. **How would you recognise this problem in a profiler?**
   High "sectors per request" for global loads, low memory-throughput efficiency, long-scoreboard (waiting on global
   memory) stalls; Nsight Compute's uncoalesced-access rule fires.

---

## Kernel 2 — coalesced

### What changed
Only the thread → (row, col) mapping. The block is now 1-D with 1024 threads.

### Thread-index math, line by line
Launch: block `1024`, grid `(N/32, N/32)`.
```cpp
const int row = blockIdx.y * 32 + threadIdx.x / 32;   // (1)
const int col = blockIdx.x * 32 + threadIdx.x % 32;   // (2)
```
1. `threadIdx.x / 32` = warp number inside the block (0..31) → which of the 32 rows of the C patch. All 32 lanes of one
   warp have the same value → one warp = one row.
2. `threadIdx.x % 32` = lane id → which of 32 columns. Lanes get consecutive columns.

Worked example, block (0,0), thread 37: `row = 37/32 = 1`, `col = 37%32 = 5` → C[1][5]; it is lane 5 of warp 1.
* `A[row*N + k]`: all lanes same address → broadcast.
* `B[k*N + col]`: lanes read consecutive floats → 1 coalesced 128-B request.
* `C[row*N + col]`: coalesced store.
(`blockIdx.y` ↔ row tiles and `blockIdx.x` ↔ column tiles; either assignment works as long as lane id walks along columns.)

### Why it is faster
Same FLOPs, same loads per thread, but each warp-load now needs ~4 sectors (B) / 1 sector (A) instead of up to 32, so
far less memory traffic and wasted bandwidth.

### Bottleneck
Still **memory-bound**: each thread streams `2N` values with no explicit data reuse; blocks re-fetch the same rows/columns from
L1/L2/DRAM. Intensity is unchanged from kernel 1 at the thread level — only efficiency improved.

### Interview questions
1. **What exactly did you change, and did the amount of work change?**
   Only the index mapping (lane id → consecutive columns). FLOPs and per-thread loads are identical; the number of memory *transactions* dropped.
2. **Why is the A access fine here even though every lane reads the same address?**
   Same-address reads within a warp are broadcast: one sector serves all 32 lanes.
3. **Is it still memory-bound? How do you tell?**
   Yes: arithmetic intensity is unchanged, so achieved FLOPs are limited by L1/L2/DRAM throughput. In ncu you'd see memory-pipe
   throughput high relative to SM compute throughput and long-scoreboard stalls.

---

## Kernel 3 — shared-memory tiling

### What changed
A block cooperatively stages 32x32 tiles of A and B in shared memory and every thread then reads operands from there.

### Thread-index math, line by line
Launch: block `(32,32)`, grid `(N/32, N/32)`.
```cpp
const int tx = threadIdx.x, ty = threadIdx.y;
const int row = blockIdx.y*32 + ty;   // C row owned by this thread
const int col = blockIdx.x*32 + tx;   // C column owned by this thread (tx = lane → consecutive)
for (int t = 0; t < N; t += 32) {
    As[ty][tx] = A[row*N + (t + tx)];   // tile of A: rows [blockIdx.y*32 .. +32), cols [t .. t+32)
    Bs[ty][tx] = B[(t + ty)*N + col];   // tile of B: rows [t .. t+32), cols [blockIdx.x*32 .. +32)
    __syncthreads();                    // tiles fully loaded before anyone reads
    for (k = 0; k < 32; ++k) acc += As[ty][k] * Bs[k][tx];
    __syncthreads();                    // everyone done reading before the next load overwrites
}
C[row*N + col] = acc;
```
Each thread loads one element of each tile (32x32 = 1024 threads = 1024 elements), both coalesced since `tx` walks the contiguous dimension.
`As[ty][k]` uses the thread's own row of the A tile; `Bs[k][tx]` its own column of the B tile → `acc` accumulates exactly
`sum_k A[row][k]*B[k][col]`, 32 k's per tile, N/32 tiles.

Bank check: warp = fixed `ty`, `tx = 0..31`. `As[ty][k]` → same address for all lanes (broadcast). `Bs[k][tx]` → 32 consecutive floats = 32 different banks. No conflicts.

### Why it is faster
Each global element is loaded once per block and used 32 times, so global traffic drops ~32x (intensity ≈ 32x higher at the global level; per block-tile
the loop does `32*32*32` FMAs for `2*32*32` loaded floats = 16 FMA per loaded float).

### Bottleneck
Mostly the **shared-memory pipe and synchronisation**: every FMA needs 2 shared loads (1 FMA : 2 LDS), so the load/store pipe saturates before the FMA
units do. Each thread also has a single accumulator → a serial dependency chain (no ILP); latency hiding relies on other warps. Two barriers per tile cost idle time.

### Interview questions
1. **Why is tiling faster if the number of FLOPs is the same?**
   It changes where operands come from: one global load feeds 32 FMAs through shared memory, cutting global traffic ~32x and moving the kernel
   up the roofline.
2. **Why two `__syncthreads()`?**
   First: no thread may read the tile until all threads have written their part (RAW hazard). Second: no thread may overwrite the tile
   in the next iteration while others are still reading it (WAR hazard).
3. **Where might bank conflicts occur and why don't they here?**
   Conflicts need different addresses in the same bank within a warp. `As[ty][k]` is a broadcast; `Bs[k][tx]` is 32 consecutive words → one per bank. (Had you
   read a *column* such as `As[tx][k]` with a width-32 array, all lanes would hit the same bank: 32-way conflict.)

---

## Kernel 4 — 2-D register blocking

### What changed
Each thread now computes an **8x8 block** of C (64 outputs) instead of 1. Block tile is 128x128, K step `BK = 8`, 256 threads.

### Thread-index math, line by line
`BM=BN=128, BK=8, TM=TN=8`. Launch: block `256`, grid `(N/128, N/128)`. 16x16 threads x 8x8 outputs = 128x128.
```cpp
const int tx = tid % 16;   // 0..15  column group
const int ty = tid / 16;   // 0..15  row group
A_blk = A + blockIdx.y*128*N;   // first row of this block's 128-row strip of A
B_blk = B + blockIdx.x*128;     // first column of this block's 128-col strip of B
```
**Loading** (per K step of 8): tiles hold 128x8 = 1024 floats each, 256 threads → 4 elements per thread:
```cpp
for r in 0..3:
  e = tid + r*256;                      // 0..1023, consecutive across threads
  a_m = e / 8,   a_k = e % 8;           // A tile: 8 consecutive threads read one 8-float row segment
  As[a_m][a_k] = A_blk[a_m*N + k0 + a_k];
  b_k = e / 128, b_n = e % 128;         // B tile: 128 consecutive threads read one full row of the tile
  Bs[b_k][b_n] = B_blk[(k0 + b_k)*N + b_n];
```
**Compute:** for each of the 8 `k`:
```cpp
ra[i] = As[i*16 + ty][k];    // thread's 8 A values: rows ty, ty+16, ty+32, ...
rb[j] = Bs[k][j*16 + tx];    // thread's 8 B values: cols tx, tx+16, ...
acc[i][j] += ra[i]*rb[j];    // 64 FMAs (outer product)
```
The interleaved ("stride 16") ownership means in a warp (2 values of `ty`, 16 of `tx`): `Bs[k][j*16+tx]` = 16 consecutive floats (distinct banks),
`As[i*16+ty][k]` = 2 addresses 8 floats apart (distinct banks) → no scalar-read bank conflicts. Write-back uses the same
`row = by*128 + i*16 + ty`, `col = bx*128 + j*16 + tx`, so each warp stores 16-float contiguous runs.

### Why it is faster
16 shared loads now feed 64 FMAs (4:1, vs 1:2 in kernel 3) → the shared-memory pipe is no longer the limit. 64 independent accumulators give ILP so a single warp keeps the FMA units busy. Global traffic also falls: per K-step the block loads `2*128*8` floats for `128*128*8` FMAs.

### Bottleneck
Now closer to **compute-bound**, but held back by: **register pressure** (64 accumulators + fragments + addresses; many registers per thread → fewer resident warps — check *registers/thread* and *achieved occupancy*),
scalar 4-byte loads (many instructions), and **exposed latency** because loads for the next tile aren't overlapped with math (single buffer + two barriers).

*Illustration (not a measurement) of the occupancy arithmetic:* an SM on cc 8.6 has 65,536 registers and at most 48 resident warps (1,536 threads). If a 256-thread block used, say, 128 registers/thread it would need 32,768 registers, so only 2 blocks (16 warps = 33 %) fit. Look up your real register count in the ncu summary and redo the arithmetic.

### Interview questions
1. **Why does computing more outputs per thread help?**
   It raises the FMA : load ratio. For an `TM x TN` tile a thread does `TM*TN` FMAs per `TM+TN` shared loads (64 : 16 here). Data held in registers is reused across
   the whole outer product, and the independent accumulators provide ILP.
2. **Why isn't a bigger register tile (e.g. 16x16) always better?**
   Registers per thread grow (256 accumulators won't fit comfortably under the 255-register limit without spilling to local memory), occupancy drops, and tile sizes must still divide the problem. There is an optimum.
3. **Is lower occupancy a problem here?**
   Not necessarily. Latency can be hidden by ILP inside a thread instead of by many warps ("occupancy is not the goal, throughput is"). It becomes a problem when too few warps remain to cover barrier and memory latency.

---

## Kernel 5 — vectorised + transposed A + padding

### What changed
1. **`float4` global loads/stores** (one 16-byte instruction instead of four).
2. **A tile stored transposed** (`As[k][m]`) so a thread's A fragment is contiguous in shared memory → `float4` shared loads.
3. **Padding** `A_LD = 128 + 4` to kill bank conflicts in the transposing stores.
4. A new thread→output mapping (two groups of 4 contiguous rows/cols per thread) so every `float4` shared read is conflict-free.

### Thread-index math, line by line
```cpp
tx = tid % 16;  ty = tid / 16;          // column group, row group (as before)
a_m = tid / 2;        a_k = (tid % 2) * 4;    // A tile 128x8: 2 float4 per row → 256 float4 = 1 per thread
b_k = tid / 32;       b_n = (tid % 32) * 4;   // B tile 8x128: 32 float4 per row → 256 float4 = 1 per thread
```
* A load: thread `tid` reads 4 consecutive k of row `a_m`: `A_blk[a_m*N + k0 + a_k .. +3]` as one `float4`
  (`a_k` is 0 or 4, so adjacent thread pairs cover the 8-wide K slice).
* Transposed store: `As[(a_k+q)*A_LD + a_m] = av.{x,y,z,w}` for q = 0..3 — element (m, k) goes to row `k`, column `m`.
* B load: `B_blk[(k0+b_k)*N + b_n]` as one `float4` (32 lanes cover 128 contiguous floats = fully coalesced); stored straight into `Bs[b_k*128 + b_n]`.
* **Why pad 4?** In a warp, `a_k` is 0 for even lanes and 4 for odd lanes, `a_m` = 0..15. Scalar store address = `(a_k+q)*A_LD + a_m`. Without padding (A_LD=128) the odd lanes are `4*128` words away → same bank as the even lanes → 2-way conflict.
  With A_LD=132: shift = `4*132 = 528 ≡ 16 (mod 32)` → odd lanes land 16 banks away from even lanes; with `a_m = 0..15` all 32 lanes hit distinct banks. 132 floats = 528 B is a multiple of 16 B so `float4` reads of `As` stay aligned.
* Compute: thread (tx, ty) owns rows `{ty*4 .. +3} ∪ {64+ty*4 .. +3}` and the same pattern for columns:
  ```cpp
  a0 = As[k*A_LD + ty*4 ..+3];   a1 = As[k*A_LD + 64 + ty*4 ..+3];   // two LDS.128
  b0 = Bs[k*128 + tx*4 ..+3];    b1 = Bs[k*128 + 64 + tx*4 ..+3];    // two LDS.128
  ```
  4 vector loads give all 16 operand floats (vs 16 scalar loads in kernel 4). For `b`, a quarter-warp (8 lanes — the group a 128-bit shared load is processed in) reads 8 consecutive
  `float4` = 128 contiguous bytes = each bank once → conflict-free. For `a`, lanes share addresses within a half-warp (broadcast).
* Write-back: row `i<4 → ty*4+i`, `i>=4 → 64+ty*4+(i-4)`; columns `tx*4` and `64+tx*4` → two `float4` stores per row.

### Why it is faster
Same FLOPs and same data reuse as kernel 4, but far fewer memory *instructions* (4x fewer per load/store group), wider transactions, and no shared-memory bank conflicts. That frees issue slots for the FMAs.

### Bottleneck
Closer to compute-bound; what's left versus cuBLAS: single-buffered tiles (loads not overlapped with math), a still-modest warp-level structure (no explicit warp tiling), no hand-tuned instruction scheduling/register-bank-aware allocation. Check ncu stall reasons: barrier and
long-scoreboard (waiting for the global load before the first barrier) are the usual leftovers. Natural next steps: double-buffering / software pipelining, warp tiling, larger BK, `cp.async` (sm_80+).

### Interview questions
1. **What is a shared-memory bank conflict and how did you remove them here?**
   Lanes of a warp accessing different addresses in the same one of 32 banks get serialised. Fixes used: padding the transposed A tile (+4 floats shifts the two lane groups by 16 banks) and arranging each thread's `float4` fragments so a quarter-warp reads 128 contiguous bytes.
2. **Why store A transposed?**
   The inner loop needs, for fixed k, a thread's several A values down a *column* of the tile. Transposed, those are contiguous so they load with vector instructions; also the loads become broadcast-friendly. The one-time cost is the scattered (but padded, conflict-free) stores during the tile fill.
3. **What are the requirements for `float4` loads and what breaks them?**
   16-byte alignment of the address (base pointer from `cudaMalloc` is aligned; row length `N` must be a multiple of 4; shared arrays declared `__align__(16)`, row stride in bytes multiple of 16) and `N` a multiple of the tile — otherwise you need scalar tail handling.

---

## Cross-cutting questions (bonus)

1. **How do you decide whether a kernel is memory- or compute-bound?** Compare achieved GFLOPS to the roofline: intensity x bandwidth vs peak FLOPs; confirm in Nsight Compute (SOL section: memory vs SM throughput, pipe utilisation, stall reasons).
2. **Why does `cublasSgemm` with row-major data use swapped arguments?** cuBLAS is column-major; row-major `C = A*B` is memory-identical to column-major `C^T = B^T * A^T`, so pass B first and A second with the same leading dimension.
3. **Why median of runs and warmup?** Warmup absorbs clock ramp-up, lazy initialisation, and cold caches; the median is robust to outliers from power/thermal changes and OS jitter.
4. **Why compare with a tolerance, not equality?** Floating-point addition isn't associative; different summation orders give ~1e-6 relative differences.
